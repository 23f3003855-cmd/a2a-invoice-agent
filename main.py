"""GA5 A2A Invoice Action Agent (A2A 1.0, HTTP+JSON binding)."""
import base64
import datetime
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from decimal import Decimal

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

DB_PATH = os.getenv("DB_PATH", "ga5.db")
LOG_PATH = os.getenv("LOG_PATH", "ga5_requests.log")
LLM_BASE = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai").rstrip("/")
LLM_KEY = os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash,gemini-3.1-flash-lite")
LLM_EFFORT = os.getenv("LLM_EFFORT", "low")
KEEP_BRACKETS = os.getenv("KEEP_BRACKETS", "0") == "1"
CACHE_VER = "v2"

MEDIA = "application/a2a+json"
BATCH_MT = "application/vnd.ga5.invoice-claim-batch+json"
PROP_MT = "application/vnd.ga5.invoice-action-proposals+json"
RES_MT = "application/vnd.ga5.invoice-action-results+json"
RCPT_MT = "application/vnd.ga5.invoice-action-receipts+json"
ACTIONS = ["settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception"]
INPUT_REQ, COMPLETED, CANCELED = "TASK_STATE_INPUT_REQUIRED", "TASK_STATE_COMPLETED", "TASK_STATE_CANCELED"

app = FastAPI()

# ---------------- storage ----------------
db = sqlite3.connect(DB_PATH, check_same_thread=False, isolation_level=None)
db.execute("PRAGMA journal_mode=WAL")
dblock = threading.RLock()
for stmt in (
    "CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, principal TEXT, context_id TEXT, "
    "batch_id TEXT, state TEXT, task TEXT, proposals TEXT, updated REAL)",
    "CREATE TABLE IF NOT EXISTS msgs (principal TEXT, message_id TEXT, msg_hash TEXT, task_id TEXT, "
    "PRIMARY KEY (principal, message_id))",
    "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, decision TEXT)",
):
    db.execute(stmt)


def qone(sql, args=()):
    with dblock:
        return db.execute(sql, args).fetchone()


def qall(sql, args=()):
    with dblock:
        return db.execute(sql, args).fetchall()


def run_tx(stmts):
    with dblock:
        db.execute("BEGIN IMMEDIATE")
        try:
            for sql, args in stmts:
                db.execute(sql, args)
            db.execute("COMMIT")
        except Exception:
            db.execute("ROLLBACK")
            raise


_locks, _meta = {}, threading.Lock()


def klock(key):
    with _meta:
        if key not in _locks:
            _locks[key] = threading.RLock()
        return _locks[key]


def canon(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def now_ts():
    d = datetime.datetime.now(datetime.timezone.utc)
    return d.strftime("%Y-%m-%dT%H:%M:%S.") + f"{d.microsecond // 1000:03d}Z"


_log_lock, _full_logs = threading.Lock(), [0]


def log_event(kind, payload):
    try:
        with _log_lock, open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": now_ts(), "kind": kind, "data": payload}, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ---------------- HTTP helpers ----------------
STATUS_NAMES = {400: "INVALID_ARGUMENT", 401: "UNAUTHENTICATED", 403: "PERMISSION_DENIED",
                404: "NOT_FOUND", 409: "ABORTED", 415: "INVALID_ARGUMENT", 503: "UNAVAILABLE"}


def err(status, reason, message):
    body = {"error": {"code": status, "status": STATUS_NAMES.get(status, "UNKNOWN"), "message": message,
                      "details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo",
                                   "reason": reason, "domain": "a2a-protocol.org"}]}}
    headers = {"A2A-Version": "1.0"}
    if status == 401:
        headers["WWW-Authenticate"] = "Bearer"
    return JSONResponse(body, status_code=status, media_type=MEDIA, headers=headers)


def respond(body, status=200):
    return JSONResponse(body, status_code=status, media_type=MEDIA, headers={"A2A-Version": "1.0"})


def gate(request, needs_media):
    m = re.fullmatch(r"Bearer\s+(\S+)", request.headers.get("authorization", "").strip(), re.I)
    if not m:
        return None, err(401, "UNAUTHENTICATED", "A valid Bearer token is required.")
    if request.headers.get("a2a-version", "").strip() != "1.0":
        return None, err(400, "VERSION_NOT_SUPPORTED", "Only A2A-Version 1.0 is supported.")
    if needs_media:
        ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if ctype != MEDIA:
            return None, err(415, "CONTENT_TYPE_NOT_SUPPORTED", "Content-Type must be application/a2a+json.")
    return sha(m.group(1)), None


def public_base(request):
    if os.getenv("PUBLIC_BASE_URL"):
        return os.getenv("PUBLIC_BASE_URL").rstrip("/") + "/"
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or "localhost"
    proto = (request.headers.get("x-forwarded-proto") or request.url.scheme).split(",")[0].strip()
    if not host.startswith(("localhost", "127.0.0.1")):
        proto = "https"
    return f"{proto}://{host}/a2a/"


# ---------------- AI decision step ----------------
RULES = """You are a careful accounts-payable controller. For EACH invoice package below choose exactly ONE action and cite the evidence.

Actions:
- settle_invoice: the invoice is valid, reconciled with the records, and within autonomous authority, so it can be paid.
- request_approval: commercially valid, but outside delegated authority (for example the amount is above the limit), so approval is needed first.
- hold_invoice: payment pauses until a stated verification completes.
- reject_duplicate: the SAME commercial invoice was already paid. A payment for a different invoice (even same vendor or same amount) is NOT a duplicate.
- open_exception: material records conflict and an exception workflow is needed.

How to read the documents:
- They mix useful facts with old examples, archive cases, training decoys, negated sentences ("no hold is needed", "this is not a duplicate") and irrelevant action words. Decide only from the CURRENT authoritative case paragraph. Ignore archive examples, training material and templates. Respect negation and the current policy.
- The decisive paragraph is the one that states the facts that determine the action. It carries exactly three bracketed references. Return exactly those three references, copied verbatim from the text, in order of appearance. Never return the cover-sheet reference, archive-example references or training-decoy references.
- Choose settle_invoice ONLY if the decisive paragraph positively says the totals reconcile, goods were received, there is no earlier payment, and the amount is within authority. If it mentions an amount above a limit or ceiling, a changed bank account or pending callback, an earlier payment, or a conflict between records, choose the matching non-settle action instead.
- facts: vendorName (string), invoiceNumber (string), amountMinor (INTEGER in minor currency units, for example 1,234.50 INR is 123450; use the amount of the current invoice), currency (ISO code such as INR).
- rationale: 1 to 3 sentences, at most 400 characters, naming the action and why.

Reply with JSON only, in this exact shape, one decision per package in the same order:
{"decisions":[{"packageId":"...","action":"one of the five actions","facts":{"vendorName":"...","invoiceNumber":"...","amountMinor":0,"currency":"INR"},"evidenceRefs":["...","...","..."],"rationale":"..."}]}
"""

_models = []


def candidate_models():
    if LLM_MODEL:
        return [m.strip() for m in LLM_MODEL.split(",") if m.strip()]
    if _models:
        return _models
    found = []
    try:
        r = httpx.get(f"{LLM_BASE}/models", headers={"Authorization": f"Bearer {LLM_KEY}"}, timeout=10)
        ids = [(m.get("id") or "").replace("models/", "") for m in r.json().get("data", [])]
        bad = ("lite", "image", "tts", "audio", "live", "embed", "vision", "8b", "exp", "thinking", "robotics")
        flash = [i for i in ids if "flash" in i and not any(b in i for b in bad)]
        flash.sort(key=lambda i: [float(x) for x in re.findall(r"\d+(?:\.\d+)?", i)], reverse=True)
        found = flash[:3]
    except Exception:
        pass
    _models.extend(found or ["gemini-2.5-flash", "gemini-2.0-flash"])
    return _models


def parse_json(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except Exception:
        return json.loads(text[text.index("{"): text.rindex("}") + 1])


def llm_call(prompt, deadline):
    if not LLM_KEY:
        raise RuntimeError("No LLM key configured")
    last = None
    for model in candidate_models():
        for use_json_mode in (True, False):
            remaining = deadline - time.time()
            if remaining < 3:
                raise RuntimeError(f"Out of time ({last})")
            body = {"model": model, "temperature": 0,
                    "messages": [{"role": "system", "content": "Reply with JSON only."},
                                 {"role": "user", "content": prompt}]}
            if LLM_EFFORT and LLM_EFFORT != "none":
                body["reasoning_effort"] = LLM_EFFORT
            if use_json_mode:
                body["response_format"] = {"type": "json_object"}
            try:
                r = httpx.post(f"{LLM_BASE}/chat/completions", json=body, timeout=min(25, remaining - 1),
                               headers={"Authorization": f"Bearer {LLM_KEY}"})
                if r.status_code == 200:
                    return parse_json(r.json()["choices"][0]["message"]["content"])
                last = f"{model}: HTTP {r.status_code} {r.text[:200]}"
                if r.status_code != 400:
                    break
            except Exception as e:
                last = f"{model}: {e}"
                break
    raise RuntimeError(f"All models failed: {last}")


def to_minor(v):
    if isinstance(v, bool):
        return 0
    if isinstance(v, (int, float)):
        return int(round(v))
    try:
        return int(round(float(re.sub(r"[^\d.\-]", "", str(v)))))
    except Exception:
        return 0


def norm_ref(r):
    r = str(r).strip()
    if r.startswith("[") and r.endswith("]"):
        r = r[1:-1].strip()
    return r


def normalize(d, pkg):
    if not isinstance(d, dict) or d.get("action") not in ACTIONS:
        return None
    found = list(dict.fromkeys(re.findall(r"\[([^\[\]]+)\]", canon(pkg))))
    refs = []
    for r in d.get("evidenceRefs") or []:
        r = norm_ref(r)
        if r not in found:
            m = [f for f in found if f.casefold() == r.casefold()] or \
                [f for f in found if r and (r in f or f in r)]
            if m:
                r = m[0]
        if r and r not in refs:
            refs.append(r)
    f = d.get("facts") if isinstance(d.get("facts"), dict) else {}
    facts = {"vendorName": str(f.get("vendorName") or ""), "invoiceNumber": str(f.get("invoiceNumber") or ""),
             "amountMinor": to_minor(f.get("amountMinor")), "currency": str(f.get("currency") or "INR").upper()}
    return {"action": d["action"], "facts": facts, "refs": refs[:3], "why": str(d.get("rationale") or "")}


def pkg_key(p):
    return CACHE_VER + ":" + sha(canon({k: v for k, v in p.items() if k != "packageId"}))


def decide_batch(packages, deadline):
    out, need, last = {}, [], None
    for p in packages:
        row = qone("SELECT decision FROM cache WHERE key=?", (pkg_key(p),))
        if row:
            out[p["packageId"]] = json.loads(row[0])
        else:
            need.append(p)
    for _ in range(2):
        if not need:
            break
        try:
            prompt = RULES + "\nPACKAGES (JSON):\n" + json.dumps(need, ensure_ascii=False, separators=(",", ":"))
            raw = llm_call(prompt, deadline)
        except Exception as e:
            last = e
            break
        items = raw.get("decisions") if isinstance(raw, dict) else raw
        items = items if isinstance(items, list) else []
        by_id = {i.get("packageId"): i for i in items if isinstance(i, dict)}
        still = []
        for idx, p in enumerate(need):
            d = by_id.get(p["packageId"]) or (items[idx] if len(items) == len(need) and idx < len(items) else None)
            n = normalize(d, p)
            if n:
                out[p["packageId"]] = n
                run_tx([("INSERT OR REPLACE INTO cache VALUES (?,?)", (pkg_key(p), json.dumps(n)))])
            else:
                still.append(p)
        need = still
    if need and not out:
        raise RuntimeError(f"Model failed: {last}")
    for p in need:  # safe fallback for anything the model could not decide
        refs = list(dict.fromkeys(re.findall(r"\[([^\[\]]+)\]", canon(p))))[:3]
        out[p["packageId"]] = {"action": "open_exception", "refs": refs, "why": "The records could not be reconciled automatically.",
                               "facts": {"vendorName": "", "invoiceNumber": "", "amountMinor": 0, "currency": "INR"}}
    return out


# ---------------- pattern guard (safety net over the model) ----------------
def _paras(p):
    for doc in (p.get("documents") or []) if isinstance(p, dict) else []:
        if isinstance(doc, dict):
            for para in str(doc.get("text", "")).split("\n\n"):
                yield para


def guard(p, n):
    n = dict(n)
    try:
        decisive = []
        for para in _paras(p):
            refs = re.findall(r"\[([^\[\]]+)\]", para)
            if len(refs) == 3 and "archive note" not in para.lower():
                decisive.append((para, refs))
        if len(decisive) == 1:
            para, refs = decisive[0]
            t = para.lower()
            n["refs"] = refs
            act = None
            if any(k in t for k in ("named financial approver", "named approver", "exceeds", "below the stated total")):
                act = "request_approval"
            elif any(k in t for k in ("exception case", "exception workflow", "exception queue",
                                      "outside the permitted", "contradictory", "incompatible")):
                act = "open_exception"
            elif any(k in t for k in ("second disbursement", "earlier settled entry",
                                      "exact commercial duplicate", "earlier posting for the same")):
                act = "reject_duplicate"
            elif any(k in t for k in ("callback", "out-of-band", "beneficiary",
                                      "destination-account", "bank account")):
                act = "hold_invoice"
            if act and act != n.get("action"):
                n["action"] = act
                n["why"] = "The decisive paragraph supports this action."
        text = " ".join(_paras(p))
        m = re.search(r"stated total ([A-Z]{3}) ([\d,]+(?:\.\d+)?)", text)
        v = re.search(r"Supplier (.+?); invoice ([^;]+);", text)
        if m and v:
            n["facts"] = {"vendorName": v.group(1).strip(), "invoiceNumber": v.group(2).strip(),
                          "amountMinor": int((Decimal(m.group(2).replace(",", "")) * 100).to_integral_value()),
                          "currency": m.group(1)}
    except Exception:
        pass
    return n


def make_rationale(action, why, refs):
    base = " ".join(why.split())[:900]
    tag = ", ".join(f"[{r}]" for r in refs)
    s = f"Action {action}: {base} Evidence refs: {tag}."
    if len(s) < 60:
        s += " The decisive paragraph and the cited references support this action."
    return s[:1500]


# ---------------- task helpers ----------------
def view(task, hist=None, artifacts=True):
    t = dict(task)
    if hist is not None and "history" in t:
        if hist == 0:
            t.pop("history")
        elif hist > 0:
            t["history"] = t["history"][-hist:]
    if not artifacts:
        t.pop("artifacts", None)
    return t


def to_int(v, default=None):
    try:
        return int(v)
    except Exception:
        return default


def find_part(msg, mediatype):
    for p in msg["parts"]:
        if isinstance(p, dict) and p.get("mediaType") == mediatype and isinstance(p.get("data"), dict):
            return p["data"], None
    return None, err(400, "CONTENT_TYPE_NOT_SUPPORTED", "The message has no supported part.")


def save_task_stmts(task, principal, batch_id, proposals, insert):
    state = task["status"]["state"]
    if insert:
        return [("INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?)",
                 (task["id"], principal, task["contextId"], batch_id, state, json.dumps(task),
                  json.dumps(proposals), time.time()))]
    return [("UPDATE tasks SET state=?, task=?, updated=? WHERE id=?",
             (state, json.dumps(task), time.time(), task["id"]))]


# ---------------- A2A operations ----------------
def handle_send(principal, req, msg):
    mid, mh = msg["messageId"], sha(canon(msg))
    cfg = req.get("configuration") if isinstance(req.get("configuration"), dict) else {}
    hist = to_int(cfg.get("historyLength"))
    with klock(("m", principal, mid)):
        row = qone("SELECT msg_hash, task_id FROM msgs WHERE principal=? AND message_id=?", (principal, mid))
        if row:
            if row[0] != mh:
                return err(409, "IDEMPOTENCY_CONFLICT", "IDEMPOTENCY_CONFLICT: this messageId was already used with different content.")
            t = qone("SELECT task FROM tasks WHERE id=?", (row[1],))
            return respond({"task": view(json.loads(t[0]), hist)})
        if msg.get("taskId"):
            return handle_continuation(principal, msg, mh, hist)
        return handle_initial(principal, msg, mh, hist)


def handle_initial(principal, msg, mh, hist):
    data, e = find_part(msg, BATCH_MT)
    if e:
        return e
    batch_id, pkgs = data.get("batchId"), data.get("packages")
    if not isinstance(batch_id, str) or not batch_id or not isinstance(pkgs, list) or not pkgs:
        return err(400, "INVALID_BATCH", "The batch needs a batchId and packages.")
    ids = [p.get("packageId") if isinstance(p, dict) else None for p in pkgs]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        return err(400, "INVALID_BATCH", "Every package needs a unique packageId.")
    if _full_logs[0] < 2:
        _full_logs[0] += 1
        log_event("batch_received", data)
    try:
        decisions = decide_batch(pkgs, time.time() + 40)
    except Exception as ex:
        log_event("model_failure", str(ex)[:500])
        print("MODEL_FAILURE", str(ex)[:300], flush=True)
        return err(503, "MODEL_UNAVAILABLE", "Temporary failure, please retry.")
    proposals = []
    for p in pkgs:
        d = guard(p, decisions[p["packageId"]])
        print("DECISION", p["packageId"][-6:], d["action"], d["refs"], flush=True)
        proposals.append({
            "packageId": p["packageId"], "actionId": "act-" + uuid.uuid4().hex[:20], "action": d["action"],
            "facts": d["facts"], "evidenceRefs": [f"[{r}]" if KEEP_BRACKETS else r for r in d["refs"]],
            "rationale": make_rationale(d["action"], d["why"], d["refs"])})
    task = {"id": str(uuid.uuid4()), "contextId": str(msg.get("contextId") or uuid.uuid4()),
            "status": {"state": INPUT_REQ, "timestamp": now_ts()},
            "artifacts": [{"artifactId": "proposals", "name": "invoice-action-proposals",
                           "parts": [{"mediaType": PROP_MT, "data": {"batchId": batch_id, "proposals": proposals}}]}],
            "history": [msg]}
    run_tx(save_task_stmts(task, principal, batch_id, proposals, True) +
           [("INSERT INTO msgs VALUES (?,?,?,?)", (principal, msg["messageId"], mh, task["id"]))])
    log_event("proposals", [{k: p[k] for k in ("packageId", "action", "facts", "evidenceRefs")} for p in proposals])
    return respond({"task": view(task, hist)})


def handle_continuation(principal, msg, mh, hist):
    tid = msg["taskId"]
    with klock(("t", tid)):
        row = qone("SELECT principal, context_id, batch_id, state, task, proposals FROM tasks WHERE id=?", (tid,))
        if not row or row[0] != principal:
            return err(404, "TASK_NOT_FOUND", "The task was not found or is not accessible.")
        if msg.get("contextId") is not None and msg.get("contextId") != row[1]:
            return err(400, "CONTEXT_MISMATCH", "The contextId does not match the task.")
        if row[3] != INPUT_REQ:
            return err(409, "TASK_NOT_ACCEPTING_INPUT", "The task is not accepting input.")
        data, e = find_part(msg, RES_MT)
        if e:
            return e
        if data.get("batchId") != row[2]:
            return err(400, "BATCH_MISMATCH", "The batchId does not match the task.")
        stored = {p["packageId"]: p for p in json.loads(row[5])}
        results = data.get("results")
        if not isinstance(results, list):
            return err(400, "INVALID_RESULTS", "results must be a list.")
        by = {}
        for r in results:
            pid = r.get("packageId") if isinstance(r, dict) else None
            if not isinstance(pid, str) or pid in by or pid not in stored:
                return err(400, "PACKAGE_MISMATCH", "A result does not match a proposal.")
            st = stored[pid]
            if r.get("actionId") != st["actionId"] or r.get("action") != st["action"]:
                return err(400, "ACTION_MISMATCH", "A result does not match its proposal.")
            if r.get("outcome") not in ("ACCEPTED", "REJECTED"):
                return err(400, "INVALID_RESULTS", "Unknown outcome.")
            if r["outcome"] == "ACCEPTED" and not (isinstance(r.get("receiptNonce"), str) and r["receiptNonce"].strip()):
                return err(400, "INVALID_RESULTS", "An accepted result needs a receiptNonce.")
            by[pid] = r
        if set(by) != set(stored):
            return err(400, "PACKAGE_MISMATCH", "Results must cover every proposal exactly once.")
        executions = []
        for pid, st in stored.items():
            if by[pid]["outcome"] == "ACCEPTED":
                executions.append({"packageId": pid, "actionId": st["actionId"], "action": st["action"],
                                   "receiptNonce": by[pid]["receiptNonce"], "facts": st["facts"],
                                   "evidenceRefs": st["evidenceRefs"]})
        task = json.loads(row[4])
        task["artifacts"].append({"artifactId": "receipts", "name": "invoice-action-receipts",
                                  "parts": [{"mediaType": RCPT_MT, "data": {"batchId": row[2], "executions": executions}}]})
        task["history"].append(msg)
        task["status"] = {"state": COMPLETED, "timestamp": now_ts()}
        run_tx(save_task_stmts(task, principal, row[2], None, False) +
               [("INSERT INTO msgs VALUES (?,?,?,?)", (principal, msg["messageId"], mh, tid))])
        return respond({"task": view(task, hist)})


def handle_cancel(principal, tid):
    with klock(("t", tid)):
        row = qone("SELECT principal, state, task FROM tasks WHERE id=?", (tid,))
        if not row or row[0] != principal:
            return err(404, "TASK_NOT_FOUND", "The task was not found or is not accessible.")
        if row[1] != INPUT_REQ:
            return err(409, "TASK_NOT_CANCELABLE", "The task can no longer be canceled.")
        task = json.loads(row[2])
        task["status"] = {"state": CANCELED, "timestamp": now_ts()}
        run_tx(save_task_stmts(task, principal, None, None, False))
        return respond(task)


# ---------------- routes ----------------
def agent_card(request):
    base = public_base(request)
    modes_in = [BATCH_MT, RES_MT, "application/json"]
    modes_out = [PROP_MT, RCPT_MT]
    return {
        "name": "GA5 Invoice Action Agent",
        "description": "Reads invoice claim batches, proposes one typed action per package with cited evidence, and records receipts only for accepted proposals.",
        "version": "1.0.0",
        "supportedInterfaces": [
            {"url": base, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"},
            {"url": base.rstrip("/"), "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}],
        "capabilities": {"streaming": False, "pushNotifications": False, "extendedAgentCard": False},
        "securitySchemes": {"bearer": {"httpAuthSecurityScheme": {"scheme": "Bearer"}}},
        "securityRequirements": [{"schemes": {"bearer": {"list": []}}}],
        "defaultInputModes": modes_in, "defaultOutputModes": modes_out,
        "skills": [{"id": "invoice_action_agent", "name": "Invoice action agent",
                    "description": "Chooses settle, approval, hold, duplicate-rejection or exception actions for invoice packages and cites the supporting evidence.",
                    "tags": ["invoice", "payments", "approval", "duplicate-detection", "evidence"],
                    "inputModes": modes_in, "outputModes": modes_out}]}


@app.get("/.well-known/agent-card.json")
@app.get("/a2a/.well-known/agent-card.json")
def card(request: Request):
    return JSONResponse(agent_card(request), headers={"Cache-Control": "no-store"})


@app.api_route("/", methods=["GET", "HEAD"])
def health():
    return {"status": "ok"}


@app.post("/a2a/message:send")
async def message_send(request: Request):
    principal, e = gate(request, True)
    if e:
        return e
    try:
        req = json.loads(await request.body())
        msg = req["message"]
        ok = (isinstance(msg, dict) and isinstance(msg.get("messageId"), str) and msg["messageId"]
              and msg.get("role") == "ROLE_USER" and isinstance(msg.get("parts"), list) and msg["parts"])
    except Exception:
        ok = False
    if not ok:
        return err(400, "INVALID_REQUEST", "A valid message with messageId, role ROLE_USER and parts is required.")
    return await run_in_threadpool(handle_send, principal, req, msg)


@app.get("/a2a/tasks/{task_id}")
async def get_task(task_id: str, request: Request):
    principal, e = gate(request, False)
    if e:
        return e
    row = qone("SELECT principal, task FROM tasks WHERE id=?", (task_id,))
    if not row or row[0] != principal:
        return err(404, "TASK_NOT_FOUND", "The task was not found or is not accessible.")
    return respond(view(json.loads(row[1]), to_int(request.query_params.get("historyLength"))))


@app.get("/a2a/tasks")
async def list_tasks(request: Request):
    principal, e = gate(request, False)
    if e:
        return e
    q = request.query_params
    size = 50 if q.get("pageSize") is None else to_int(q.get("pageSize"))
    if size is None or not 1 <= size <= 100:
        return err(400, "INVALID_ARGUMENT", "pageSize must be between 1 and 100.")
    try:
        offset = int(base64.urlsafe_b64decode(q["pageToken"].encode()).decode()) if q.get("pageToken") else 0
    except Exception:
        return err(400, "INVALID_ARGUMENT", "Invalid pageToken.")
    rows = qall("SELECT task FROM tasks WHERE principal=? ORDER BY updated DESC, rowid DESC", (principal,))
    tasks = [json.loads(r[0]) for r in rows]
    if q.get("contextId"):
        tasks = [t for t in tasks if t["contextId"] == q["contextId"]]
    if q.get("status"):
        tasks = [t for t in tasks if t["status"]["state"] == q["status"]]
    page = tasks[offset: offset + size]
    nxt = base64.urlsafe_b64encode(str(offset + size).encode()).decode() if offset + size < len(tasks) else ""
    inc = (q.get("includeArtifacts") or "").lower() == "true"
    hist = to_int(q.get("historyLength"))
    return respond({"tasks": [view(t, hist, inc) for t in page], "nextPageToken": nxt,
                    "pageSize": size, "totalSize": len(tasks)})


@app.post("/a2a/tasks/{task_id}:cancel")
async def cancel_task(task_id: str, request: Request):
    body = await request.body()
    principal, e = gate(request, bool(body.strip()))
    if e:
        return e
    return await run_in_threadpool(handle_cancel, principal, task_id)
