"""Core retrieval and orchestration logic for the PX RAG application.

This module is responsible for:
- loading the synthetic corpus and prompt files
- embedding and ranking relevant document chunks
- preparing the final prompt sent to the LLM
- enforcing confirmation before dangerous actions are accepted
- returning structured JSON to the FastAPI layer
"""

import json, re, uuid
from pathlib import Path

import numpy as np, yaml

from . import config as C, llm

BASE = Path(__file__).parent
CORPUS = json.loads((BASE / "data" / "corpus.json").read_text(encoding="utf-8"))
MANIFEST = yaml.safe_load((BASE / "prompts" / "manifest.yaml").read_text())
PROMPTS = {
    k: (BASE / "prompts" / v["file"]).read_text().strip()
    for k, v in MANIFEST["versions"].items()
}

# Two document collections are available: the real corpus and a test corpus that
# includes a deliberately poisoned chunk used to demonstrate injection attacks.
INDEXES = {
    "px-docs": CORPUS["main"],
    "px-docs-injection-test": CORPUS["main"] + CORPUS["injection_extra"],
}
_vec = {}


def _index_vectors(name):
    """Cache embeddings for one index so retrieval does not recompute them."""
    if name not in _vec:
        _vec[name] = llm.embed([c["text"] for c in INDEXES[name]])
    return _vec[name]


def warm():
    """Preload all indexes during app startup."""
    for n in INDEXES:
        _index_vectors(n)


def retrieve(question, index, top_k):
    """Return the highest-scoring chunks for the given question and document index."""
    v = llm.embed([question])[0]
    sims = _index_vectors(index) @ v
    order = np.argsort(-sims)[:top_k]
    return [dict(INDEXES[index][i], score=round(float(sims[i]), 3)) for i in order]


USER_TEMPLATE = """Documents:
{docs}

Cite the documents you use by adding their id in square brackets at the END of the sentence that uses them, for example "The limit is 10 bar. [px100-s3-0]". Never use an id as a word inside a sentence.
Question: {q}"""

# This note is injected before the model prompt so the model knows the service
# request tool only drafts a request and is not an actual submission.
TOOL_NOTE = "Note from the application: create_service_request only drafts a request; the application asks the user to confirm before it is submitted."

_EMP = re.compile(r"EMP-?\d{4,}", re.I)
_PHONE = re.compile(r"\+\d{1,3}(?:[ -]?\d{2,}){2,}")
_TECH = re.compile(r"(Technician )[^,]+(, employee id)")


def redact(text):
    """Mask personal data in retrieved chunks before the model sees them."""
    text = _TECH.sub(r"\1[REDACTED]\2", text)
    text = _EMP.sub("[EMPLOYEE ID REDACTED]", text)
    return _PHONE.sub("[PHONE REDACTED]", text)


SESSIONS = {}  # session id -> {"history": [...], "pending": {...} | None}
YES = re.compile(
    r"^\s*(yes|yep|confirm|confirmed|submit( it)?|go ahead|please submit( it)?|yes,? please.*)\W*$",
    re.I,
)
NO = re.compile(r"^\s*(no|cancel|don'?t|do not).*$", re.I)


def _citations(text, chunks):
    """Extract any [id] markers from the model answer and map them to sources."""
    ids = re.findall(r"\[([A-Za-z0-9\-_.]+)\]", text)
    by = {c["id"]: c for c in chunks}
    seen, cites = set(), []
    for i in ids:
        if i in by and i not in seen:
            seen.add(i)
            cites.append({"doc": by[i]["doc"], "section": by[i]["section"]})
    clean = re.sub(r"\s*\[[A-Za-z0-9\-_.]+\]", "", text).strip()
    return clean, cites


def answer(req):
    """Handle one user question end-to-end.

    The function resolves a prompt version, retrieves relevant documents, calls the
    model, and manages confirmation for any tool-based action.
    """
    version = (req.get("prompt_version") or MANIFEST["default"]).replace(".", "_")
    if version not in PROMPTS:
        raise KeyError(f"unknown prompt_version {version!r}")
    index = req.get("index") or "px-docs"
    if index not in INDEXES:
        raise KeyError(f"unknown index {index!r}")
    temperature = req.get("temperature", C.DEFAULT_TEMPERATURE)
    top_p = req.get("top_p", C.DEFAULT_TOP_P)
    top_k = int(req.get("top_k", C.DEFAULT_TOP_K))
    seed = req.get("seed")
    sid = req.get("session") or uuid.uuid4().hex[:12]
    sess = SESSIONS.setdefault(sid, {"history": [], "pending": None})
    q = req["question"]
    settings = {
        "prompt_version": version,
        "index": index,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed,
        "model": (req.get("model") or {}).get("deployment")
        or C.CHAT_DEPLOYMENT
        or C.MODE,
        "model_source": "learner" if req.get("model") else "shared",
        "confirm_guard": not C.UNSAFE_NO_CONFIRM_GUARD,
        "redact_personal_data": C.REDACT_PERSONAL_DATA,
    }
    base = {"session": sid, "settings": settings}

    # confirmation turn: handled by the application, not the model
    if sess["pending"] and YES.match(q):
        p = sess["pending"]
        sess["pending"] = None
        call = {
            "name": p["name"],
            "status": "created",
            "args": p["args"],
            "request_id": "SR-" + uuid.uuid4().hex[:6].upper(),
        }
        return {
            **base,
            "answer": f"Service request {call['request_id']} has been created for {p['args'].get('asset_id')}.",
            "citations": [],
            "chunks": [],
            "tool_calls": [call],
        }
    if sess["pending"] and NO.match(q):
        sess["pending"] = None
        return {
            **base,
            "answer": "Understood. The service request was not submitted.",
            "citations": [],
            "chunks": [],
            "tool_calls": [],
        }

    # retrieval: caller may inject chunks (used by ordering-sensitivity tests)
    if req.get("chunks"):
        chunks = [
            {
                "id": c.get("id", f"inj-{i}"),
                "doc": c.get("doc", "injected"),
                "section": c.get("section", ""),
                "text": c["text"],
            }
            for i, c in enumerate(req["chunks"])
        ]
    else:
        chunks = retrieve(q, index, top_k)
        if (
            index == "px-docs-injection-test"
        ):  # make sure the poisoned entry always reaches the model
            for x in CORPUS["injection_extra"]:
                if x["id"] not in {c["id"] for c in chunks}:
                    chunks.append(x)
    if C.REDACT_PERSONAL_DATA:
        chunks = [dict(c, text=redact(c["text"])) for c in chunks]
    docs = "\n".join(f"[{c['id']}] {c['text']}" for c in chunks)
    text, calls, ignored = llm.chat(
        PROMPTS[version],
        sess["history"][-6:],
        TOOL_NOTE + "\n\n" + USER_TEMPLATE.format(docs=docs, q=q),
        temperature,
        top_p,
        seed,
        req.get("model"),
    )
    if ignored:
        settings["ignored_by_model"] = ignored
    clean, cites = _citations(text, chunks)
    tool_calls = []
    for c in calls:
        if C.UNSAFE_NO_CONFIRM_GUARD:
            tool_calls.append(
                {
                    "name": c["name"],
                    "status": "created",
                    "args": c["args"],
                    "request_id": "SR-" + uuid.uuid4().hex[:6].upper(),
                }
            )
        else:
            tool_calls.append(
                {"name": c["name"], "status": "pending_confirmation", "args": c["args"]}
            )
            sess["pending"] = c
    sess["history"] += [
        {"role": "user", "content": q},
        {"role": "assistant", "content": clean},
    ]
    return {
        **base,
        "answer": clean,
        "citations": cites,
        "chunks": [{k: c[k] for k in ("id", "doc", "section", "text")} for c in chunks],
        "tool_calls": tool_calls,
    }
