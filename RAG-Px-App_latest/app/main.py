"""FastAPI entry point for the PX service assistant.

This file exposes the HTTP interface used by learners and testers. The app is
purely API-based: there is no frontend here; requests are sent by Postman, a
Python test script, or another application.
"""

import logging, time
from urllib.parse import urlparse
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from . import config as C, rag

log = logging.getLogger("px")
app = FastAPI(title="PX Service Assistant (training application)", version="1.0")


class Ask(BaseModel):
    """Request body for /ask.

    Each field is optional because the system defaults can be used when the caller
    does not provide explicit tuning values.
    """

    question: str
    prompt_version: Optional[str] = None
    index: Optional[str] = None
    session: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    seed: Optional[int] = None
    chunks: Optional[list[dict[str, Any]]] = None


def auth(x_api_key: Optional[str] = Header(default=None)):
    """Require a valid X-API-Key when a training key is configured."""
    if C.TRAINING_API_KEY and x_api_key != C.TRAINING_API_KEY:
        raise HTTPException(401, "Missing or wrong X-API-Key")


def learner_model(endpoint, key, deployment, api_version):
    """Validate per-request model headers sent by an individual learner.

    If the app is configured to require a learner model, and the request does not
    include an Azure OpenAI endpoint/key/deployment pair, it rejects the call.
    """
    if not (endpoint or key or deployment):
        if C.REQUIRE_LEARNER_MODEL and C.MODE != "stub":
            raise HTTPException(
                400,
                "Send your own model in headers: X-Model-Endpoint, X-Model-Key, X-Model-Chat-Deployment",
            )
        return None
    if not (endpoint and key and deployment):
        raise HTTPException(
            400,
            "X-Model-Endpoint, X-Model-Key and X-Model-Chat-Deployment must all be sent together",
        )
    u = urlparse(endpoint)
    if u.scheme != "https" or not (u.hostname or "").endswith(C.ALLOWED_HOST_SUFFIXES):
        raise HTTPException(
            400,
            "X-Model-Endpoint must be an https Azure OpenAI address, e.g. https://<name>.openai.azure.com/",
        )
    return {
        "endpoint": f"https://{u.hostname}/",
        "key": key,
        "deployment": deployment,
        "api_version": api_version,
    }


@app.on_event("startup")
def _warm():
    """Precompute embeddings once when the app starts."""
    try:
        rag.warm()
        log.warning("index warmed (%s)", C.MODE)
    except Exception as e:  # Keep the app running even if warm-up fails.
        log.error("warm-up failed: %s", e)


@app.get("/health")
def health():
    """Public health endpoint used by smoke tests and deployment validation."""
    return {
        "status": "ok",
        "mode": C.MODE,
        "prompt_versions": list(rag.PROMPTS),
        "indexes": list(rag.INDEXES),
        "chunks": len(rag.INDEXES["px-docs"]),
        "auth_required": bool(C.TRAINING_API_KEY),
        "confirm_guard": not C.UNSAFE_NO_CONFIRM_GUARD,
        "learner_model_required": C.REQUIRE_LEARNER_MODEL,
        "redact_personal_data": C.REDACT_PERSONAL_DATA,
    }


@app.get("/versions", dependencies=[Depends(auth)])
def versions():
    """Expose the available prompt versions and their text."""
    return {
        "default": rag.MANIFEST["default"],
        "versions": {
            k: {**v, "system_prompt": rag.PROMPTS[k]}
            for k, v in rag.MANIFEST["versions"].items()
        },
    }


@app.post("/ask", dependencies=[Depends(auth)])
def ask(
    body: Ask,
    x_model_endpoint: Optional[str] = Header(default=None),
    x_model_key: Optional[str] = Header(default=None),
    x_model_chat_deployment: Optional[str] = Header(default=None),
    x_model_api_version: Optional[str] = Header(default=None),
):
    """Main RAG endpoint: retrieve, prompt, model, and return a response."""
    req = {k: v for k, v in body.model_dump().items() if v is not None}
    lm = learner_model(
        x_model_endpoint, x_model_key, x_model_chat_deployment, x_model_api_version
    )
    if lm:
        req["model"] = lm
    t0 = time.time()
    try:
        out = rag.answer(req)
    except KeyError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        log.exception("ask failed")
        raise HTTPException(502, f"Model call failed: {type(e).__name__}: {e}")
    out["latency_ms"] = int((time.time() - t0) * 1000)
    return out


@app.post("/reset", dependencies=[Depends(auth)])
def reset(session: str):
    """Clear the in-memory state for a given session."""
    rag.SESSIONS.pop(session, None)
    return {"reset": session}
