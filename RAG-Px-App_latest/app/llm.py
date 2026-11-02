"""Model access layer for the PX RAG service.

This module wraps all calls to Azure OpenAI and also provides a local stub mode
used for dry-run demos when the real Azure model is unavailable. In production,
this is the piece the rest of the app relies on for both embeddings and chat.
"""

import hashlib, math, re
import numpy as np
from . import config as C

# Tool schema exposed to the model. The model can only draft a service request;
# the application itself is responsible for confirmation before any action is
# accepted as real.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_service_request",
            "description": "Draft a service request for a pump asset. This only DRAFTS the request: the application itself asks the user to confirm before anything is submitted. Call it as soon as the user asks to raise a service request, with the asset id and a short issue description.",
            "parameters": {
                "type": "object",
                "properties": {
                    "asset_id": {"type": "string", "description": "e.g. PX100-L3-007"},
                    "issue": {"type": "string"},
                },
                "required": ["asset_id", "issue"],
            },
        },
    }
]

_clients = {}


def client(endpoint=None, key=None, api_version=None):
    """Return a shared AzureOpenAI client for the configured model.

    Embeddings use the shared deployment, while per-request learner models use
    their own keys and endpoint values.
    """
    from openai import AzureOpenAI

    endpoint = endpoint or C.AZURE_OPENAI_ENDPOINT
    key = key or C.AZURE_OPENAI_API_KEY
    api_version = api_version or C.AZURE_OPENAI_API_VERSION
    k = (endpoint, hash(key), api_version)  # key itself is never stored as a dict key
    if k not in _clients:
        if len(_clients) > 200:
            _clients.clear()
        _clients[k] = AzureOpenAI(
            azure_endpoint=endpoint, api_key=key, api_version=api_version, max_retries=2
        )
    return _clients[k]


# ---- embeddings ---------------------------------------------------------------


def _hash_embed(text, dim=256):
    """Fallback embedding used only in stub mode.

    This is a lightweight deterministic hash-based embedding that lets the app work
    without Azure while still producing a vector space for retrieval tests.
    """
    v = np.zeros(dim, dtype=np.float32)
    for w in re.findall(r"[a-z0-9\-\.]+", text.lower()):
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        v[h % dim] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v


def embed(texts):
    """Create embeddings for one or more text fragments.

    In stub mode, we use a local hash-based approximation. In Azure mode, we call
    the configured embeddings deployment and normalize the output.
    """
    if C.MODE == "stub":
        return np.vstack([_hash_embed(t) for t in texts])
    out = []
    for i in range(0, len(texts), 64):
        r = client().embeddings.create(
            model=C.EMBED_DEPLOYMENT, input=texts[i : i + 64]
        )
        out += [d.embedding for d in r.data]
    m = np.array(out, dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


# ---- chat ---------------------------------------------------------------------


def chat(system, history, user, temperature, top_p, seed, model=None):
    """Call the model and return text, tool calls, and ignored parameters.

    Returns a tuple of: (answer_text, tool_calls, ignored_params)
    where tool_calls is a list of dicts like {"name": "create_service_request",
    "args": {...}}.
    """
    if C.MODE == "stub":
        return _stub_chat(user) + ([],)
    msgs = (
        [{"role": "system", "content": system}]
        + history
        + [{"role": "user", "content": user}]
    )
    import json

    ignored = []
    params = {"temperature": temperature, "top_p": top_p}
    if seed is not None:
        params["seed"] = seed
    while True:
        try:
            m_ = model or {}
            r = client(
                m_.get("endpoint"), m_.get("key"), m_.get("api_version")
            ).chat.completions.create(
                model=m_.get("deployment") or C.CHAT_DEPLOYMENT,
                messages=msgs,
                tools=TOOLS,
                **params,
            )
            break
        except Exception as e:
            # Some newer models reject sampling settings such as temperature or
            # top_p; the app removes the unsupported parameter and retries.
            msg = str(e).lower()
            bad = [
                p
                for p in list(params)
                if p in msg
                and ("unsupported" in msg or "not supported" in msg or "invalid" in msg)
            ]
            if not bad:
                raise
            for p in bad:
                params.pop(p)
                ignored.append(p)
    m = r.choices[0].message
    calls = []
    for tc in m.tool_calls or []:
        try:
            args = json.loads(tc.function.arguments)
        except Exception:
            args = {}
        calls.append({"name": tc.function.name, "args": args})
    return (m.content or ""), calls, ignored


def _stub_chat(user):
    """Offline stand-in for local demos.

    This is intentionally simple: it looks for a service-request pattern and
    returns a fake tool call instead of contacting Azure.
    """
    q = user.split("Question:")[-1].strip()
    m = re.search(r"(PX\d{3}-L\d-\d{3})", q)
    if re.search(r"service request|raise a request", q, re.I) and m:
        return "I have drafted a service request. Please confirm to submit it.", [
            {
                "name": "create_service_request",
                "args": {"asset_id": m.group(1), "issue": q[:120]},
            }
        ]
    docs = re.findall(r"\[([^\]]+)\] (.+)", user.split("Question:")[0])
    if not docs:
        return "The documents provided do not state that.", []
    return f"{docs[0][1][:240]} [{docs[0][0]}]", []
