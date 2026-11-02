"""Minimal smoke test for the PX RAG API.

This script exercises the most important learner-facing behaviors:
- grounded answers from the documents
- honest missing-evidence responses
- action drafting with confirmation enforcement

Run after deployment with:
  python smoke_test.py https://<your-app>.azurewebsites.net <TRAINING_API_KEY>
"""

import os, sys

import requests

base, key = sys.argv[1].rstrip("/"), (sys.argv[2] if len(sys.argv) > 2 else "")

# API headers. Learners can optionally send their own Azure OpenAI model in the
# request headers instead of using the shared deployment.
H = {"X-API-Key": key}
if os.environ.get("AOAI_ENDPOINT"):  # learner-style call: own Azure model in headers
    H.update(
        {
            "X-Model-Endpoint": os.environ["AOAI_ENDPOINT"],
            "X-Model-Key": os.environ["AOAI_KEY"],
            "X-Model-Chat-Deployment": os.environ["AOAI_CHAT"],
        }
    )


def ask(q, **kw):
    """Send a single POST /ask request and return the parsed JSON response."""
    r = requests.post(f"{base}/ask", headers=H, json={"question": q, **kw}, timeout=60)
    r.raise_for_status()
    return r.json()


# Health check: confirms the service is running and exposing the expected runtime
# configuration.
h = requests.get(f"{base}/health", timeout=30).json()
print("health:", h)
assert h["status"] == "ok"

checks = []

# 1) Fact-grounding check: the answer should use the retrieved documents and not
# invent an incorrect pressure value.
r = ask("What is the maximum operating pressure of the PX-100?", prompt_version="v1")
checks.append(
    (
        "fact: PX-100 limit says 10 bar, not 15 bar",
        "10 bar" in r["answer"] and "15 bar" not in r["answer"],
    )
)
checks.append(("citations present", len(r["citations"]) > 0))

# 2) Missing-evidence check: if the docs do not cover a fact, the answer should be
# honest and not make it up.
r = ask("What is the warranty period of the PX-100?", prompt_version="v1")
checks.append(
    (
        "missing evidence admitted (v1)",
        any(
            p in r["answer"].lower()
            for p in (
                "do not state",
                "could not find",
                "not found",
                "no information",
                "does not",
            )
        ),
    )
)

# 3) Draft-vs-submit flow: the app should create a pending draft tool call, not a
# real service request, until the user explicitly confirms.
r = ask(
    "Raise a service request for PX100-L3-007. It keeps overheating.", session="smoke"
)
checks.append(
    (
        "tool drafted, not created",
        bool(r["tool_calls"])
        and all(c["status"] == "pending_confirmation" for c in r["tool_calls"]),
    )
)

# 4) Confirmation check: after a yes answer, the app should mark the request as
# created in the same session.
r = ask("Yes, please submit it.", session="smoke")
checks.append(
    (
        "created after confirmation",
        r["tool_calls"] and r["tool_calls"][0]["status"] == "created",
    )
)

for name, ok in checks:
    print("PASS" if ok else "FAIL", "-", name)
sys.exit(0 if all(ok for _, ok in checks) else 1)
