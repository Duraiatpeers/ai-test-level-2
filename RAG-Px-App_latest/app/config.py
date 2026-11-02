"""Application configuration for the PX training RAG service.

This file centralizes all environment settings used by the app. The values are
read from environment variables so the same code can run locally in dry-run mode
or in Azure with real model credentials.
"""

import os


def _f(name, default):
    """Read an environment value and cast it to float when present."""
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else default


# Runtime mode: "azure" uses real Azure OpenAI access; "stub" uses a local
# stand-in model for offline demo runs.
MODE = os.environ.get("LLM_MODE", "azure")  # "azure" or "stub" (offline dry run)

# Azure OpenAI connection settings for the shared deployment used for embeddings
# and, when configured, the default chat deployment.
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_API_VERSION = os.environ.get(
    "AZURE_OPENAI_API_VERSION", "2025-04-01-preview"
)
CHAT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "")
EMBED_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBED_DEPLOYMENT", "")

# Shared training auth key. Empty means no auth is enforced. This is meant for a
# controlled lab environment where each learner uses a known key.
TRAINING_API_KEY = os.environ.get("TRAINING_API_KEY", "")

# Default generation settings used by the /ask endpoint when callers do not send
# explicit values for temperature, top_p or top_k.
DEFAULT_TEMPERATURE = _f("DEFAULT_TEMPERATURE", 0.2)
DEFAULT_TOP_P = _f("DEFAULT_TOP_P", 0.9)
DEFAULT_TOP_K = int(_f("DEFAULT_TOP_K", 4))

# Teaching switch for module M2.2: when "1", a drafted service request is created
# immediately (no code-level confirmation guard) so learners can see a prompt-only
# control fail.
UNSAFE_NO_CONFIRM_GUARD = os.environ.get("UNSAFE_NO_CONFIRM_GUARD", "0") == "1"

# Each learner brings their own chat model via request headers. Set to "0" to
# fall back to the shared deployment instead of requiring a learner-provided model.
REQUIRE_LEARNER_MODEL = os.environ.get("REQUIRE_LEARNER_MODEL", "1") == "1"
ALLOWED_HOST_SUFFIXES = (
    ".openai.azure.com",
    ".services.ai.azure.com",
    ".cognitiveservices.azure.com",
)

# Application-level control for personal data: mask technician names, employee IDs
# and mobile numbers in retrieved chunks before the model sees them. Off by
# default so learners can discover the leak (module M2.2, defense in depth).
REDACT_PERSONAL_DATA = os.environ.get("REDACT_PERSONAL_DATA", "0") == "1"
