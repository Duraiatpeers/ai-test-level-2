#!/bin/bash
# Azure App Service startup command.
# We run a single worker so all in-memory session state and embeddings stay in
# one process. This matches the app's design assumption that sessions are local to
# one instance and are reset on restart.
gunicorn -k uvicorn.workers.UvicornWorker -w 1 --timeout 120 -b 0.0.0.0:${PORT:-8000} app.main:app
