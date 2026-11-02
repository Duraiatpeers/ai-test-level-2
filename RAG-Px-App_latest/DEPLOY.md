# PX Service Assistant: deploy to Azure App Service (Linux, Python 3.12)

Training application for Bosch AI Testing Academy Level 2. All documents are synthetic.

## 0. What you need before you start
1. Azure CLI installed, then `az login` and `az account set --subscription "<subscription name or id>"`.
2. An Azure OpenAI resource (the Level 1 Foundry resource works) with TWO deployments:
   - a chat model, e.g. `gpt-5.4-mini`  -> note the DEPLOYMENT NAME
   - an embedding model, e.g. `text-embedding-3-small` -> note the DEPLOYMENT NAME
   In the Azure portal: open the Azure OpenAI resource > Resource Management > Keys and Endpoint. Copy ENDPOINT and KEY 1.
   Deployment names: ai.azure.com > your project > Models + endpoints (or portal: Azure OpenAI > Model deployments > Manage Deployments).
3. A training key you invent (long random string). Learners send it in the `X-API-Key` header.
   Generate one: `python -c "import secrets; print(secrets.token_urlsafe(32))"`

## 1. Create the web app (values in <> are yours)
```
az group create -n rg-px-training -l centralindia
az appservice plan create -g rg-px-training -n plan-px-training --is-linux --sku B1
az webapp create -g rg-px-training -p plan-px-training -n <globally-unique-app-name> --runtime "PYTHON:3.12"
az webapp config set -g rg-px-training -n <app-name> --startup-file "startup.sh" --always-on true
```

## 2. Application settings (keys go here, never in code)
```
az webapp config appsettings set -g rg-px-training -n <app-name> --settings \
  LLM_MODE=azure \
  AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com/" \
  AZURE_OPENAI_API_KEY="<key 1>" \
  AZURE_OPENAI_API_VERSION=2025-04-01-preview \
  AZURE_OPENAI_CHAT_DEPLOYMENT="<chat deployment name>" \
  AZURE_OPENAI_EMBED_DEPLOYMENT="<embedding deployment name>" \
  TRAINING_API_KEY="<training key>" \
  DEFAULT_TEMPERATURE=0.2 DEFAULT_TOP_P=0.9 \
  UNSAFE_NO_CONFIRM_GUARD=0 \
  SCM_DO_BUILD_DURING_DEPLOYMENT=true
```
Portal equivalent: App Service > Settings > Environment variables > App settings > + Add (one row per name above) > Apply > Confirm.
The startup command is under Settings > Configuration > General settings > Startup Command (`startup.sh`).

## 3. Deploy the code
From the folder that contains this file:
```
zip -r ../px-rag-app.zip . -x "*.pyc" "__pycache__/*" ".env"
az webapp deploy -g rg-px-training -n <app-name> --src-path ../px-rag-app.zip --type zip
```
First start takes 2 to 4 minutes (dependencies install, index embeds).

## 4. Verify
```
curl https://<app-name>.azurewebsites.net/health
python smoke_test.py https://<app-name>.azurewebsites.net "<training key>"
```
`/health` must show `"mode": "azure"` and `"chunks": 28`. smoke_test.py prints PASS/FAIL per check.
If something fails: `az webapp log tail -g rg-px-training -n <app-name>`.

## 5. What learners use
- Base URL: `https://<app-name>.azurewebsites.net`   Header: `X-API-Key: <training key>`
- `POST /ask` body: `{"question": "...", "prompt_version": "v1|v2|v2.1", "temperature": 0.2, "top_p": 0.9, "seed": 1, "top_k": 4, "session": "<id>", "index": "px-docs|px-docs-injection-test", "chunks": [ ... optional injected chunks ... ]}`
- Response: `answer, citations[{doc,section}], chunks[{id,doc,section,text}], tool_calls[{name,status,args}], session, settings, latency_ms`
- `GET /versions` lists the three prompt versions with their text. `GET /health` is open (no key).
- `POST /reset?session=<id>` clears a session.

Each learner uses THEIR OWN chat model. It is sent in three headers on every call (embeddings stay on the shared deployment):
`X-Model-Endpoint` (e.g. https://<name>.openai.azure.com/), `X-Model-Key`, `X-Model-Chat-Deployment`.
Learner finds them: Azure portal > their Azure OpenAI resource > Resource Management > Keys and Endpoint (endpoint and key);
deployment name: ai.azure.com > Models + endpoints. The key travels over HTTPS and is not stored or logged by the app. Use a key you can rotate.
Without these headers `/ask` returns 400. To let the app fall back to the shared model instead, set REQUIRE_LEARNER_MODEL=0.

Adapter used in the module decks:
```python
import os, requests
BASE_URL = os.environ["PX_URL"]
HEADERS = {"X-API-Key": os.environ["PX_KEY"],
           "X-Model-Endpoint": os.environ["AOAI_ENDPOINT"],
           "X-Model-Key": os.environ["AOAI_KEY"],
           "X-Model-Chat-Deployment": os.environ["AOAI_CHAT"]}
def ask(question, version=None, **settings):
    version = version or os.environ.get("PROMPT_VERSION", "v1")
    r = requests.post(f"{BASE_URL}/ask", headers=HEADERS,
        json={"question": question, "prompt_version": version, **settings}, timeout=60)
    r.raise_for_status(); return r.json()
```

## 6. Trainer notes
- Prompt versions: v1 baseline, v2 (the two-change edit that causes regressions), v2.1 (the fix). Edit `app/prompts/` and `manifest.yaml`, then redeploy, to change them.
- `UNSAFE_NO_CONFIRM_GUARD=1` removes the code-level confirmation guard so a prompt-only failure can be demonstrated (M2.2 defence in depth). Set it back to 0 afterwards.
- Cost and limits: chat calls use each learner's own deployment and quota. Only the embedding of each question uses the shared text-embedding-3-small deployment (very small).
- Sessions and embeddings live in memory in a single worker. A restart clears sessions. Do not scale out to several instances.
- Rotate or delete the training key after the course: change `TRAINING_API_KEY`, or `az group delete -n rg-px-training` to remove everything.
