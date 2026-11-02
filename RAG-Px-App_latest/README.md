Local dry run (no Azure, offline stand-in model; plumbing only, answers are NOT realistic):
  pip install -r requirements.txt
  LLM_MODE=stub TRAINING_API_KEY=k bash startup.sh   (then python smoke_test.py http://127.0.0.1:8000 k)
See DEPLOY.md for Azure App Service deployment.
