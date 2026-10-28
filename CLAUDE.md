# SMART TRADE — Sentiment engine

Python engine for `/pro/sentiment` in the HUB. Public repo `sicketor/smart-trade-sentiment`, separate from the private HUB repo:
never put HUB code or any secret here. Source of truth: `SPEC-he.md` (update it with every shipped change); research: `docs/research-he.md`.

- Python 3.12 venv in `.venv` (system Python 3.15 has no torch wheels). Run tools as `.venv\Scripts\python.exe`.
- Secrets only in `.env` (gitignored) or Actions secrets: `GEMINI_API_KEY`, `SEC_USER_AGENT`, Supabase service key. Never print them.
- Layout: `sentiment/sources/*` adapters → `linking`, `dedup`, `features` → `models/runner` → `meta` (+`llm`) → `aggregate` → `pipeline`/`cli`.
- Run: `.venv\Scripts\python.exe -m sentiment analyze NVDA AMD --no-llm --items --out out.json`.

## Verification commands
- Unit tests: `.venv\Scripts\python.exe -m pytest -q -m "not network"`
