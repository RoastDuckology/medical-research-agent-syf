# Security Notes

This repository is prepared for public demonstration.

- Do not commit `.env` or `.streamlit/secrets.toml`.
- Do not commit API keys, access tokens, private logs, patient data, or internal project notes.
- Demo files under `data/demo_logs/` and `examples/demo_reports/` are synthetic.
- Local ChromaDB files under `data/vector_db/` are ignored and should be rebuilt locally.

If a credential is committed by mistake, rotate it immediately from the provider console and remove it from Git history before publishing.
