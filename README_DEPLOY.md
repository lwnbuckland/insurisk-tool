# Deploying the Insurance Risk Analyzer

This repository contains everything you need to run the Insurance Risk Analyzer, which detects the core occupation on a business’s website and assigns a risk score using your Document10 ratings.  It is designed to run on Streamlit or any platform that can execute Python.

---

## Option A — Streamlit Community Cloud (easiest)
1. Create a public GitHub repository and upload the contents of this folder.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/) and create a new app pointing to `app.py` on the `main` branch of your repo.
3. In **Advanced settings**, add an environment variable:
   - `PUBLIC_BASE_URL` = the URL Streamlit assigns to your app (e.g. `https://your-subdomain.streamlit.app`).
4. Click **Deploy**.  You can share assessments directly by appending `?url=` to your app URL, for example:
   - `https://your-subdomain.streamlit.app/?url=https://example.com`

## Option B — Render/Heroku
1. Create a new Web Service from your GitHub repository.
2. Runtime: Python  •  Build command: `pip install -r requirements.txt`
3. Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. Add an environment variable `PUBLIC_BASE_URL` set to your service’s URL (e.g. `https://your-app.onrender.com`).
5. Deploy.  Use the `?url=` query parameter for shareable assessments.

## Important — Populate your risk dictionary

The included `risk_dictionary.csv` currently contains only a header.  You must paste your Document10 occupations and ratings into this file.  Use columns:

```
occupation,rating
Builder's Merchant,9
Restaurant (Licenced),4
Roofing Contractor,8
...etc
```

You can also edit `synonyms.yaml` to map real‑world phrases to your canonical Document10 entries, and `sector_keywords.json` to refine how sectors are chosen.

---

## Local run

To run the app locally:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open your browser at `http://localhost:8501` and append `?url=https://www.example.com` to analyze a site directly.
