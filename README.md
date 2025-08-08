
# Insurance Risk Analyzer (Streamlit)

A shareable web tool that accepts a business website URL, detects the **single core occupation** using your Document10 ratings, assigns a **1–10 score** (no averaging), chooses **one sector**, and shows **evidence snippets**. 

- **1–3 = Low / In appetite**
- **4–6 = Medium / Needs review**
- **7–8 = High / Hard to place**
- **9 = Very High / Potentially placeable**
- **10 = Out of appetite / Decline**

## Contents
- `app.py` — Streamlit web app
- `risk_dictionary.csv` — parsed from your uploaded Document10.docx
- `synonyms.yaml` — common phrases → canonical Document10 entries (edit to tune)
- `sector_keywords.json` — keywords used to pick a **single** sector (edit to tune)
- `requirements.txt` — Python dependencies

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open: http://localhost:8501

To share a direct assessment link: `http://localhost:8501/?url=https://www.example.com`  
(When deployed, set `PUBLIC_BASE_URL` env var to your app's public URL for nicer links.)

## Deploy (Streamlit Community Cloud)
1. Push this folder to GitHub.
2. Create a new Streamlit app pointing to `app.py`.
3. Add the environment variable `PUBLIC_BASE_URL=https://<your-app-subdomain>.streamlit.app` for crisp share links.

## Deploy (Render/Heroku)
- Use a Docker or simple Python deploy; run command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`  
- Set `PUBLIC_BASE_URL` to your public URL.

## Notes
- The crawler uses `requests + BeautifulSoup` and fetches the homepage plus a handful of likely internal pages (`about`, `services`, etc.). For heavy JS sites, deploy behind a JS-rendering crawler (e.g., Playwright-based service) and swap `fetch()`.
- The app never **averages**. It selects the **most prominent single** occupation and uses its Document10 rating. Additional **flags (≥7)** are listed but do not alter the main score.
- Always exactly **one sector** is chosen. Edit `sector_keywords.json` to refine.
