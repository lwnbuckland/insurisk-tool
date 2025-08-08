#!/usr/bin/env python3
"""
Simple HTTP server to expose the Insurance Risk Analyzer as a lightweight web app.

This server does not rely on Streamlit, which isn't available in this environment.
Instead, it implements a minimal HTML interface using Python's standard library.
Users can enter a business website URL into the form. When submitted, the server
fetches content from the provided site, detects the most prominent occupation
based on a risk dictionary (parsed from Document10), assigns a 1–10 risk score,
chooses a sector, and renders evidence snippets. The logic closely follows
the original `app.py` from the provided Streamlit app.

To run the server, execute this file directly. It defaults to port 8501 but
can be changed via the PORT environment variable. Once running, open your
browser to http://localhost:8501 to access the interface.
"""

import os
import re
import json
import csv
import time
import math
from urllib.parse import urlparse, urljoin, parse_qs
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

import requests
from bs4 import BeautifulSoup
import pandas as pd
from rapidfuzz import process, fuzz


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def load_risk_dict(csv_path: str):
    """Load the risk dictionary from a CSV into a DataFrame and mapping."""
    df = pd.read_csv(csv_path)
    # Normalize occupation names for lookup
    df["occupation_norm"] = df["occupation"].astype(str).str.strip().str.lower()
    mapping = dict(zip(df["occupation_norm"], df["rating"].astype(int)))
    return df, mapping


def parse_synonyms(yaml_path: str):
    """
    Parse a very simple YAML mapping file into a dictionary.

    The synonyms file provided uses a straightforward mapping with keys and
    values separated by a colon. It may contain comments (lines starting with
    '#') and quoted values. Since PyYAML may not be installed, this parser
    implements a minimal YAML subset suitable for the file's structure.
    """
    synonyms = {}
    with open(yaml_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ':' not in line:
                continue
            # Split on the first colon only
            key, value = line.split(":", 1)
            key = key.strip().strip('"').strip("'")
            value = value.strip().strip('"').strip("'")
            if key:
                synonyms[key.lower()] = value
    return synonyms


def load_sector_keywords(json_path: str):
    """Load sector keywords from a JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Ensure keys and keywords are lower-cased for matching
    lowered = {}
    for sector, kws in data.items():
        lowered[sector] = [kw.lower() for kw in kws]
    return lowered


# -----------------------------------------------------------------------------
# Core logic (mirrors app.py)
# -----------------------------------------------------------------------------

def tier_for(score: int):
    """Return tier label and colour given a numeric score."""
    tiers = [
        (1, 3, "Low / In appetite", "green"),
        (4, 6, "Medium / Needs review", "orange"),
        (7, 8, "High / Hard to place", "red"),
        (9, 9, "Very High / Potentially placeable", "red"),
        (10, 10, "Out of appetite / Decline", "black"),
    ]
    for lo, hi, label, colour in tiers:
        if lo <= score <= hi:
            return label, colour
    return "Unknown", "grey"


def sanitize_url(u: str):
    if not u:
        return None
    u = u.strip()
    if not u.startswith("http://") and not u.startswith("https://"):
        u = "https://" + u
    return u


def same_domain(u: str, base: str) -> bool:
    try:
        return urlparse(u).netloc == urlparse(base).netloc
    except Exception:
        return False


def fetch(url: str, timeout: int = 12):
    """Fetch a URL and return its text content or None."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; RiskAnalyzer/1.0)"}
        resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        if resp.status_code == 200 and resp.content:
            return resp.text
    except Exception:
        return None
    return None


def extract_links_and_text(html: str, base_url: str):
    """
    Parse HTML and return a tuple (links, text).

    - links: list of unique internal links prioritised by likely relevance.
    - text: main textual content with whitespace normalised.
    """
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text)
    # Collect candidate internal links
    links = []
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"])
        if same_domain(href, base_url):
            links.append(href)
    # Prioritise likely pages
    wanted = [
        "about",
        "service",
        "services",
        "what-we-do",
        "industr",
        "product",
        "sectors",
        "our-work",
    ]
    priority = [l for l in links if any(w in l.lower() for w in wanted)]
    # Ensure uniqueness preserving order
    seen = set()
    ordered = []
    for l in priority + links:
        if l not in seen:
            seen.add(l)
            ordered.append(l)
    return ordered[:12], text


def detect_candidates(all_text: str, mapping: dict, synonyms: dict, canonicals: set):
    """
    Detect potential occupation matches in the given text.
    Returns a dict mapping canonical names to info including rating and source.
    """
    found = {}
    text_norm = all_text.lower()
    # Exact matches from canonical dict
    for name_norm, rating in mapping.items():
        pattern = r"\b" + re.escape(name_norm) + r"\b"
        if re.search(pattern, text_norm):
            found[name_norm] = {
                "name": name_norm,
                "rating": int(rating),
                "source": "exact",
            }
    # Synonym expansion
    for syn, canonical in (synonyms or {}).items():
        syn_norm = syn.lower()
        can_norm = canonical.strip().lower()
        if syn_norm in text_norm and can_norm in canonicals:
            found[can_norm] = {
                "name": can_norm,
                "rating": int(mapping[can_norm]),
                "source": f"synonym:{syn}",
            }
    # Fuzzy (guarded): only if high similarity and appears near service phrases
    windows = re.findall(r"(?:our services|we (?:provide|offer)|services include).{0,200}", text_norm)
    candidates = set()
    for w in windows:
        # extractOne may return None if the collection is empty or no match is found
        result = process.extractOne(w, list(canonicals), scorer=fuzz.partial_ratio)
        if result:
            match, score, _ = result
            if score >= 92 and match:
                candidates.add(match)
    for c in candidates:
        found[c] = {
            "name": c,
            "rating": int(mapping[c]),
            "source": "fuzzy-guarded",
        }
    return found


def prominence_rank(text: str, candidates: dict):
    """
    Rank candidates by prominence: weighting early occurrences and service headings.
    Returns a tuple (ordered list of canonical names, weight mapping).
    """
    weights = {}
    t = text.lower()
    for c_norm in candidates:
        freq = t.count(c_norm)
        weight = freq
        # Early text/hero/title weighting
        if re.search(r"\b" + re.escape(c_norm) + r"\b", t[:400]):
            weight += 5
        # Service heading weighting
        if re.search(r"(services|what we do).{0,120}" + re.escape(c_norm), t):
            weight += 3
        weights[c_norm] = weight
    ranked = sorted(candidates.keys(), key=lambda x: (-weights.get(x, 0), -candidates[x]["rating"]))
    return ranked, weights


def evidence_snippets(text: str, term: str, max_snips: int = 3):
    """Return up to max_snips snippets containing the term for evidence display."""
    tn = term.lower()
    spans = []
    for m in re.finditer(re.escape(tn), text.lower()):
        start = max(0, m.start() - 120)
        end = min(len(text), m.end() + 120)
        spans.append(text[start:end])
        if len(spans) >= max_snips:
            break
    return ["…" + s.strip() + "…" for s in spans]


def choose_sector(text: str, sector_kws: dict):
    """Choose the best sector based on keyword frequencies in the text."""
    t = text.lower()
    best = None
    best_score = 0
    for sector, kws in sector_kws.items():
        score = 0
        for kw in kws:
            score += t.count(kw)
        if score > best_score:
            best_score = score
            best = sector
    # Fallback to 'Services' if nothing matched
    return best or "Services"


def analyze_url(url: str, mapping: dict, synonyms: dict, sector_kws: dict):
    """
    High-level analysis pipeline for a given URL.

    Fetches the main page and a handful of internal pages, aggregates text,
    identifies candidate occupations, ranks them by prominence and rating, then
    returns a summary dict suitable for HTML rendering.
    """
    url = sanitize_url(url)
    if not url:
        return {"error": "Invalid URL"}
    html = fetch(url)
    if not html:
        return {"error": "Could not fetch URL. If the site is heavily JS-driven, this simple crawler may fail."}
    links, text = extract_links_and_text(html, url)
    collected_texts = [text]
    # Crawl up to six secondary pages
    for link in links[:6]:
        html2 = fetch(link)
        if html2:
            _, t2 = extract_links_and_text(html2, link)
            collected_texts.append(t2)
    all_text = " ".join(collected_texts)
    canonicals = set(mapping.keys())
    found = detect_candidates(all_text, mapping, synonyms, canonicals)
    # Build result structure when nothing was found
    if not found:
        return {
            "url": url,
            "business_name": urlparse(url).netloc,
            "score": None,
            "tier": None,
            "sector": choose_sector(all_text, sector_kws),
            "core": None,
            "flags": [],
            "notes": "No qualifying occupations detected from on-site evidence.",
            "evidence": [],
        }
    ranked, weights = prominence_rank(all_text, found)
    core_key = ranked[0]
    core = found[core_key]
    score = int(core["rating"])
    tier_label, tier_colour = tier_for(score)
    # Additional flags: other items with rating >=7
    flags = []
    for k, v in found.items():
        if k == core_key:
            continue
        if int(v["rating"]) >= 7:
            flags.append({"occupation": k, "rating": int(v["rating"])})
    # Evidence snippets
    evidence = evidence_snippets(all_text, core_key, 3)
    # Business name heuristic: pick from early capitalised words
    name_guess = urlparse(url).netloc
    m = re.search(r"([A-Z][A-Za-z0-9&' ]{2,60})(?:\s[-|•|–])", collected_texts[0])
    if m:
        name_guess = m.group(1).strip()
    # Notes summarising logic
    notes = (
        f"Core activity selected by prominence weighting and on-site phrasing. "
        f"Matched '{core_key}' via {core.get('source')} with rating {score} from Document10. "
        "9 = 'Very High / Potentially placeable' is treated as such; no averaging applied. "
        + ("Additional high-risk flags present." if flags else "No additional high-risk flags detected.")
    )
    return {
        "url": url,
        "business_name": name_guess,
        "score": score,
        "tier": tier_label,
        "sector": choose_sector(all_text, sector_kws),
        "core": {"occupation": core_key, "rating": score, "source": core.get("source")},
        "flags": flags,
        "notes": notes,
        "evidence": evidence,
    }


# -----------------------------------------------------------------------------
# HTTP server definitions
# -----------------------------------------------------------------------------

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in separate threads."""
    daemon_threads = True


class RiskHandler(BaseHTTPRequestHandler):
    """
    HTTP request handler for the risk analyzer web app.

    This handler serves a simple HTML page with a form on GET requests. If the
    `url` query parameter is provided, it runs the analysis and embeds the
    results into the response. Otherwise, it renders only the form.
    """
    # Pre-load data once for all handler instances
    base_dir = os.path.dirname(os.path.abspath(__file__))
    df, mapping = load_risk_dict(os.path.join(base_dir, "risk_dictionary.csv"))
    synonyms = parse_synonyms(os.path.join(base_dir, "synonyms.yaml"))
    sector_kws = load_sector_keywords(os.path.join(base_dir, "sector_keywords.json"))

    def do_GET(self):
        # Parse query string
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        url = params.get("url", [""])[0]
        url = url.strip()
        # Run analysis if URL present
        result = None
        if url:
            result = analyze_url(url, self.mapping, self.synonyms, self.sector_kws)
        # Build HTML response
        html = self.render_page(url, result)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html.encode("utf-8"))))
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def log_message(self, format: str, *args) -> None:  # type: ignore[override]
        # Silence default logging to stderr for cleaner output
        return

    def render_page(self, url: str, res: dict | None) -> str:
        """
        Construct an HTML page displaying the form and, if available, the
        analysis result. Uses basic CSS for readability. The form submits
        via GET so that the URL remains shareable.
        """
        # Escape HTML characters for safety
        import html as html_lib

        def esc(s: str) -> str:
            return html_lib.escape(s, quote=True) if isinstance(s, str) else str(s)

        # Form HTML
        form_html = f"""
            <form method="get" action="/" style="margin-bottom: 1.5rem;">
                <input type="text" name="url" placeholder="https://example.com" value="{esc(url)}" style="width: 60%; padding: 0.5rem;" />
                <button type="submit" style="padding: 0.5rem 1rem;">Analyze</button>
            </form>
        """
        result_html = ""
        if res:
            if "error" in res:
                result_html = f"<div style='color: red; font-weight: bold;'>Error: {esc(res['error'])}</div>"
            else:
                # Build result sections
                # Score and tier
                if res.get("score") is not None:
                    tier_label = esc(res.get("tier", ""))
                    score = esc(res.get("score"))
                    metric_html = f"<div style='margin-bottom: 0.5rem;'><strong>Risk Score (1–10):</strong> {score} &nbsp;&nbsp; <strong>Tier:</strong> {tier_label}</div>"
                else:
                    metric_html = "<div style='margin-bottom: 0.5rem;'><strong>Risk Score:</strong> N/A</div>"
                # Business name and URL
                name = esc(res.get("business_name", res.get("url", "")))
                url_display = esc(res.get("url", ""))
                basic_html = f"<h2 style='margin-top:0;'>{name}</h2><p>{url_display}</p>"
                # Sector
                sector_html = f"<p><strong>Business Sector (chosen):</strong> {esc(res.get('sector', ''))}</p>"
                # Core occupation
                core_occ = res.get("core", {})
                if core_occ:
                    core_html = f"<p><strong>Detected Occupation/Industry (core):</strong> {esc(core_occ.get('occupation', ''))} — rating {esc(core_occ.get('rating', ''))}</p>"
                else:
                    core_html = "<p><strong>Detected Occupation/Industry:</strong> None</p>"
                # Flags
                flags = res.get("flags", [])
                if flags:
                    flags_html = "<p><strong>Additional High-Risk Flags (≥7, advisory only):</strong></p><ul>"
                    for f in flags:
                        flags_html += f"<li>{esc(f['occupation'])} — {esc(f['rating'])}</li>"
                    flags_html += "</ul>"
                else:
                    flags_html = ""
                # Notes
                notes_html = f"<p><strong>Reasoning/Notes:</strong> {esc(res.get('notes', ''))}</p>"
                # Evidence
                evidence_html = ""
                ev = res.get("evidence", [])
                if ev:
                    evidence_html = "<p><strong>Evidence Snippets:</strong></p>"
                    for snip in ev:
                        evidence_html += f"<pre style='background:#f0f0f0;padding:0.5rem;'>{esc(snip)}</pre>"
                # Compose all
                result_html = (
                    "<div style='border:1px solid #ccc;padding:1rem;border-radius:4px;'>"
                    + basic_html
                    + metric_html
                    + sector_html
                    + core_html
                    + flags_html
                    + notes_html
                    + evidence_html
                    + "</div>"
                )
        # Top-level page structure
        page = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Insurance Risk Analyzer</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 2rem auto; max-width: 800px; }}
        h1 {{ margin-bottom: 0.5rem; }}
    </style>
</head>
<body>
    <h1>Insurance Risk Analyzer</h1>
    <p>Enter a business website URL to detect the core occupation and risk score.</p>
    {form_html}
    {result_html}
    <hr/>
    <p style="font-size:0.8rem;color:#666;">Scoring dictionary loaded from Document10. You can edit 'synonyms.yaml' and 'sector_keywords.json' to refine matching and sector choice.</p>
</body>
</html>
        """
        return page


def main():
    port = int(os.environ.get("PORT", "8501"))
    server_address = ("0.0.0.0", port)
    httpd = ThreadedHTTPServer(server_address, RiskHandler)
    print(f"Insurance Risk Analyzer server running on http://localhost:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down server…")
        httpd.server_close()


if __name__ == "__main__":
    main()