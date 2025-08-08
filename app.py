
import streamlit as st
import csv
import requests, re, json, os, time, math
from urllib.parse import urlparse, urljoin, parse_qs
from bs4 import BeautifulSoup
# Removed dependency on rapidfuzz to avoid binary compilation issues on some platforms.
# Instead of using RapidFuzz for fuzzy matching, we'll implement a simple partial
# ratio function using Python's built‑in difflib.SequenceMatcher. This avoids the
# need for native extensions and still provides basic fuzzy matching capability.
from difflib import SequenceMatcher

def _simple_partial_ratio(a: str, b: str) -> int:
    """Compute a rough partial ratio between two strings.

    This mirrors the behaviour of rapidfuzz.fuzz.partial_ratio by sliding the
    shorter string over the longer one and taking the best match. The return
    value is an integer between 0 and 100 representing percentage similarity.
    """
    a = (a or "").lower()
    b = (b or "").lower()
    if not a or not b:
        return 0
    # Ensure a is the shorter string
    if len(a) > len(b):
        a, b = b, a
    best = 0.0
    len_a = len(a)
    for i in range(len(b) - len_a + 1):
        part = b[i : i + len_a]
        # Ratio returns float in [0,1]; multiply by 100 for percentage
        ratio = SequenceMatcher(None, part, a).ratio()
        if ratio > best:
            best = ratio
    return int(best * 100)

def _extract_one(query: str, choices: list[str]):
    """Return the choice with the highest simple partial ratio to the query.

    Returns a tuple (best_match, score). If choices is empty or no match,
    returns (None, 0).
    """
    best_choice = None
    best_score = 0
    for choice in choices:
        score = _simple_partial_ratio(query, choice)
        if score > best_score:
            best_choice = choice
            best_score = score
    return best_choice, best_score

st.set_page_config(page_title="Insurance Risk Analyzer", page_icon="🛡️", layout="wide")

# Load risk dictionary
@st.cache_data
def load_risk_dict():
    """
    Load the risk dictionary from a CSV file without requiring pandas. This reads
    each row into a dict and builds a mapping of normalized occupation names to ratings.
    Returns a tuple (rows, mapping). If the file is missing or empty, returns empty values.
    """
    rows = []
    mapping = {}
    try:
        with open("risk_dictionary.csv", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Normalize occupation name
                occ = (row.get("occupation") or "").strip().lower()
                rating = row.get("rating")
                # Preserve the row for potential future use
                rows.append(row)
                if occ:
                    # Ratings are stored as strings; preserve as-is for now
                    mapping[occ] = rating
    except Exception:
        pass
    return rows, mapping

# Load synonyms and sector keywords
@st.cache_data
def load_aux():
    import yaml
    with open("synonyms.yaml", "r", encoding="utf-8") as f:
        syn = yaml.safe_load(f) or {}
    with open("sector_keywords.json", "r", encoding="utf-8") as f:
        sekt = json.load(f)
    return syn, sekt

DICT_DF, DICT_MAP = load_risk_dict()
SYNONYMS, SECTOR_KW = load_aux()
CANONICALS = set(DICT_MAP.keys())

# Tier definitions reflect the insurer's appetite for risk. Scores 1–3 are low
# risk and within appetite. Scores 4–8 warrant a closer look but are often
# acceptable with caution. A score of 9 is very high risk but might still be
# placeable, whereas 10 indicates the business is out of appetite altogether.
TIERS = [
    (1, 3, "Low / In appetite", "green"),
    (4, 8, "Medium / Needs review", "orange"),
    (9, 9, "Very High / Potentially placeable", "red"),
    (10, 10, "Out of appetite / Decline", "black"),
]

def tier_for(score:int):
    for lo, hi, label, color in TIERS:
        if lo <= score <= hi:
            return label, color
    return "Unknown","grey"

def sanitize_url(u):
    if not u:
        return None
    if not u.startswith("http"):
        u = "https://" + u
    return u

def same_domain(u, base):
    try:
        return urlparse(u).netloc == urlparse(base).netloc
    except:
        return False

def fetch(url, timeout=12):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; RiskAnalyzer/1.0)"}
        r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        if r.status_code == 200 and r.content:
            return r.text
    except Exception as e:
        return None
    return None

def extract_links_and_text(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text)
    # Candidate internal links
    links = []
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"])
        if same_domain(href, base_url):
            links.append(href)
    # Prioritize likely pages
    wanted = ["about", "service", "services", "what-we-do", "industr", "product", "sectors", "our-work"]
    priority = [l for l in links if any(w in l.lower() for w in wanted)]
    # unique preserve order
    seen = set()
    ordered = []
    for l in priority + links:
        if l not in seen:
            seen.add(l); ordered.append(l)
    return ordered[:12], text

def normalize_phrase(s):
    return s.strip().lower()

def detect_candidates(all_text):
    # Exact matches from canonical dict
    found = {}
    text_norm = all_text.lower()
    for name_norm, rating in DICT_MAP.items():
        # strict phrase match boundaries (allow hyphen/space variants)
        pattern = r'\b' + re.escape(name_norm) + r'\b'
        if re.search(pattern, text_norm):
            found[name_norm] = {"name": name_norm, "rating": int(rating), "source": "exact"}
    # Synonym expansion
    for syn, canonical in (SYNONYMS or {}).items():
        syn_norm = normalize_phrase(syn)
        can_norm = normalize_phrase(canonical)
        if syn_norm in text_norm and can_norm in CANONICALS:
            found[can_norm] = {"name": can_norm, "rating": int(DICT_MAP[can_norm]), "source": "synonym:"+syn}
    # Fuzzy (guarded): only if high similarity and appears near "our services"/"we provide"
    # We'll scan common service phrases windows
    windows = re.findall(r"(?:our services|we (?:provide|offer)|services include).{0,200}", text_norm)
    candidates: set[str] = set()
    # Use custom simple partial ratio instead of rapidfuzz. Only consider a match
    # if the similarity score is 92% or higher.
    for w in windows:
        best_match, score = _extract_one(w, list(CANONICALS))
        if best_match and score >= 92:
            candidates.add(best_match)
    for c in candidates:
        found[c] = {"name": c, "rating": int(DICT_MAP[c]), "source": "fuzzy-guarded"}
    return found

def prominence_rank(text, candidates):
    """Rank candidates by prominence: title/H1 (x5), headings (x3), frequency (x1)."""
    weights = {}
    # naive heading detection via markup keywords; we don't have original markup here, but keep heuristic
    t = text.lower()
    # Title weight
    title_matches = []
    # headings approximated by capitalized words preceding line breaks removed in our text; we fallback to frequency
    for c in candidates:
        c_norm = c
        freq = t.count(c_norm)
        w = freq
        if re.search(r"\b" + re.escape(c_norm) + r"\b", t[:400]):  # early text/hero/title
            w += 5
        if re.search(r"(services|what we do).{0,120}" + re.escape(c_norm), t):
            w += 3
        weights[c] = w
    ranked = sorted(candidates, key=lambda x: (-weights.get(x,0), -candidates[x]["rating"]))  # tie-break by higher risk
    return ranked, weights

def evidence_snippets(text, term, max_snips=3):
    tn = term.lower()
    spans = []
    for m in re.finditer(re.escape(tn), text.lower()):
        start = max(0, m.start()-120)
        end = min(len(text), m.end()+120)
        spans.append(text[start:end])
        if len(spans) >= max_snips:
            break
    return ["…"+s.strip()+"…" for s in spans]

def choose_sector(text):
    t = text.lower()
    best = None; best_score = 0
    for sector, kws in (SECTOR_KW or {}).items():
        score = 0
        for kw in kws:
            score += t.count(kw.lower())
        if score > best_score:
            best_score = score; best = sector
    # Fallback to Services if tie
    return best or "Services"

def analyze(url):
    url = sanitize_url(url)
    if not url:
        return {"error":"Invalid URL"}
    html = fetch(url)
    if not html:
        return {"error":"Could not fetch URL. If site is JS-heavy, deploy with a JS-rendering crawler."}
    links, text = extract_links_and_text(html, url)

    # include some secondary pages
    collected_texts = [text]
    for link in links[:6]:
        html2 = fetch(link)
        if html2:
            _, t2 = extract_links_and_text(html2, link)
            collected_texts.append(t2)

    all_text = " ".join(collected_texts)
    # Detect potential occupations in the aggregated text.
    found = detect_candidates(all_text)
    # If no occupation is detected, fall back to a sector-based risk rating.
    if not found:
            # When no specific occupations are detected from the page text, fall back to a
            # sector-based risk score. We map the chosen sector to a default risk score
            # representative of the typical risk exposure for that industry. This ensures
            # that every business receives a numeric score even without an exact
            # occupation match.
            sector = choose_sector(all_text)
            # Define a default risk score for each sector. Higher numbers indicate
            # greater inherent risk. These values can be tuned as more data becomes
            # available.
            sector_risk = {
                "Construction": 7,
                "Electrical and electronic equipment": 5,
                "Wholesale": 4,
                "Retail": 3,
                "Manufacturing": 5,
                "Hospitality": 4,
                "Leisure and recreation": 3,
                "Farming": 5,
                "Professional services": 2,
                "Distribution": 5,
                "Waste and sewage services": 7,
                "Public authorities and defense": 4,
                "Utilities – energy, water, and telecoms": 7,
                "Metalworking, engineering, and machinery": 6,
                "Motor trade": 6,
                "Woodworking": 5,
                "Food and drink": 4,
                "Services": 3,
                "Railways, roads, waterways, marine, and aviation services": 7,
                "Printing and paper": 4,
                "Chemicals, oils, and gas": 8,
                "Mining and quarrying": 8,
                "Plastics and rubber": 5,
                "Ceramics and glass": 5,
                "Clothing, footwear, textiles, and soft furnishings": 4,
                "Education": 3,
                "Health, hospitals, and care": 4,
                "Property owners": 3,
                "Estates": 3,
                "Consumer and business services": 3,
            }
            # Use the sector-based score if available; otherwise default to a low risk of 3.
            fallback_score = sector_risk.get(sector, 3)
            tier_label, _ = tier_for(fallback_score)
            # Try to guess a business name using the same heuristic as when a core
            # occupation is found. This attempts to extract a capitalised phrase from
            # the page's text, using the first collected text (typically the home page).
            name_guess = urlparse(url).netloc
            m = re.search(r"([A-Z][A-Za-z0-9&' ]{2,60})(?:\s[-|•|–])", collected_texts[0])
            if m:
                name_guess = m.group(1).strip()
            notes = (
                "No qualifying occupations detected; assigned risk score based on the "
                f"business sector '{sector}' using fallback mapping. "
                "The rating scale is 1–3 = Low/In appetite, 4–8 = Medium/Needs review, "
                "9 = Very High/Potentially placeable, and 10 = Out of appetite/Decline."
            )
            return {
                "url": url,
                "business_name": name_guess,
                "score": fallback_score,
                "tier": tier_label,
                "sector": sector,
                "core": None,
                "flags": [],
                "notes": notes,
                "evidence": []
            }

    ranked, weights = prominence_rank(all_text, found)
    core_key = ranked[0]
    core = found[core_key]
    score = int(core["rating"])
    tier_label, tier_color = tier_for(score)

    # Additional flags: other items with very high ratings (≥9)
    flags = []
    for k, v in found.items():
        if k == core_key:
            continue
        # Only treat ratings of 9 or above as high-risk flags under the revised scale.
        if int(v["rating"]) >= 9:
            flags.append({"occupation": k, "rating": int(v["rating"])})
    # Evidence
    ev = evidence_snippets(all_text, core_key, 3)

    # Business name heuristic: title tag or H1 approximation
    name_guess = urlparse(url).netloc
    m = re.search(r"([A-Z][A-Za-z0-9&' ]{2,60})(?:\s[-|•|–])", collected_texts[0])
    if m:
        name_guess = m.group(1).strip()

    # Notes
    notes = (
        f"Core activity selected by prominence weighting and on-site phrasing. "
        f"Matched '{core_key}' via {core.get('source')} with rating {score} from Document10. "
        "The rating scale is 1–3 = Low/In appetite, 4–8 = Medium/Needs review, "
        "9 = Very High/Potentially placeable, and 10 = Out of appetite/Decline. "
        + ("Additional very high-risk flags present." if flags else "No additional very high-risk flags detected.")
    )

    result = {
        "url": url,
        "business_name": name_guess,
        "score": score,
        "tier": tier_label,
        "sector": choose_sector(all_text),
        "core": {"occupation": core_key, "rating": score, "source": core.get("source")},
        "flags": flags,
        "notes": notes,
        "evidence": ev,
    }
    return result

st.title("Insurance Risk Analyzer")
st.caption("URL → One core occupation → Document10 risk score (1–10) with evidence. No averaging.")

qs = st.experimental_get_query_params()
prefill = (qs.get("url") or [""])[0]

url = st.text_input("Website URL", value=prefill, placeholder="https://example.com")
run = st.button("Analyze Risk Level")

if run and url:
    with st.spinner("Analyzing..."):
        res = analyze(url)
    if "error" in res:
        st.error(res["error"])
    else:
        # Output card
        col1, col2 = st.columns([3,1])
        with col1:
            st.subheader(res["business_name"] or url)
            st.write(res["url"])
        with col2:
                # Even if the core occupation is missing, we still want to display a
                # risk indicator. If a core score exists, use it. Otherwise, if
                # there are any high‑risk flags (rating ≥7), use the highest
                # flagged rating as a proxy for the overall risk. If neither exist,
                # leave the score as None.
                score_display = res["score"]
                if score_display is None and res.get("flags"):
                    try:
                        score_display = max(f.get("rating") or 0 for f in res["flags"])
                    except ValueError:
                        score_display = None

                if score_display is not None:
                    # Determine tier label and colour for the risk score
                    tier_lbl, color = tier_for(score_display)
                    # Display the raw numeric score using the built‑in metric widget
                    st.metric(label="Risk Score (1–10)", value=score_display)
                    # Highlight the tier using Streamlit status messages.
                    # Scores ≥9 are considered very high or out-of-appetite and are shown as an error.
                    # Scores 4–8 warrant caution and are shown as a warning, while 1–3 are within appetite.
                    if score_display >= 9:
                        st.error(f"{tier_lbl} (score {score_display}/10)")
                    elif score_display >= 4:
                        st.warning(f"{tier_lbl} (score {score_display}/10)")
                    else:
                        st.success(f"{tier_lbl} (score {score_display}/10)")
                else:
                    # If we have no risk information at all, display a neutral metric and info
                    st.metric(label="Risk Score (1–10)", value="N/A")
                    st.info("No qualifying occupations detected; risk score unavailable")
        st.divider()
        st.markdown("**Business Sector (chosen)**")
        st.write(res["sector"])
        st.markdown("**Detected Occupation/Industry (core)**")
        # Only display the core occupation and rating if it exists. When no matching
        # occupations are found, res["core"] will be None. Without this check,
        # attempting to index into None would raise a TypeError (as reported). If
        # there is no core occupation, inform the user instead of crashing.
        if res.get("core"):
            occ = res['core']['occupation']
            rating = res['core']['rating']
            # Highlight the core occupation based on its risk rating.
            # Colour-code the core occupation rating according to the revised scale:
            # 9–10: very high/out of appetite → error; 4–8: medium → warning; 1–3: low → success
            if rating >= 9:
                st.error(f"{occ} — rating {rating}")
            elif rating >= 4:
                st.warning(f"{occ} — rating {rating}")
            else:
                st.success(f"{occ} — rating {rating}")
        else:
            st.write("No qualifying occupations detected")
        if res["flags"]:
            st.markdown("**Additional High‑Risk Flags (≥9, advisory only)**")
            # Display each flagged occupation as an error to emphasise its very high risk.
            for f in res["flags"]:
                st.error(f"{f['occupation']} — {f['rating']}")
        st.markdown("**Reasoning/Notes**")
        st.write(res["notes"])
        st.markdown("**Evidence Snippets**")
        for s in res["evidence"]:
            st.code(s)

        # Export
        export = {
            "Business Name": res["business_name"],
            "Website URL": res["url"],
            "Risk Score": res["score"],
            "Risk Tier": res["tier"],
            "Business Sector": res["sector"],
            # Safely export the core occupation; if none exists, leave blank.
            "Detected Occupation/Industry (core)": res["core"]["occupation"] if res.get("core") else "",
            "Additional Flags": "; ".join([f"{f['occupation']}({f['rating']})" for f in res["flags"]]) if res["flags"] else "",
            "Reasoning/Notes": res["notes"],
            "Evidence": " | ".join(res["evidence"]),
        }
        # Build CSV data manually to avoid the pandas dependency. We create a CSV
        # string with a header row and a single data row using the built-in csv module.
        import io
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(export.keys()))
        writer.writeheader()
        writer.writerow(export)
        csv_data = output.getvalue()
        st.download_button("Download CSV", data=csv_data.encode("utf-8"), file_name="assessment.csv", mime="text/csv")

        # Shareable link
        base = os.getenv("PUBLIC_BASE_URL", "")  # if deployed, set this; otherwise fallback to query param link
        if base:
            share_url = f"{base}?url={requests.utils.quote(res['url'])}"
        else:
            # local/share via query param — user copies current app URL and append ?url=
            share_url = f"?url={res['url']}"
        st.text_input("Shareable link (paste into browser)", value=share_url)
else:
    st.info("Enter a website URL and click Analyze. You can also append '?url=https://example.com' to the app URL to share a direct assessment link.")

st.markdown("---")
st.caption("Scoring dictionary loaded from Document10 (uploaded). Edit 'synonyms.yaml' and 'sector_keywords.json' to tune matching and sector choice.")
