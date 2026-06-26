import streamlit as st
import fitz
import docx
from groq import Groq
import tempfile
import os
import re
import hashlib
import json

API_KEY = st.secrets["GROQ_API_KEY"]

# ══════════════════════════════════════════════════════════════
# PROMPTS — two-pass for accuracy + consistency
# ══════════════════════════════════════════════════════════════

# Pass 1: Extract structured data from both documents deterministically
EXTRACT_SYSTEM = """You are a precise document parser. Extract structured information from a resume and job description.

Respond in EXACTLY this JSON format, nothing else, no markdown fences:
{
  "jd_keywords": ["list", "of", "every", "significant", "skill", "tool", "technology", "keyword", "from", "JD"],
  "jd_required_skills": ["explicit", "required", "skills", "from", "JD"],
  "jd_preferred_skills": ["preferred", "or", "nice-to-have", "skills"],
  "jd_min_years": <number or null>,
  "jd_education": "<required education level or null>",
  "jd_certifications": ["required", "certs"],
  "resume_keywords": ["every", "keyword", "skill", "tool", "technology", "found", "in", "resume"],
  "resume_skills": ["explicit", "skills", "in", "resume"],
  "resume_years_exp": <estimated years of relevant experience as number>,
  "resume_education": "<highest education found>",
  "resume_certifications": ["certs", "found"],
  "resume_role_domain": "<domain or industry of resume>",
  "jd_role_domain": "<domain or industry of JD>",
  "has_formatting_issues": <true or false>
}"""

# Pass 2: Score deterministically from extracted data
SCORE_SYSTEM = """You are an ATS scoring engine. You receive pre-extracted structured data and must score deterministically.

SCORING RULES (follow exactly):

1. KEYWORD MATCH (0-40pts):
   matched = count of jd_keywords that appear in resume_keywords (case-insensitive, allow plural/variant)
   total = count of jd_keywords
   keyword_score = round((matched / total) * 40, 1)

2. SKILLS ALIGNMENT (0-25pts):
   matched_skills = count of jd_required_skills found in resume_skills
   total_skills = count of jd_required_skills
   skills_score = round((matched_skills / total_skills) * 25, 1)

3. EXPERIENCE RELEVANCE (0-20pts):
   - If resume_years_exp >= jd_min_years AND same domain: 20
   - If resume_years_exp >= jd_min_years but different domain: 12
   - If resume_years_exp < jd_min_years by 1-2 years AND same domain: 14
   - If resume_years_exp < jd_min_years by 1-2 years, diff domain: 8
   - If resume_years_exp < jd_min_years by 3+ years: 5
   - If jd_min_years is null: score 15 if same domain, 8 if different

4. EDUCATION (0-10pts):
   - Exact match or higher: 10
   - One level below (e.g. diploma vs degree): 6
   - Two levels below: 3
   - No match: 0

5. FORMAT (0-5pts):
   - has_formatting_issues = false: 5
   - has_formatting_issues = true: 2

total_score = keyword_score + skills_score + experience_score + education_score + format_score

For FIXES: each fix must name the EXACT keyword or skill missing AND a concrete instruction of where/how to add it in the resume. Make fixes highly actionable — if someone follows them literally, the score must go up.

Respond in EXACTLY this format, nothing else:

SCORE: [X.X/100]
BREAKDOWN:
- Keywords: [X.X/40] ([matched]/[total] matched)
- Skills: [X.X/25] ([matched_skills]/[total_skills] matched)
- Experience: [X.X/20]
- Education: [X.X/10]
- Format: [X/5]
MATCHED: [comma-separated matched keywords, lowercase]
MISSING: [comma-separated unmatched jd_keywords, lowercase]
STRENGTHS:
- [strength backed by specific resume evidence]
- [strength backed by specific resume evidence]
FIXES:
1. [KEYWORD: exact term] — [where to add it and how, e.g. "Add 'Kubernetes' to your Skills section and mention it in your most recent job description where you managed containers"]
2. [KEYWORD: exact term] — [where to add it and how]
3. [KEYWORD: exact term] — [where to add it and how]"""


# ══════════════════════════════════════════════════════════════
# FILE READING
# ══════════════════════════════════════════════════════════════

def read_file(file_obj, filename):
    suffix = os.path.splitext(filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_obj.read() if hasattr(file_obj, 'read') else file_obj)
        tmp_path = tmp.name
    text = ""
    try:
        if suffix == ".pdf":
            doc = fitz.open(tmp_path)
            for page in doc:
                text += page.get_text()
            doc.close()
        elif suffix in (".docx", ".doc"):
            doc = docx.Document(tmp_path)
            text = "\n".join([p.text for p in doc.paragraphs])
        elif suffix == ".txt":
            with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass
    return text.strip()

def trim(text, max_chars):
    return text[:max_chars] + "\n[truncated]" if len(text) > max_chars else text

def file_hash(file_obj):
    """Stable hash of file bytes for caching."""
    file_obj.seek(0)
    h = hashlib.md5(file_obj.read()).hexdigest()
    file_obj.seek(0)
    return h

def parse_score(text):
    m = re.search(r'SCORE:\s*(\d+(?:\.\d+)?)/100', text)
    return float(m.group(1)) if m else None

def score_color(score, dark):
    if score is None: return "#888"
    if score >= 75: return "#16a34a" if not dark else "#22c55e"
    if score >= 50: return "#d97706" if not dark else "#f59e0b"
    return "#dc2626" if not dark else "#ef4444"

def score_label(score):
    if score is None: return "—"
    if score >= 75: return "Strong"
    if score >= 50: return "Fair"
    return "Weak"


# ══════════════════════════════════════════════════════════════
# TWO-PASS GROQ SCORING
# ══════════════════════════════════════════════════════════════

def run_extract(resume_text, jd_text):
    client = Groq(api_key=API_KEY)
    prompt = f"RESUME:\n{trim(resume_text, 2200)}\n\nJOB DESCRIPTION:\n{trim(jd_text, 1500)}"
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": prompt}
        ],
        max_tokens=800,
        temperature=0,
        seed=42,
    )
    return response.choices[0].message.content

def run_score(extracted_json):
    client = Groq(api_key=API_KEY)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SCORE_SYSTEM},
            {"role": "user", "content": f"EXTRACTED DATA:\n{extracted_json}"}
        ],
        max_tokens=700,
        temperature=0,
        seed=42,
    )
    return response.choices[0].message.content

def run_ats(resume_text, jd_text):
    extracted = run_extract(resume_text, jd_text)
    # Validate it's parseable JSON (best effort)
    try:
        json.loads(extracted)
    except Exception:
        # If extraction failed, strip markdown fences and retry parse
        extracted = re.sub(r'```json|```', '', extracted).strip()
    result = run_score(extracted)
    return result


# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ATSlens",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Session state ─────────────────────────────────────────────
for key, default in {
    "results": [],
    "dark": True,
    "cache": {},      # {resume_hash + jd_hash: result}
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

dark = st.session_state.dark

# ── Theme values ──────────────────────────────────────────────
if dark:
    BG        = "#0a0a0f"
    SURFACE   = "#13131f"
    SURFACE2  = "#0f0f1a"
    BORDER    = "#1e1e2e"
    BORDER2   = "#2a2a40"
    TEXT      = "#e8e6e0"
    TEXT2     = "#b0aec8"
    MUTED     = "#5a5a78"
    MUTED2    = "#3d3d5c"
    ACCENT    = "#7c6fff"
    SCROLLBG  = "#0a0a0f"
    SCROLLTHUMB = "#2a2a40"
    EXPBG     = "#13131f"
    SUMTEXT   = "#b0aec8"
else:
    BG        = "#f8f7f4"
    SURFACE   = "#ffffff"
    SURFACE2  = "#f2f1ee"
    BORDER    = "#e2e0d8"
    BORDER2   = "#ccc9be"
    TEXT      = "#1a1a2e"
    TEXT2     = "#3d3b52"
    MUTED     = "#7a7890"
    MUTED2    = "#a8a6b8"
    ACCENT    = "#5b4de8"
    SCROLLBG  = "#f8f7f4"
    SCROLLTHUMB = "#ccc9be"
    EXPBG     = "#ffffff"
    SUMTEXT   = "#3d3b52"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Space+Grotesk:wght@400;500;600&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
#MainMenu, footer, header {{ visibility: hidden; }}
.stDeployButton {{ display: none; }}

.stApp {{ background: {BG}; color: {TEXT}; }}

.top-bar {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 1rem 2rem; border-bottom: 1px solid {BORDER}; margin-bottom: 2rem;
}}
.logo {{
    font-family: 'Space Grotesk', sans-serif; font-size: 1.1rem;
    font-weight: 600; color: {TEXT}; letter-spacing: 0.05em;
}}
.logo span {{ color: {ACCENT}; }}

.hero {{ text-align: center; padding: 2rem 1rem 2.5rem; max-width: 620px; margin: 0 auto; }}
.hero-eyebrow {{
    font-size: 0.68rem; font-weight: 500; letter-spacing: 0.15em;
    color: {ACCENT}; text-transform: uppercase; margin-bottom: 0.8rem;
}}
.hero-title {{
    font-family: 'Space Grotesk', sans-serif; font-size: 2.4rem;
    font-weight: 600; line-height: 1.15; color: {TEXT}; margin-bottom: 0.6rem;
}}
.hero-title em {{ font-style: normal; color: {ACCENT}; }}
.hero-sub {{ font-size: 0.9rem; color: {MUTED}; line-height: 1.6; }}

.upload-label {{
    font-size: 0.7rem; font-weight: 500; letter-spacing: 0.08em;
    color: {MUTED}; text-transform: uppercase; margin-bottom: 0.4rem;
}}

.chip {{
    display: inline-flex; align-items: center; gap: 6px;
    padding: 0.2rem 0.65rem; border-radius: 20px; font-size: 0.72rem; font-weight: 500;
}}
.chip-success {{ background: rgba(34,197,94,0.1); color: #22c55e; border: 1px solid rgba(34,197,94,0.2); }}
.chip-neutral {{ background: rgba(124,111,255,0.08); color: {ACCENT}; border: 1px solid rgba(124,111,255,0.2); }}

.score-box {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px;
    padding: 1.4rem; text-align: center; margin-bottom: 0.8rem;
}}
.score-bar-track {{
    background: {BORDER}; border-radius: 4px; height: 5px; margin: 0.6rem 0; overflow: hidden;
}}
.breakdown-row {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.38rem 0; border-bottom: 1px solid {BORDER}; font-size: 0.81rem;
}}
.breakdown-row:last-child {{ border-bottom: none; }}
.breakdown-key {{ color: {MUTED}; }}
.breakdown-val {{ color: {TEXT}; font-weight: 500; }}

.tag {{
    display: inline-block; padding: 0.18rem 0.5rem; border-radius: 5px;
    font-size: 0.68rem; font-weight: 500; margin: 2px;
}}
.tag-match {{ background: rgba(34,197,94,0.08); color: #16a34a; border: 1px solid rgba(34,197,94,0.18); }}
.tag-miss  {{ background: rgba(239,68,68,0.08); color: #dc2626; border: 1px solid rgba(239,68,68,0.18); }}

.section-label {{
    font-size: 0.65rem; font-weight: 500; letter-spacing: 0.1em; color: {MUTED2};
    text-transform: uppercase; margin-bottom: 0.45rem;
    padding-bottom: 0.35rem; border-bottom: 1px solid {BORDER};
}}

.fix-row {{
    display: flex; gap: 10px; padding: 0.55rem 0;
    border-bottom: 1px solid {BORDER}; align-items: flex-start;
}}
.fix-num {{
    background: {ACCENT}; color: #fff; border-radius: 50%; width: 18px; height: 18px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.65rem; font-weight: 600; min-width: 18px; margin-top: 1px;
}}
.fix-kw {{ color: {ACCENT}; font-weight: 600; font-size: 0.78rem; }}
.fix-body {{ font-size: 0.8rem; color: {SUMTEXT}; line-height: 1.5; }}

div[data-testid="stFileUploader"] {{
    background: {SURFACE2} !important; border: 1px dashed {BORDER2} !important;
    border-radius: 10px !important;
}}
div[data-testid="stFileUploader"]:hover {{ border-color: {ACCENT} !important; }}
div[data-testid="stFileUploader"] label {{ color: {MUTED} !important; }}

div[data-testid="stButton"] > button {{
    background: {ACCENT} !important; color: #fff !important; border: none !important;
    border-radius: 10px !important; font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important; font-size: 0.9rem !important;
    padding: 0.65rem 2rem !important; width: 100% !important;
    letter-spacing: 0.02em !important; transition: opacity 0.15s !important;
}}
div[data-testid="stButton"] > button:hover {{ opacity: 0.85 !important; }}
div[data-testid="stButton"] > button:disabled {{
    background: {BORDER2} !important; color: {MUTED2} !important;
}}

div[data-testid="stExpander"] {{
    background: {EXPBG} !important; border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
}}
details summary {{ color: {TEXT2} !important; }}
.stSpinner > div {{ border-top-color: {ACCENT} !important; }}
hr {{ border-color: {BORDER} !important; margin: 1.5rem 0 !important; }}

section[data-testid="stSidebar"] {{
    background: {BG}; border-right: 1px solid {BORDER};
}}
::-webkit-scrollbar {{ width: 5px; }}
::-webkit-scrollbar-track {{ background: {SCROLLBG}; }}
::-webkit-scrollbar-thumb {{ background: {SCROLLTHUMB}; border-radius: 3px; }}

/* Toggle button override — smaller, right-aligned */
div[data-testid="stButton"].toggle-btn > button {{
    background: {SURFACE} !important; color: {TEXT} !important;
    border: 1px solid {BORDER2} !important; border-radius: 20px !important;
    font-size: 0.75rem !important; padding: 0.25rem 0.8rem !important;
    width: auto !important;
}}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TOP BAR with theme toggle
# ══════════════════════════════════════════════════════════════

bar_left, bar_mid, bar_right = st.columns([2, 6, 2])
with bar_left:
    st.markdown(f'<div class="logo" style="padding-top:0.5rem">◈ ATS<span>lens</span></div>', unsafe_allow_html=True)
with bar_right:
    toggle_label = "☀ Light" if dark else "☾ Dark"
    if st.button(toggle_label, key="theme_toggle"):
        st.session_state.dark = not dark
        st.rerun()

st.markdown(f'<hr style="margin-top:0">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="hero">
    <div class="hero-eyebrow">Powered by Groq · LLaMA 3.3 · Two-pass scoring</div>
    <div class="hero-title">Screen resumes<br>at <em>scale</em></div>
    <div class="hero-sub">Upload a job description and any number of candidate resumes.<br>
    Scores are deterministic — same resume always gives the same score.</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# UPLOAD SECTION
# ══════════════════════════════════════════════════════════════

col_left, col_right = st.columns([1, 1], gap="medium")

with col_left:
    st.markdown(f'<div class="upload-label">Job description</div>', unsafe_allow_html=True)
    jd_file = st.file_uploader(
        "Upload JD", type=["pdf", "docx", "doc", "txt"],
        key="jd", label_visibility="collapsed"
    )
    if jd_file:
        jd_file.seek(0)
        jd_text = read_file(jd_file, jd_file.name)
        jd_hash = file_hash(jd_file)
        st.markdown(f'<span class="chip chip-success">✓ {jd_file.name}</span>', unsafe_allow_html=True)
        st.caption(f"{len(jd_text):,} chars · {len(jd_text.split()):,} words")
    else:
        jd_text = ""
        jd_hash = ""

with col_right:
    st.markdown(f'<div class="upload-label">Candidate resumes</div>', unsafe_allow_html=True)
    resume_files = st.file_uploader(
        "Upload Resumes", type=["pdf", "docx", "doc"],
        accept_multiple_files=True, key="resumes", label_visibility="collapsed"
    )
    if resume_files:
        st.markdown(
            f'<span class="chip chip-neutral">◈ {len(resume_files)} resume{"s" if len(resume_files)>1 else ""} ready</span>',
            unsafe_allow_html=True
        )

st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# ANALYZE BUTTON
# ══════════════════════════════════════════════════════════════

can_run = bool(jd_text and resume_files)

if st.button(
    "Run ATS Analysis" if can_run else "Upload files to begin",
    disabled=not can_run,
    use_container_width=True
):
    st.session_state.results = []
    progress = st.progress(0, text="Starting analysis…")

    for i, rf in enumerate(resume_files):
        cache_key = file_hash(rf) + "_" + jd_hash

        if cache_key in st.session_state.cache:
            # Use cached result — guaranteed same score
            cached = st.session_state.cache[cache_key]
            st.session_state.results.append(cached)
        else:
            rf.seek(0)
            resume_text = read_file(rf, rf.name)
            progress.progress((i + 0.5) / len(resume_files), text=f"Scoring {rf.name}…")
            try:
                result = run_ats(resume_text, jd_text)
                score = parse_score(result)
            except Exception as e:
                result = f"Error: {e}"
                score = None

            entry = {"name": rf.name, "result": result, "score": score}
            st.session_state.cache[cache_key] = entry
            st.session_state.results.append(entry)

        progress.progress((i + 1) / len(resume_files), text=f"Done {i+1}/{len(resume_files)}")

    progress.empty()
    st.session_state.results.sort(key=lambda x: x["score"] or 0, reverse=True)

# ══════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════

if st.session_state.results:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='section-label'>Results · {len(st.session_state.results)} candidate{'s' if len(st.session_state.results)>1 else ''} ranked</div>",
        unsafe_allow_html=True
    )

    for idx, r in enumerate(st.session_state.results):
        score  = r["score"]
        color  = score_color(score, dark)
        label  = score_label(score)
        sdisplay = f"{score:.1f}" if score is not None else "—"
        pct    = min(score or 0, 100)
        raw    = r["result"]

        def extract(pattern, text, default=""):
            m = re.search(pattern, text, re.DOTALL)
            return m.group(1).strip() if m else default

        breakdown_raw = extract(r'BREAKDOWN:(.*?)(?=MATCHED:|$)', raw)
        matched_raw   = extract(r'MATCHED:(.*?)(?=MISSING:|$)', raw)
        missing_raw   = extract(r'MISSING:(.*?)(?=STRENGTHS:|$)', raw)
        strengths_raw = extract(r'STRENGTHS:(.*?)(?=FIXES:|$)', raw)
        fixes_raw     = extract(r'FIXES:(.*?)$', raw)

        clean_name = re.sub(r'\.(pdf|docx|doc)$', '', r['name'], flags=re.IGNORECASE)
        medal = ["🥇","🥈","🥉"][idx] if idx < 3 else "◈"
        cached_tag = " · ⚡ cached" if idx < len(st.session_state.results) else ""

        with st.expander(
            f"{medal}  {clean_name}  ·  {sdisplay}/100  ·  {label}",
            expanded=(idx == 0)
        ):
            c1, c2, c3 = st.columns([1, 2, 2])

            # ── Score box ──────────────────────────────────────
            with c1:
                st.markdown(f"""
<div class="score-box">
  <div style="font-family:'Space Grotesk',sans-serif;font-size:2.8rem;font-weight:600;color:{color};line-height:1">{sdisplay}</div>
  <div style="font-size:0.68rem;color:{MUTED2};margin-top:3px">/ 100</div>
  <div style="font-size:0.7rem;font-weight:600;color:{color};margin-top:7px;letter-spacing:0.06em">{label.upper()}</div>
  <div class="score-bar-track" style="margin-top:10px">
    <div style="background:{color};width:{pct}%;height:100%;border-radius:4px"></div>
  </div>
</div>""", unsafe_allow_html=True)

            # ── Breakdown ──────────────────────────────────────
            with c2:
                st.markdown('<div class="section-label">Score breakdown</div>', unsafe_allow_html=True)
                if breakdown_raw:
                    for line in breakdown_raw.strip().split('\n'):
                        line = line.strip().lstrip('- ')
                        if ':' in line:
                            k, v = line.split(':', 1)
                            st.markdown(f"""
<div class="breakdown-row">
  <span class="breakdown-key">{k.strip()}</span>
  <span class="breakdown-val">{v.strip()}</span>
</div>""", unsafe_allow_html=True)

            # ── Strengths ──────────────────────────────────────
            with c3:
                st.markdown('<div class="section-label">Strengths</div>', unsafe_allow_html=True)
                if strengths_raw:
                    for line in strengths_raw.strip().split('\n'):
                        line = line.strip().lstrip('- ')
                        if line:
                            st.markdown(
                                f"<div style='font-size:0.81rem;color:{SUMTEXT};padding:0.28rem 0;border-bottom:1px solid {BORDER}'>→ {line}</div>",
                                unsafe_allow_html=True
                            )

            # ── Keywords ───────────────────────────────────────
            kw1, kw2 = st.columns(2)
            with kw1:
                st.markdown(f'<div class="section-label" style="margin-top:1rem">Matched keywords</div>', unsafe_allow_html=True)
                if matched_raw:
                    tags = [t.strip() for t in matched_raw.split(',') if t.strip()]
                    st.markdown("".join(f'<span class="tag tag-match">{t}</span>' for t in tags[:25]), unsafe_allow_html=True)
            with kw2:
                st.markdown(f'<div class="section-label" style="margin-top:1rem">Missing keywords</div>', unsafe_allow_html=True)
                if missing_raw:
                    tags = [t.strip() for t in missing_raw.split(',') if t.strip()]
                    st.markdown("".join(f'<span class="tag tag-miss">{t}</span>' for t in tags[:25]), unsafe_allow_html=True)

            # ── Fixes ──────────────────────────────────────────
            if fixes_raw:
                st.markdown(f'<div class="section-label" style="margin-top:1.2rem">3 changes that will raise your score</div>', unsafe_allow_html=True)
                fix_lines = [l.strip() for l in fixes_raw.strip().split('\n') if l.strip()]
                for fi, line in enumerate(fix_lines[:3], 1):
                    line = re.sub(r'^\d+[\.\)]\s*', '', line)
                    # Split on " — " to highlight keyword separately
                    parts = line.split(' — ', 1)
                    kw_part = parts[0].strip() if parts else line
                    body_part = parts[1].strip() if len(parts) > 1 else ""
                    st.markdown(f"""
<div class="fix-row">
  <div class="fix-num">{fi}</div>
  <div>
    <div class="fix-kw">{kw_part}</div>
    <div class="fix-body">{body_part}</div>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    if st.button("Clear results", key="clear"):
        st.session_state.results = []
        st.rerun()
