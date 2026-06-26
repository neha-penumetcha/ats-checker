import streamlit as st
import fitz
import docx
from groq import Groq
import tempfile
import os
import re

API_KEY = st.secrets["GROQ_API_KEY"]

# ── Prompt ────────────────────────────────────────────────────

ATS_SYSTEM = """You are an ATS scoring engine. Follow this exact formula — no estimation, no rounding to neat numbers.

SCORING FORMULA (execute in order):

1. KEYWORD MATCH (0-40pts):
   - Extract every significant noun, skill, tool, technology, and qualification from the JD
   - Count exactly how many appear in the resume (exact match or clear variant)
   - Score = (matched_count / total_jd_keywords) * 40, round to 1 decimal

2. SKILLS ALIGNMENT (0-25pts):
   - List required skills from JD
   - Count how many the resume explicitly demonstrates
   - Score = (matching_skills / total_required_skills) * 25, round to 1 decimal

3. EXPERIENCE RELEVANCE (0-20pts):
   - Compare role type, domain, seniority level, and years of experience
   - Score 0-20 based on fit, with one specific justification

4. EDUCATION & CERTIFICATIONS (0-10pts):
   - Does the resume meet the JD education/cert requirements exactly?
   - Score 0-10 based on match

5. FORMATTING (0-5pts):
   - Clean, parseable, no tables/images/headers breaking flow: 5pts
   - Minor issues: 3pts
   - Major issues: 1pt

Add all five for TOTAL. Do not round the total.

Respond in EXACTLY this format, nothing else:

SCORE: [X.X/100]
BREAKDOWN:
- Keywords: [X.X/40]
- Skills: [X.X/25]
- Experience: [X.X/20]
- Education: [X.X/10]
- Format: [X/5]
MATCHED: [comma-separated lowercase keywords found in resume]
MISSING: [comma-separated lowercase keywords not found in resume]
STRENGTHS:
- [specific strength with direct evidence from resume text]
- [specific strength with direct evidence from resume text]
FIXES:
1. [exact missing keyword or skill to add, with where to add it]
2. [exact missing keyword or skill to add, with where to add it]
3. [exact missing keyword or skill to add, with where to add it]"""

# ── File reading ──────────────────────────────────────────────

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

def trim(text, max_chars=2500):
    return text[:max_chars] + "\n[truncated]" if len(text) > max_chars else text

def parse_score(text):
    m = re.search(r'SCORE:\s*(\d+(?:\.\d+)?)/100', text)
    return float(m.group(1)) if m else None

def score_color(score):
    if score is None: return "#888"
    if score >= 75: return "#22c55e"
    if score >= 50: return "#f59e0b"
    return "#ef4444"

def score_label(score):
    if score is None: return "—"
    if score >= 75: return "Strong"
    if score >= 50: return "Fair"
    return "Weak"

# ── Groq call ─────────────────────────────────────────────────

def run_ats(resume_text, jd_text):
    client = Groq(api_key=API_KEY)
    prompt = f"RESUME:\n{trim(resume_text, 2000)}\n\nJOB DESCRIPTION:\n{trim(jd_text, 1200)}"
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": ATS_SYSTEM},
            {"role": "user", "content": prompt}
        ],
        max_tokens=600,
        temperature=0,
        seed=42,
    )
    return response.choices[0].message.content

# ════════════════════════════════════════════════════════════
# PAGE CONFIG & STYLES
# ════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ATS Checker",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Space+Grotesk:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

.stApp {
    background: #0a0a0f;
    color: #e8e6e0;
}

.top-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.2rem 2rem;
    border-bottom: 1px solid #1e1e2e;
    margin-bottom: 2rem;
}

.logo {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #e8e6e0;
    letter-spacing: 0.05em;
}

.logo span { color: #7c6fff; }

.nav-pill {
    background: #13131f;
    border: 1px solid #1e1e2e;
    border-radius: 20px;
    padding: 0.35rem 0.9rem;
    font-size: 0.75rem;
    color: #6b6b8a;
}

.hero {
    text-align: center;
    padding: 2.5rem 1rem 3rem;
    max-width: 640px;
    margin: 0 auto;
}

.hero-eyebrow {
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.15em;
    color: #7c6fff;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.6rem;
    font-weight: 600;
    line-height: 1.15;
    color: #f0eee8;
    margin-bottom: 0.75rem;
}

.hero-title em {
    font-style: normal;
    color: #7c6fff;
}

.hero-sub {
    font-size: 0.95rem;
    color: #5a5a78;
    line-height: 1.6;
}

.upload-label {
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    color: #5a5a78;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

.chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 0.2rem 0.65rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 500;
}

.chip-success {
    background: rgba(34,197,94,0.1);
    color: #22c55e;
    border: 1px solid rgba(34,197,94,0.2);
}

.chip-neutral {
    background: rgba(124,111,255,0.08);
    color: #7c6fff;
    border: 1px solid rgba(124,111,255,0.2);
}

.score-hero {
    background: #13131f;
    border: 1px solid #1e1e2e;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    margin-bottom: 1rem;
}

.score-bar-track {
    background: #1e1e2e;
    border-radius: 4px;
    height: 4px;
    margin: 0.5rem 0;
    overflow: hidden;
}

.breakdown-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.4rem 0;
    border-bottom: 1px solid #1a1a28;
    font-size: 0.82rem;
}

.breakdown-row:last-child { border-bottom: none; }
.breakdown-key { color: #6b6b8a; }
.breakdown-val { color: #e8e6e0; font-weight: 500; }

.tag {
    display: inline-block;
    padding: 0.2rem 0.55rem;
    border-radius: 6px;
    font-size: 0.7rem;
    font-weight: 500;
    margin: 2px;
}

.tag-match {
    background: rgba(34,197,94,0.08);
    color: #22c55e;
    border: 1px solid rgba(34,197,94,0.15);
}

.tag-miss {
    background: rgba(239,68,68,0.08);
    color: #ef4444;
    border: 1px solid rgba(239,68,68,0.15);
}

.section-label {
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    color: #3d3d5c;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #1a1a28;
}

div[data-testid="stFileUploader"] {
    background: #0f0f1a !important;
    border: 1px dashed #2a2a40 !important;
    border-radius: 10px !important;
}

div[data-testid="stFileUploader"]:hover {
    border-color: #7c6fff !important;
}

div[data-testid="stFileUploader"] label {
    color: #5a5a78 !important;
}

div[data-testid="stButton"] > button {
    background: #7c6fff !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    padding: 0.7rem 2rem !important;
    width: 100% !important;
    letter-spacing: 0.02em !important;
    transition: opacity 0.15s !important;
}

div[data-testid="stButton"] > button:hover { opacity: 0.88 !important; }

div[data-testid="stButton"] > button:disabled {
    background: #2a2a40 !important;
    color: #3d3d5c !important;
}

div[data-testid="stExpander"] {
    background: #13131f !important;
    border: 1px solid #1e1e2e !important;
    border-radius: 10px !important;
}

details summary { color: #b0aec8 !important; }

.stSpinner > div { border-top-color: #7c6fff !important; }

hr { border-color: #1a1a28 !important; }

section[data-testid="stSidebar"] {
    background: #0a0a0f;
    border-right: 1px solid #1e1e2e;
}

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: #2a2a40; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Top bar ───────────────────────────────────────────────────
st.markdown("""
<div class="top-bar">
    <div class="logo">◈ ATS<span>checker</span></div>
    <div class="nav-pill">Resume Intelligence</div>
</div>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Powered by Groq · LLaMA 3.3</div>
    <div class="hero-title">Screen resumes<br>at <em>scale</em></div>
    <div class="hero-sub">Upload a job description and any number of candidate resumes.<br>Get ATS scores in seconds.</div>
</div>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = []

# ── Upload section ────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="medium")

with col_left:
    st.markdown('<div class="upload-label">Job description</div>', unsafe_allow_html=True)
    jd_file = st.file_uploader(
        "Upload JD",
        type=["pdf", "docx", "doc", "txt"],
        key="jd",
        label_visibility="collapsed"
    )
    if jd_file:
        jd_file.seek(0)
        jd_text = read_file(jd_file, jd_file.name)
        st.markdown(f'<span class="chip chip-success">✓ {jd_file.name}</span>', unsafe_allow_html=True)
        st.caption(f"{len(jd_text):,} chars · {len(jd_text.split()):,} words")
    else:
        jd_text = ""

with col_right:
    st.markdown('<div class="upload-label">Candidate resumes</div>', unsafe_allow_html=True)
    resume_files = st.file_uploader(
        "Upload Resumes",
        type=["pdf", "docx", "doc"],
        accept_multiple_files=True,
        key="resumes",
        label_visibility="collapsed"
    )
    if resume_files:
        st.markdown(
            f'<span class="chip chip-neutral">◈ {len(resume_files)} resume{"s" if len(resume_files) > 1 else ""} ready</span>',
            unsafe_allow_html=True
        )

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# ── Analyze button ────────────────────────────────────────────
can_run = bool(jd_text and resume_files)

if st.button(
    "Run ATS Analysis" if can_run else "Upload files to begin",
    disabled=not can_run,
    use_container_width=True
):
    st.session_state.results = []
    progress = st.progress(0, text="Analyzing resumes…")

    for i, rf in enumerate(resume_files):
        rf.seek(0)
        resume_text = read_file(rf, rf.name)
        with st.spinner(f"Scoring {rf.name}…"):
            try:
                result = run_ats(resume_text, jd_text)
                score = parse_score(result)
            except Exception as e:
                result = f"Error: {e}"
                score = None

        st.session_state.results.append({
            "name": rf.name,
            "result": result,
            "score": score,
        })
        progress.progress((i + 1) / len(resume_files), text=f"Analyzed {i+1}/{len(resume_files)}")

    progress.empty()
    st.session_state.results.sort(key=lambda x: x["score"] or 0, reverse=True)

# ── Results ───────────────────────────────────────────────────
if st.session_state.results:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='section-label'>Results · {len(st.session_state.results)} candidates ranked</div>",
        unsafe_allow_html=True
    )

    for idx, r in enumerate(st.session_state.results):
        score = r["score"]
        color = score_color(score)
        label = score_label(score)
        score_display = f"{score:.1f}" if score is not None else "—"
        pct = score or 0
        raw = r["result"]

        def extract(pattern, text, default=""):
            m = re.search(pattern, text, re.DOTALL)
            return m.group(1).strip() if m else default

        breakdown_raw = extract(r'BREAKDOWN:(.*?)(?=MATCHED:|$)', raw)
        matched_raw   = extract(r'MATCHED:(.*?)(?=MISSING:|$)', raw)
        missing_raw   = extract(r'MISSING:(.*?)(?=STRENGTHS:|$)', raw)
        strengths_raw = extract(r'STRENGTHS:(.*?)(?=FIXES:|$)', raw)
        fixes_raw     = extract(r'FIXES:(.*?)$', raw)

        clean_name = r['name'].replace('.pdf','').replace('.docx','').replace('.doc','')
        medal = "🥇" if idx == 0 else ("🥈" if idx == 1 else ("🥉" if idx == 2 else "◈"))

        with st.expander(
            f"{medal}  {clean_name}  ·  {score_display}/100  ·  {label}",
            expanded=(idx == 0)
        ):
            top_col1, top_col2, top_col3 = st.columns([1, 2, 2])

            with top_col1:
                st.markdown(f"""
<div class="score-hero">
    <div style="font-family:'Space Grotesk',sans-serif;font-size:2.6rem;font-weight:600;color:{color};line-height:1">{score_display}</div>
    <div style="font-size:0.7rem;color:#3d3d5c;margin-top:4px;">/ 100</div>
    <div style="font-size:0.72rem;font-weight:500;color:{color};margin-top:8px;letter-spacing:0.05em;">{label.upper()}</div>
    <div class="score-bar-track" style="margin-top:12px">
        <div style="background:{color};width:{pct}%;height:100%;border-radius:4px"></div>
    </div>
</div>""", unsafe_allow_html=True)

            with top_col2:
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

            with top_col3:
                st.markdown('<div class="section-label">Strengths</div>', unsafe_allow_html=True)
                if strengths_raw:
                    for line in strengths_raw.strip().split('\n'):
                        line = line.strip().lstrip('- ')
                        if line:
                            st.markdown(
                                f"<div style='font-size:0.82rem;color:#b0aec8;padding:0.3rem 0;border-bottom:1px solid #1a1a28'>→ {line}</div>",
                                unsafe_allow_html=True
                            )

            kw_col1, kw_col2 = st.columns(2)
            with kw_col1:
                st.markdown('<div class="section-label" style="margin-top:1rem">Matched keywords</div>', unsafe_allow_html=True)
                if matched_raw:
                    tags = [t.strip() for t in matched_raw.split(',') if t.strip()]
                    st.markdown("".join(f'<span class="tag tag-match">{t}</span>' for t in tags[:20]), unsafe_allow_html=True)

            with kw_col2:
                st.markdown('<div class="section-label" style="margin-top:1rem">Missing keywords</div>', unsafe_allow_html=True)
                if missing_raw:
                    tags = [t.strip() for t in missing_raw.split(',') if t.strip()]
                    st.markdown("".join(f'<span class="tag tag-miss">{t}</span>' for t in tags[:20]), unsafe_allow_html=True)

            if fixes_raw:
                st.markdown('<div class="section-label" style="margin-top:1rem">Top improvements</div>', unsafe_allow_html=True)
                fix_lines = [l.strip() for l in fixes_raw.strip().split('\n') if l.strip()]
                for line in fix_lines[:3]:
                    line = re.sub(r'^\d+[\.\)]\s*', '', line)
                    st.markdown(f"""
<div style="display:flex;gap:12px;padding:0.6rem 0;border-bottom:1px solid #1a1a28;align-items:flex-start">
    <span style="color:#7c6fff;font-size:0.8rem;font-weight:600;min-width:16px;margin-top:1px">›</span>
    <span style="font-size:0.83rem;color:#b0aec8;line-height:1.5">{line}</span>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
    if st.button("Clear results", key="clear"):
        st.session_state.results = []
        st.rerun()
