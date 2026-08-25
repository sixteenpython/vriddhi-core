"""Visual system for the desktop-first BTI showcase."""

CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
:root { --ink:#12231c; --muted:#637068; --paper:#f7f4ed; --card:#fffef9; --green:#0d7651; --lime:#c9f277; --gold:#d69e2e; --red:#b94b46; --line:#dfe5df; }
.stApp { background:linear-gradient(180deg,#f4f0e7 0,#fbfaf6 18rem,#f7f8f5 100%); color:var(--ink); }
[data-testid="stHeader"] { background:transparent; }
.block-container { max-width:1440px; padding:1.6rem 2.6rem 4rem; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
h1,h2,h3 { color:var(--ink); letter-spacing:-.025em; }
.bti-kicker { color:var(--green); font-weight:700; letter-spacing:.16em; font-size:.72rem; text-transform:uppercase; }
.bti-hero { font-family:'Playfair Display',serif; font-size:clamp(3rem,6vw,6.6rem); line-height:.92; letter-spacing:-.055em; margin:.45rem 0 1rem; max-width:1050px; }
.bti-hero span { color:var(--green); }
.bti-lede { color:#46564d; font-size:1.22rem; line-height:1.65; max-width:780px; }
.bti-strip { display:flex; gap:.65rem; flex-wrap:wrap; margin:1.4rem 0; }
.bti-pill { border:1px solid #cfd8d1; background:rgba(255,255,255,.72); border-radius:999px; padding:.45rem .75rem; font-size:.82rem; color:#35463d; }
.bti-panel { background:rgba(255,254,249,.92); border:1px solid var(--line); border-radius:22px; padding:1.2rem 1.35rem; box-shadow:0 12px 38px rgba(20,43,32,.055); margin:.55rem 0 1rem; }
.bti-dark { color:white; background:linear-gradient(135deg,#10291f,#174d3a); border-radius:24px; padding:1.5rem; box-shadow:0 16px 44px rgba(13,58,41,.22); }
.bti-dark h2,.bti-dark h3 { color:white; }
.bti-metric-label { color:#77827c; font-size:.72rem; text-transform:uppercase; letter-spacing:.08em; }
.bti-metric-value { font-family:'Playfair Display',serif; font-size:1.85rem; color:var(--ink); }
.bti-score { font-family:'Playfair Display',serif; font-size:clamp(4rem,8vw,7rem); line-height:.9; }
.bti-class { color:var(--lime); font-weight:700; letter-spacing:.12em; }
.bti-up { color:#087f5b; font-weight:700; } .bti-down { color:var(--red); font-weight:700; }
.bti-note { color:#65736b; font-size:.88rem; line-height:1.55; }
.bti-step { display:inline-grid; place-items:center; width:1.8rem; height:1.8rem; border-radius:50%; background:#e8f1eb; color:var(--green); font-weight:700; margin-right:.5rem; }
div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:16px; overflow:hidden; }
div[data-testid="stMetric"] { background:rgba(255,254,249,.82); border:1px solid var(--line); border-radius:16px; padding:.8rem 1rem; }
.stButton>button { border-radius:12px; min-height:2.8rem; font-weight:700; border-color:#bfd0c5; }
.stButton>button[kind="primary"] { background:var(--green); color:white; border-color:var(--green); }
div[data-baseweb="select"]>div, .stNumberInput input, .stTextInput input { border-radius:12px!important; }
@media(max-width:800px){.block-container{padding:1rem}.bti-hero{font-size:3.2rem}}
</style>
"""
