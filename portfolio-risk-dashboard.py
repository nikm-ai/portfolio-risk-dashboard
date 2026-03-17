import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
# plotly.figure_factory imported but not used — removed
import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import minimize
import math
import io, warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Equity Portfolio Risk Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════
# CSS  — refined editorial with sharp data-ink aesthetics
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;0,600;1,400&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

  .block-container { padding-top: 3.5rem; padding-bottom: 5rem; max-width: 1200px; }
  [data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none; }

  /* ── Typography system ── */
  .paper-title {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 34px; font-weight: 500; line-height: 1.2;
    color: var(--text-color); margin-bottom: 0.35rem; letter-spacing: -0.02em;
  }
  .paper-byline {
    font-family: 'DM Sans', sans-serif;
    font-size: 12.5px; color: var(--text-color); opacity: 0.45;
    margin-bottom: 1.5rem; letter-spacing: 0.02em;
  }
  .abstract-box {
    border-top: 1px solid rgba(128,128,128,0.2);
    border-bottom: 1px solid rgba(128,128,128,0.2);
    padding: 1.25rem 0; margin-bottom: 2rem;
  }
  .abstract-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 9.5px; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; opacity: 0.38; color: var(--text-color); margin-bottom: 8px;
  }
  .abstract-text {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 15.5px; line-height: 1.85; color: var(--text-color); max-width: 920px;
  }
  .sec-header {
    font-family: 'DM Sans', sans-serif;
    font-size: 9.5px; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--text-color); opacity: 0.38;
    margin: 2.75rem 0 0.85rem; padding-bottom: 6px;
    border-bottom: 1px solid rgba(128,128,128,0.13);
  }
  .insight-box {
    border-left: 2px solid #1a4f82;
    padding: 0.6rem 1rem; margin: 0.75rem 0 1.25rem;
    background: rgba(26,79,130,0.04);
  }
  .insight-text {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 14.5px; line-height: 1.75; color: var(--text-color); opacity: 0.88;
  }
  /* ── KPI cards ── */
  .kpi-card {
    border: 0.75px solid rgba(128,128,128,0.18);
    border-radius: 2px; padding: 0.9rem 1rem;
    background: rgba(128,128,128,0.025); margin-bottom: 0.5rem;
  }
  .kpi-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 9px; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; opacity: 0.4; color: var(--text-color); margin-bottom: 5px;
  }
  .kpi-value {
    font-family: 'DM Sans', sans-serif;
    font-size: 24px; font-weight: 500; line-height: 1.1; color: var(--text-color);
  }
  .kpi-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 11.5px; margin-top: 3px; opacity: 0.5; color: var(--text-color);
  }
  .kpi-delta {
    font-family: 'DM Mono', monospace;
    font-size: 10.5px; margin-top: 4px; letter-spacing: 0.02em;
  }
  /* ── Highlight card for new sections ── */
  .kpi-card-highlight {
    border: 0.75px solid rgba(26,79,130,0.35);
    border-radius: 2px; padding: 0.9rem 1rem;
    background: rgba(26,79,130,0.04); margin-bottom: 0.5rem;
  }
  /* ── Stress card variants ── */
  .stress-card {
    border: 0.75px solid rgba(185,64,64,0.25);
    border-radius: 2px; padding: 0.85rem 1rem; margin-bottom: 0.5rem;
    background: rgba(185,64,64,0.02);
  }
  .stress-card-mild {
    border: 0.75px solid rgba(196,122,0,0.25);
    border-radius: 2px; padding: 0.85rem 1rem; margin-bottom: 0.5rem;
    background: rgba(196,122,0,0.02);
  }
  /* ── Regime badge ── */
  .regime-badge {
    display: inline-block;
    font-family: 'DM Sans', sans-serif;
    font-size: 10px; font-weight: 600; letter-spacing: 0.09em;
    text-transform: uppercase; padding: 2px 9px; border-radius: 2px; margin-right: 6px;
  }
  .regime-low  { background: rgba(46,125,79,0.12); color: #2e7d4f; }
  .regime-high { background: rgba(185,64,64,0.12); color: #b94040; }
  /* ── Colour helpers ── */
  .pos  { color: #2e7d4f !important; }
  .neg  { color: #b94040 !important; }
  .neut { color: #1a4f82 !important; }
  .warn { color: #c47a00 !important; }
  .mono { font-family: 'DM Mono', monospace !important; font-size: 13px !important; }
  /* ── Captions ── */
  .fig-caption {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 13.5px; line-height: 1.8; color: var(--text-color);
    opacity: 0.78; margin-top: 0.1rem; margin-bottom: 1.5rem; font-style: italic;
  }
  .fig-caption b { font-style: normal; font-weight: 600; color: var(--text-color); }
  .explainer-body {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 15px; line-height: 1.85; color: var(--text-color); opacity: 0.84;
    margin-bottom: 0.75rem;
  }
  .appendix-group { margin-bottom: 2rem; }
  .appendix-group-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 9.5px; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--text-color); opacity: 0.45;
    margin-bottom: 0.6rem; padding-bottom: 4px;
    border-bottom: 1px solid rgba(128,128,128,0.1);
  }
  .appendix-term {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 15px; line-height: 1.88; color: var(--text-color); margin-bottom: 0.65rem;
  }
  .appendix-term b { font-weight: 600; font-style: normal; }
  .paper-footer {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 12px; color: var(--text-color); opacity: 0.32;
    margin-top: 4rem; padding-top: 1rem;
    border-top: 1px solid rgba(128,128,128,0.12); line-height: 1.75;
  }
  label, .stSlider label, .stNumberInput label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 11px !important; font-weight: 500 !important;
    letter-spacing: 0.02em; opacity: 0.65;
  }
  /* ── New-section badge ── */
  .new-badge {
    display: inline-block;
    font-family: 'DM Sans', sans-serif;
    font-size: 9px; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; padding: 1px 7px;
    background: rgba(26,79,130,0.12); color: #1a4f82;
    border-radius: 2px; margin-left: 8px; vertical-align: middle;
  }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# CONSTANTS & HELPERS
# ══════════════════════════════════════════════════════════════════════════
CHART_BG = "#F5F0E8"
CREAM    = "#F5F0E8"
FONT_CH  = dict(size=12, color="#1a1a1a", family="DM Sans, Arial, sans-serif")
LEGEND   = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                font=dict(size=11, color="#1a1a1a"), bgcolor="rgba(0,0,0,0)")
BASE     = dict(plot_bgcolor=CHART_BG, paper_bgcolor=CHART_BG, font=FONT_CH,
                margin=dict(l=8, r=8, t=20, b=8), legend=LEGEND)

def layout(**overrides):
    """Return a copy of BASE with overrides applied. Py3.8-safe alternative to {**BASE, k:v}."""
    d = dict(BASE)
    d.update(overrides)
    return d

BLUE   = "#1a4f82"; LBLUE  = "#3d7ab5"; LLBLUE = "#b8cfe0"
GREEN  = "#2e7d4f"; LGREEN = "#6ab06a"; LLGREEN = "#c8e6c9"
RED    = "#b94040"; LRED   = "#c47a7a"; LLRED  = "#f5c6c6"
GOLD   = "#c47a00"; LGOLD  = "#e8a020"
GRAY   = "#888888"; LGRAY  = "#cccccc"
PURPLE = "#5c3d82"; LPURPLE = "#9b72c8"

RF_ANNUAL = 0.02
RF_DAILY  = RF_ANNUAL / 252

def ax(title="", grid=True, pct=False, suffix=""):
    """Return a complete Plotly axis dict. Safe to use with dict(**ax(...))
    as long as you don't re-specify any of its keys in the same dict() call.
    Keys returned: title, tickfont, gridcolor, linecolor, linewidth, showline,
    showgrid, zeroline, ticks, ticklen. Optional: tickformat, ticksuffix."""
    d = dict(
        title=dict(text=title, font=dict(size=11, color="#555555")),
        tickfont=dict(size=11, color="#444444"),
        gridcolor="#e8e0d0" if grid else "rgba(0,0,0,0)",
        linecolor="#d4c9b8", linewidth=1, showline=True,
        showgrid=grid, zeroline=False, ticks="outside", ticklen=3,
    )
    if pct:
        d["tickformat"] = ".1%"
    if suffix:
        d["ticksuffix"] = suffix
    return d

def ax_bare(title=""):
    """Return ONLY title+tickfont. Use when you need to add other axis keys
    manually without risk of duplicate-key conflicts."""
    return dict(
        title=dict(text=title, font=dict(size=11, color="#555555")),
        tickfont=dict(size=11, color="#444444"),
    )

AXIS_STYLE = dict(
    gridcolor="#e8e0d0", linecolor="#d4c9b8", linewidth=1,
    showline=True, showgrid=True, zeroline=False, ticks="outside", ticklen=3,
)
AXIS_STYLE_NOGRID = dict(
    gridcolor="rgba(0,0,0,0)", linecolor="#d4c9b8", linewidth=1,
    showline=True, showgrid=False, zeroline=False, ticks="outside", ticklen=3,
)

def fmt_pct(v, d=2): return f"{v:.{d}%}"
def fmt_f(v, d=2):   return f"{v:.{d}f}"
def sgn(v):          return f"+{v:.2%}" if v >= 0 else f"{v:.2%}"
def sgn_f(v, d=2):   return f"+{v:.{d}f}" if v >= 0 else f"{v:.{d}f}"

def kde_curve(data, n_pts=300, bw_factor=1.0):
    data = np.asarray(data)
    n, std = len(data), np.std(data)
    if std == 0 or n < 2:
        return np.array([]), np.array([])
    bw = bw_factor * 1.06 * std * n**(-0.2)
    x  = np.linspace(data.min() - 3*bw, data.max() + 3*bw, n_pts)
    z  = (x[:, None] - data[None, :]) / bw
    density = np.exp(-0.5*z**2).sum(axis=1) / (n*bw*np.sqrt(2*np.pi))
    return x, density

def ols(y, X):
    b  = np.linalg.lstsq(X, y, rcond=None)[0]
    yh = X @ b
    ss_res = ((y - yh)**2).sum()
    ss_tot = ((y - y.mean())**2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return b, r2

# ══════════════════════════════════════════════════════════════════════════
# TICKERS
# ══════════════════════════════════════════════════════════════════════════
TICKERS = [
    "NVDA","MSFT","AAPL","AMZN","GOOGL","META","AVGO","TSLA","JPM","WMT",
    "V","LLY","ORCL","NFLX","MA","XOM","COST","PG","JNJ","HD"
]
BENCHMARK = "^GSPC"
NAMES = {
    "NVDA":"NVIDIA","MSFT":"Microsoft","AAPL":"Apple","AMZN":"Amazon",
    "GOOGL":"Alphabet","META":"Meta Platforms","AVGO":"Broadcom","TSLA":"Tesla",
    "JPM":"JPMorgan Chase","WMT":"Walmart","V":"Visa","LLY":"Eli Lilly",
    "ORCL":"Oracle","NFLX":"Netflix","MA":"Mastercard","XOM":"ExxonMobil",
    "COST":"Costco","PG":"Procter & Gamble","JNJ":"Johnson & Johnson","HD":"Home Depot"
}

FF3_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)

# Historical stress scenario date ranges (approximate)
STRESS_SCENARIOS = {
    "COVID Crash (Feb–Mar 2020)": ("2020-02-19", "2020-03-23"),
    "2022 Rate Shock (Jan–Oct 2022)": ("2022-01-03", "2022-10-14"),
    "GFC Lehman (Sep–Nov 2008)": ("2008-09-15", "2008-11-20"),
    "Dot-com Peak–Trough (Mar 2000–Oct 2002)": ("2000-03-24", "2002-10-09"),
    "2018 Q4 Selloff": ("2018-09-20", "2018-12-24"),
    "Aug 2015 Flash Crash": ("2015-08-10", "2015-08-25"),
}

# ══════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="paper-title">Equity Portfolio Risk Dashboard</div>
<div class="paper-byline">
  Risk attribution · factor exposure · return distribution · drawdown analysis ·
  efficient frontier · stress testing · regime detection — 20-stock US equity portfolio
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="abstract-box">
  <div class="abstract-label">Overview</div>
  <div class="abstract-text">
    This dashboard computes and attributes risk across a 20-stock US large-cap equity portfolio,
    benchmarked against the S&P 500. Core metrics include annualized return, volatility, Sharpe,
    Sortino, Calmar, Value at Risk and Expected Shortfall under both historical and parametric
    assumptions, market beta, and CAPM alpha. Rolling 60-day windows track how the portfolio's
    risk profile shifts over time. Section 11 performs Fama-French three-factor attribution.
    Section 12 constructs the mean-variance efficient frontier and derives the minimum-variance
    and maximum-Sharpe optimal portfolios relative to the equal-weight baseline. Section 13
    stress-tests the portfolio against six named historical market dislocations. Section 14
    detects high- and low-volatility regimes via a rolling HMM-inspired threshold approach.
    A plain-language methodology appendix defines every metric.
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">Data</div>', unsafe_allow_html=True)

end_date   = datetime.today()
start_date = end_date - timedelta(days=365)
# Extended window for stress scenario simulation
stress_start = datetime(2000, 1, 1)

@st.cache_data(show_spinner=False)
def load_prices(tickers, start, end):
    raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
    return raw['Close'].dropna()

@st.cache_data(show_spinner=False)
def load_prices_extended(tickers, start, end):
    """Extended history for stress test beta computation."""
    raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
    return raw['Close'].dropna()

@st.cache_data(show_spinner=False)
def load_ff3(start, end):
    try:
        ff = pd.read_csv(
            FF3_URL, compression="zip", skiprows=3,
            names=["Date","Mkt-RF","SMB","HML","RF"],
            index_col=0, parse_dates=True,
        )
        ff = ff[ff.index.astype(str).str.len() == 8]
        ff.index = pd.to_datetime(ff.index, format="%Y%m%d")
        ff = ff.apply(pd.to_numeric, errors="coerce").dropna() / 100
        return ff.loc[start:end]
    except Exception:
        return None

dl_col1, dl_col2 = st.columns([2, 1])
with dl_col1:
    st.markdown("""<div class="explainer-body">
      By default the dashboard pulls one year of adjusted daily closing prices from Yahoo Finance
      for the 20 stocks listed below, plus the S&P 500 as benchmark. Fama-French factor data
      is fetched from Kenneth French's public data library. For stress testing, historical
      S&P 500 prices since 2000 are fetched separately. To use your own return data, upload
      a CSV with dates as the index and one column per asset containing daily returns (not prices).
    </div>""", unsafe_allow_html=True)
with dl_col2:
    uploaded_file = st.file_uploader("Upload returns CSV (optional)", type=["csv"])

with st.spinner("Fetching price data…"):
    all_prices = load_prices(TICKERS + [BENCHMARK], start_date, end_date)

with st.spinner("Fetching Fama-French factors…"):
    ff3 = load_ff3(start_date, end_date)

with st.spinner("Fetching extended S&P 500 history for stress tests…"):
    spx_ext = load_prices_extended([BENCHMARK], stress_start, end_date)

all_returns   = all_prices.pct_change().dropna()
bench_returns = all_returns[BENCHMARK].copy()
asset_returns = all_returns.drop(columns=[BENCHMARK]).copy()

if uploaded_file:
    custom = pd.read_csv(uploaded_file, index_col=0, parse_dates=True).dropna()
    df = custom.copy()
    st.success("Custom dataset loaded.")
else:
    df = asset_returns.copy()

display_names = {t: NAMES.get(t, t) for t in df.columns}

buf = io.StringIO()
df.to_csv(buf)
st.download_button("Download return data (CSV)", data=buf.getvalue(),
                   file_name="portfolio_returns.csv", mime="text/csv")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 1 — RETURN DATA PREVIEW
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">1. Return data preview</div>', unsafe_allow_html=True)
st.dataframe(df.sort_index(ascending=False).head(5).rename(columns=display_names),
             use_container_width=True)
st.markdown("""<div class="fig-caption">
  <b>Table 1.</b> Five most recent trading days of daily returns.
  Returns are simple (not log-transformed) and used directly in all downstream calculations.
</div>""", unsafe_allow_html=True)
st.latex(r"r_t = \frac{P_t - P_{t-1}}{P_{t-1}}")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 2 — PORTFOLIO WEIGHTS
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">2. Portfolio weights</div>', unsafe_allow_html=True)
st.markdown("""<div class="explainer-body" style="margin-bottom:1rem;">
  Edit the Weight column directly. Weights are normalized to sum to 100% before any calculation.
  The default is equal-weight at 5% per stock. The efficient frontier (Section 12) will show
  how this compares to analytically optimal allocations.
</div>""", unsafe_allow_html=True)

numeric_df = df.select_dtypes(include=np.number)
n_assets   = numeric_df.shape[1]
tickers_in = list(numeric_df.columns)

weight_init = pd.DataFrame({
    "Ticker":  tickers_in,
    "Company": [NAMES.get(t, t) for t in tickers_in],
    "Weight":  [5.0] * n_assets,
})

wt_col, bar_col = st.columns([1, 1])
with wt_col:
    edited = st.data_editor(
        weight_init, use_container_width=True, hide_index=True,
        disabled=["Ticker", "Company"],
        column_config={
            "Ticker":  st.column_config.TextColumn("Ticker", width="small"),
            "Company": st.column_config.TextColumn("Company", width="medium"),
            "Weight":  st.column_config.NumberColumn(
                "Weight", min_value=0.0, max_value=100.0,
                step=0.5, format="%.1f", width="small"),
        },
        key="weight_editor",
    )

raw_weights_arr = edited["Weight"].values.astype(float)
if raw_weights_arr.sum() == 0:
    st.error("All weights are zero.")
    st.stop()
weights = raw_weights_arr / raw_weights_arr.sum()
is_equal_weight = np.std(weights) < 1e-6

with bar_col:
    sorted_idx     = np.argsort(weights)[::-1]
    sorted_names   = [NAMES.get(tickers_in[i], tickers_in[i]) for i in sorted_idx]
    sorted_weights = np.round(weights[sorted_idx] * 100, 1)
    n = len(sorted_names)
    fig_alloc = go.Figure(go.Bar(
        x=sorted_weights, y=sorted_names, orientation="h",
        marker_color=LBLUE, opacity=0.8,
        text=[f"{w:.1f}%" for w in sorted_weights],
        textposition="outside", textfont=dict(size=9, color="#333"),
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    equal_w = round(100/n_assets, 1)
    fig_alloc.add_vline(x=equal_w, line_dash="dot", line_color=GRAY, line_width=1,
                        annotation_text=f"Equal weight ({equal_w:.1f}%)",
                        annotation_font=dict(size=9, color=GRAY), annotation_position="top right")
    fig_alloc.update_layout(
        **layout(margin=dict(l=8, r=55, t=10, b=8), height=max(320, n_assets*22)),
        xaxis=dict(**ax("Normalized weight (%)"), ticksuffix="%"),
        yaxis=dict(tickfont=dict(size=10, color="#444"), showgrid=False,
                   linecolor="#d4c9b8", linewidth=1, autorange="reversed"),
        showlegend=False,
    )
    st.plotly_chart(fig_alloc, use_container_width=True)

st.markdown("""<div class="fig-caption">
  <b>Portfolio allocation.</b> Normalized weights after adjustment. Dotted line marks
  the equal-weight threshold. Sections 12–13 benchmark these weights against optimal portfolios.
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# CORE CALCULATIONS
# ══════════════════════════════════════════════════════════════════════════
clean_df = numeric_df.dropna()
if clean_df.shape[0] < 30:
    st.error("Insufficient clean data (fewer than 30 trading days).")
    st.stop()

port_returns = clean_df.dot(weights)
cov_matrix   = clean_df.cov() * 252
port_vol     = np.sqrt(weights @ cov_matrix.values @ weights)

excess  = port_returns - RF_DAILY
sharpe  = (excess.mean() / excess.std()) * np.sqrt(252)

downside     = excess.copy(); downside[downside > 0] = 0
downside_std = np.sqrt((downside**2).mean()) * np.sqrt(252)
sortino      = (excess.mean() * 252) / downside_std if downside_std > 0 else np.nan

pr_clean     = port_returns.dropna().values
var_hist_95  = np.percentile(pr_clean, 5)
var_hist_99  = np.percentile(pr_clean, 1)
# Expected Shortfall (CVaR) — mean of returns beyond VaR
cvar_hist_95 = pr_clean[pr_clean <= var_hist_95].mean() if (pr_clean <= var_hist_95).any() else var_hist_95
cvar_hist_99 = pr_clean[pr_clean <= var_hist_99].mean() if (pr_clean <= var_hist_99).any() else var_hist_99

mu_daily     = port_returns.mean()
sig_daily    = port_returns.std()
z95, z99     = 1.6449, 2.3263
var_param_95 = mu_daily - z95*sig_daily
var_param_99 = mu_daily - z99*sig_daily
# Parametric CVaR
cvar_param_95 = mu_daily - sig_daily * (math.exp(-0.5*z95**2) / math.sqrt(2*math.pi)) / (1 - 0.95)
cvar_param_99 = mu_daily - sig_daily * (math.exp(-0.5*z99**2) / math.sqrt(2*math.pi)) / (1 - 0.99)

bench_aligned = bench_returns.reindex(port_returns.index).dropna()
port_aligned  = port_returns.reindex(bench_aligned.index)
cov_pm        = np.cov(port_aligned.values, bench_aligned.values)[0,1]
var_m         = np.var(bench_aligned.values)
beta          = cov_pm / var_m if var_m > 0 else np.nan
alpha_ann     = ((port_aligned.mean()-RF_DAILY) - beta*(bench_aligned.mean()-RF_DAILY)) * 252

bench_vol    = bench_aligned.std() * np.sqrt(252)
bench_sharpe = ((bench_aligned.mean()-RF_DAILY)/bench_aligned.std()) * np.sqrt(252)

roll_win    = 60
roll_vol    = port_returns.rolling(roll_win).std() * np.sqrt(252)
roll_sharpe = (port_returns.rolling(roll_win).mean() /
               port_returns.rolling(roll_win).std()) * np.sqrt(252)
roll_beta   = (port_aligned.rolling(roll_win).cov(bench_aligned) /
               bench_aligned.rolling(roll_win).var())

cum_port        = (1 + port_returns).cumprod()
cum_bench       = (1 + bench_aligned).cumprod()
total_ret       = cum_port.iloc[-1] - 1
bench_total_ret = cum_bench.iloc[-1] - 1

rolling_max  = cum_port.cummax()
drawdown     = (cum_port - rolling_max) / rolling_max
max_dd       = drawdown.min()
trough_date  = drawdown.idxmin()
pre_trough   = cum_port.loc[:trough_date]
peak_date    = pre_trough.idxmax()
post_dd      = drawdown.loc[trough_date:]
recovered    = post_dd[post_dd >= 0]
recovery_date = recovered.index[0] if len(recovered) > 0 else None
calmar        = (total_ret / abs(max_dd)) if max_dd != 0 else np.nan
port_ret_ann  = port_returns.mean() * 252

# ══════════════════════════════════════════════════════════════════════════
# PORTFOLIO SUMMARY
# ══════════════════════════════════════════════════════════════════════════
excess_ret   = total_ret - bench_total_ret
sharpe_diff  = sharpe - bench_sharpe
vol_diff     = port_vol - bench_vol

if is_equal_weight:
    weight_desc = f"At equal weight across {n_assets} positions"
else:
    top3_idx   = np.argsort(weights)[::-1][:3]
    top3_names = [NAMES.get(tickers_in[i], tickers_in[i]) for i in top3_idx]
    top3_wts   = [weights[i]*100 for i in top3_idx]
    weight_desc = (f"With a tilted allocation (largest positions: "
                   f"{top3_names[0]} {top3_wts[0]:.1f}%, "
                   f"{top3_names[1]} {top3_wts[1]:.1f}%, "
                   f"{top3_names[2]} {top3_wts[2]:.1f}%)")

if sharpe_diff > 0.1:
    sharpe_prose = (f"The Sharpe ratio of {sharpe:.2f} exceeds the S&P 500's "
                    f"{bench_sharpe:.2f} by {sharpe_diff:.2f}, indicating materially "
                    f"better risk-adjusted returns.")
elif sharpe_diff > 0:
    sharpe_prose = (f"The Sharpe ratio of {sharpe:.2f} modestly exceeds the S&P 500's {bench_sharpe:.2f}.")
else:
    sharpe_prose = (f"The Sharpe ratio of {sharpe:.2f} trails the S&P 500's {bench_sharpe:.2f}.")

beta_prose = ("captures more than proportional market moves" if beta > 1.1
              else "is more defensive than the index" if beta < 0.9
              else "tracks the market closely")

st.markdown(f"""
<div class="abstract-box" style="margin-top:2rem;">
  <div class="abstract-label">Portfolio summary</div>
  <div class="abstract-text">
    {weight_desc}, the portfolio returned {total_ret:.2%} over the past year against
    {bench_total_ret:.2%} for the S&P 500, a {sgn(excess_ret)} excess return.
    Annualized volatility of {port_vol:.2%} compares to {bench_vol:.2%} for the benchmark
    ({sgn(vol_diff)}). {sharpe_prose}
    The Sortino ratio of {sortino:.2f} penalizes only downside volatility, and the
    Calmar ratio of {calmar:.2f} expresses return earned per unit of maximum drawdown risk.
    Expected Shortfall at 95% confidence is {abs(cvar_hist_95):.2%} — the average loss
    on the worst 5% of days, a more complete tail risk picture than VaR alone.
    With a market beta of {beta:.2f}, the portfolio {beta_prose}.
    CAPM alpha of {sgn(alpha_ann)} annualized represents residual return beyond what market
    exposure alone predicts. The maximum drawdown of {max_dd:.2%} reached its trough in
    {trough_date.strftime('%B %Y')}.
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 3 — KPI CARDS
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">3. Portfolio metrics</div>', unsafe_allow_html=True)

kpi_rows = [
    [
        ("Annualized return",    fmt_pct(total_ret),
         f"S&P 500: {fmt_pct(bench_total_ret)}",
         sgn(excess_ret), "pos" if total_ret > bench_total_ret else "neg"),
        ("Annualized volatility", fmt_pct(port_vol),
         f"S&P 500: {fmt_pct(bench_vol)}",
         sgn(vol_diff) + " vs benchmark", "warn" if port_vol > bench_vol*1.1 else "neut"),
        ("Max drawdown",         fmt_pct(max_dd),
         f"Trough: {trough_date.strftime('%b %Y')}",
         f"Recovery: {recovery_date.strftime('%b %d, %Y') if recovery_date else 'Incomplete'}",
         "neg" if abs(max_dd) > 0.1 else "warn"),
    ],
    [
        ("Sharpe ratio",  fmt_f(sharpe),  f"S&P 500: {fmt_f(bench_sharpe)}",
         sgn_f(sharpe_diff) + " vs benchmark", "pos" if sharpe > bench_sharpe else "neut"),
        ("Sortino ratio", fmt_f(sortino), "Downside deviation only",
         "↑ upside-dominant volatility" if sortino > sharpe else "", "pos" if sortino > 1.0 else "neut"),
        ("Calmar ratio",  fmt_f(calmar),  "Return / |max drawdown|",
         "≥2 is strong" if calmar >= 2 else "≥1 is adequate",
         "pos" if calmar > 2.0 else "neut" if calmar > 1.0 else "warn"),
    ],
    [
        ("Historical VaR (95%)", fmt_pct(abs(var_hist_95)),
         f"99%: {fmt_pct(abs(var_hist_99))}",
         f"CVaR 95%: {fmt_pct(abs(cvar_hist_95))}", "neg" if abs(var_hist_95) > 0.02 else "neut"),
        ("Parametric VaR (95%)", fmt_pct(abs(var_param_95)),
         f"Gaussian, 99%: {fmt_pct(abs(var_param_99))}",
         f"CVaR 95%: {fmt_pct(abs(cvar_param_95))}", "neut"),
        ("CAPM alpha (ann.)", sgn(alpha_ann),
         f"Beta: {beta:.2f} vs. S&P 500",
         "statistically significant" if abs(alpha_ann/port_vol) > 0.5 else "borderline significance",
         "pos" if alpha_ann > 0 else "neg"),
    ],
]

for row in kpi_rows:
    cols = st.columns(3)
    for col, (label, value, sub, delta, cls) in zip(cols, row):
        with col:
            delta_html = f'<div class="kpi-delta {cls}">{delta}</div>' if delta else ""
            st.markdown(f"""<div class="kpi-card">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value {cls}">{value}</div>
              <div class="kpi-sub">{sub}</div>
              {delta_html}
            </div>""", unsafe_allow_html=True)

weight_note = "equal-weight" if is_equal_weight else "current-weight"
st.markdown(f"""<div class="fig-caption" style="margin-top:0.75rem;">
  <b>Table 2.</b> Summary risk and return metrics for the {weight_note} portfolio.
  CVaR (Expected Shortfall) is reported alongside VaR — it measures the average loss
  on the worst tail days, making it a more complete picture of downside risk.
  Beta estimated by OLS on daily returns; alpha is the annualized intercept net of the risk-free rate.
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 4 — CORRELATION MATRIX
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">4. Correlation matrix</div>', unsafe_allow_html=True)

corr = clean_df.corr()
colorscale = [
    [0.0, "#b94040"], [0.4, "#e8c8c8"], [0.5, CREAM],
    [0.65, "#b8cfe0"], [1.0, "#1a4f82"],
]
fig_corr = go.Figure(go.Heatmap(
    z=corr.values, x=list(corr.columns), y=list(corr.index),
    colorscale=colorscale, zmid=0, zmin=-1, zmax=1,
    colorbar=dict(title=dict(text="ρ", font=dict(size=12, color="#333")),
                  tickfont=dict(size=10, color="#444"), thickness=10, len=0.8),
    hovertemplate="%{y} / %{x}: %{z:.3f}<extra></extra>", showscale=True,
))
fig_corr.update_layout(
    **layout(margin=dict(l=8, r=60, t=20, b=8), height=420),
    xaxis=dict(tickfont=dict(size=10, color="#444"), showgrid=False,
               linecolor="#d4c9b8", linewidth=1),
    yaxis=dict(tickfont=dict(size=10, color="#444"), showgrid=False,
               linecolor="#d4c9b8", linewidth=1, autorange="reversed"),
)
st.plotly_chart(fig_corr, use_container_width=True)
st.markdown(f"""<div class="fig-caption">
  <b>Figure 1.</b> Pearson correlation matrix across all {n_assets} constituents.
  Blue = positive co-movement; red = near-zero or negative. Hover for exact values.
  High pairwise correlations are the primary constraint on diversification.
</div>""", unsafe_allow_html=True)

corr_pairs = corr.unstack().rename_axis(['T1','T2']).reset_index(name='Correlation')
corr_pairs = corr_pairs[corr_pairs['T1'] != corr_pairs['T2']]
corr_pairs['Pair'] = corr_pairs.apply(lambda r: tuple(sorted([r['T1'],r['T2']])), axis=1)
corr_pairs = corr_pairs.drop_duplicates('Pair').drop(columns='Pair')
corr_pairs = corr_pairs.reindex(corr_pairs['Correlation'].abs().sort_values(ascending=False).index)
corr_pairs['Company 1'] = corr_pairs['T1'].map(NAMES)
corr_pairs['Company 2'] = corr_pairs['T2'].map(NAMES)
top_pairs = corr_pairs[['Company 1','Company 2','Correlation']].head(15)

st.markdown('<div class="sec-header" style="margin-top:1rem;">Top 15 pairwise correlations</div>',
            unsafe_allow_html=True)
st.dataframe(top_pairs.style.format({"Correlation": "{:.3f}"}), use_container_width=True)
st.markdown("""<div class="fig-caption">
  <b>Table 3.</b> Fifteen highest pairwise return correlations. High within-sector correlations
  are the primary constraint on diversification; this is quantified in Section 12.
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 5 — RETURN DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">5. Return distribution</div>', unsafe_allow_html=True)

fig_dist = go.Figure()
fig_dist.add_trace(go.Histogram(
    x=port_returns.values, nbinsx=50, marker_color=LBLUE, opacity=0.55,
    name="Portfolio", hovertemplate="Return: %{x:.3%}<br>Count: %{y}<extra></extra>",
))
kde_x, kde_d = kde_curve(port_returns.values, bw_factor=0.9)
if len(kde_x):
    bin_w   = (port_returns.max()-port_returns.min())/50
    kde_cnt = kde_d * len(port_returns) * bin_w
    fig_dist.add_trace(go.Scatter(
        x=kde_x, y=kde_cnt, mode="lines",
        line=dict(color="#2a2a2a", width=2.5), name="KDE", hoverinfo="skip",
    ))
fig_dist.add_trace(go.Histogram(
    x=bench_aligned.values, nbinsx=50, marker_color=LRED, opacity=0.35,
    name="S&P 500", hovertemplate="S&P 500: %{x:.3%}<br>Count: %{y}<extra></extra>",
))
fig_dist.add_vline(x=var_hist_95, line_dash="dash", line_color=RED, line_width=1.5,
                   annotation_text=f"VaR 95% ({var_hist_95:.2%})",
                   annotation_font=dict(size=9, color=RED), annotation_position="top left")
fig_dist.add_vline(x=cvar_hist_95, line_dash="dash", line_color=PURPLE, line_width=1.5,
                   annotation_text=f"CVaR 95% ({cvar_hist_95:.2%})",
                   annotation_font=dict(size=9, color=PURPLE), annotation_position="bottom left")
fig_dist.add_vline(x=0, line_dash="dot", line_color=GRAY, line_width=1)
fig_dist.update_layout(
    **BASE, height=320, barmode="overlay",
    xaxis=dict(**ax("Daily return"), tickformat=".1%"),
    yaxis=dict(**ax("Count")),
)
st.plotly_chart(fig_dist, use_container_width=True)

skew_val  = float(port_returns.skew())
kurt_val  = float(port_returns.kurtosis())
skew_desc = ('slight left tail' if skew_val < -0.1
             else 'slight right tail' if skew_val > 0.1
             else 'approximately symmetric')
kurt_note = (
    f"Excess kurtosis of {kurt_val:.2f} is very high; over a one-year window this "
    f"is typically driven by one or two extreme return days rather than a structurally "
    f"fat-tailed distribution. Moment estimates should be interpreted cautiously."
    if kurt_val > 5 else
    f"Excess kurtosis of {kurt_val:.2f} indicates "
    f"{'mildly fat tails' if kurt_val > 0.5 else 'near-normal tail behavior'}."
)
st.markdown(f"""<div class="fig-caption">
  <b>Figure 2.</b> Daily return distribution for the portfolio (blue) and S&P 500 (red),
  with Gaussian KDE overlay. Daily mean {port_returns.mean():.3%},
  standard deviation {port_returns.std():.3%}. Skewness {skew_val:.3f} ({skew_desc}).
  {kurt_note} Dashed red = empirical 95% VaR; dashed purple = CVaR (Expected Shortfall at 95%),
  the average loss beyond the VaR threshold.
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 6 — CUMULATIVE RETURNS
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">6. Cumulative returns</div>', unsafe_allow_html=True)

fig_cum = go.Figure()
fig_cum.add_trace(go.Scatter(
    x=cum_port.index, y=cum_port.values, mode="lines", name="Portfolio",
    line=dict(color=BLUE, width=2.5),
    hovertemplate="Date: %{x|%Y-%m-%d}<br>Value: %{y:.3f}<extra></extra>",
))
fig_cum.add_trace(go.Scatter(
    x=cum_bench.index, y=cum_bench.values, mode="lines", name="S&P 500",
    line=dict(color=GRAY, width=1.8, dash="dash"),
    hovertemplate="Date: %{x|%Y-%m-%d}<br>S&P 500: %{y:.3f}<extra></extra>",
))
fig_cum.add_hline(y=1.0, line_dash="dot", line_color="#d4c9b8", line_width=1)
fig_cum.update_layout(
    **BASE, height=320,
    xaxis=dict(**ax("Date")),
    yaxis=dict(**ax("Growth of $1")),
)
st.plotly_chart(fig_cum, use_container_width=True)
st.markdown(f"""<div class="fig-caption">
  <b>Figure 3.</b> Growth of $1 invested in the portfolio vs. the S&P 500.
  Portfolio returned {total_ret:.2%} against {bench_total_ret:.2%}
  for the benchmark ({sgn(total_ret-bench_total_ret)} outperformance).
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 7 — DRAWDOWN ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">7. Drawdown analysis</div>', unsafe_allow_html=True)

fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(
    x=drawdown.index, y=drawdown.values*100, mode="lines", fill="tozeroy",
    fillcolor="rgba(185,64,64,0.15)", line=dict(color=RED, width=1.5),
    hovertemplate="Date: %{x|%Y-%m-%d}<br>Drawdown: %{y:.2f}%<extra></extra>",
))
fig_dd.add_hline(y=max_dd*100, line_dash="dot", line_color=RED, line_width=1,
                 annotation_text=f"Max DD {max_dd:.2%}",
                 annotation_font=dict(size=9, color=RED), annotation_position="bottom right")
fig_dd.update_layout(
    **BASE, height=280,
    xaxis=dict(**ax("Date")),
    yaxis=dict(**ax("Drawdown (%)"), ticksuffix="%"),
    showlegend=False,
)
st.plotly_chart(fig_dd, use_container_width=True)

dd1, dd2, dd3 = st.columns(3)
for col, label, val, sub, cls in [
    (dd1, "Maximum drawdown", f"{max_dd:.2%}", "Peak-to-trough decline", "neg"),
    (dd2, "Drawdown peak", peak_date.strftime('%b %d, %Y'), "Last high before trough", "neut"),
    (dd3, "Recovery date",
     recovery_date.strftime('%b %d, %Y') if recovery_date else "Not yet recovered",
     "First date drawdown reached 0", "pos" if recovery_date else "warn"),
]:
    with col:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value {cls}">{val}</div>
          <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

recovery_prose = (
    f"The portfolio recovered to a new high by {recovery_date.strftime('%B %d, %Y')}."
    if recovery_date else
    "The portfolio had not fully recovered to its prior peak within the analysis window."
)
st.markdown(f"""<div class="fig-caption">
  <b>Figure 4.</b> Drawdown as percentage decline from the rolling peak.
  The maximum drawdown of {max_dd:.2%} reached its trough on
  {trough_date.strftime('%B %d, %Y')}, following a peak on {peak_date.strftime('%B %d, %Y')}.
  {recovery_prose}
</div>""", unsafe_allow_html=True)
st.latex(r"D_t = \frac{V_t - \max_{s \leq t} V_s}{\max_{s \leq t} V_s}")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 8 — ROLLING RISK METRICS
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">8. Rolling risk metrics (60-day window)</div>',
            unsafe_allow_html=True)
st.markdown("""<div class="explainer-body">
  Rolling metrics reveal how the portfolio's risk profile has shifted over time.
  Rising rolling volatility during a drawdown confirms risk escalation; declining rolling
  Sharpe confirms deteriorating risk-adjusted returns. Rolling beta identifies when the
  portfolio became more or less sensitive to the market. Regime coloring in Section 14
  maps these transitions to distinct market states.
</div>""", unsafe_allow_html=True)

rc1, rc2 = st.columns(2)
with rc1:
    bench_roll_vol = bench_aligned.rolling(roll_win).std() * np.sqrt(252)
    fig_rv = go.Figure()
    fig_rv.add_trace(go.Scatter(x=roll_vol.index, y=roll_vol.values*100, mode="lines",
        name="Portfolio", line=dict(color=BLUE, width=2.2),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Vol: %{y:.2f}%<extra></extra>"))
    fig_rv.add_trace(go.Scatter(x=bench_roll_vol.index, y=bench_roll_vol.values*100,
        mode="lines", name="S&P 500", line=dict(color=GRAY, width=1.5, dash="dash"),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>S&P 500: %{y:.2f}%<extra></extra>"))
    fig_rv.update_layout(
        **layout(margin=dict(l=60, r=8, t=20, b=8), height=280,
                 xaxis=dict(**ax("Date")),
                 yaxis=dict(**ax("Annualized volatility (%)"), ticksuffix="%"))
    )
    
    st.plotly_chart(fig_rv, use_container_width=True)
    st.markdown("""<div class="fig-caption">
      <b>Figure 5.</b> 60-day rolling annualized volatility vs. S&P 500.
    </div>""", unsafe_allow_html=True)

with rc2:
    fig_rs = go.Figure()
    fig_rs.add_trace(go.Scatter(x=roll_sharpe.index, y=roll_sharpe.values, mode="lines",
        name="Rolling Sharpe", line=dict(color=GREEN, width=2.2),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Sharpe: %{y:.2f}<extra></extra>"))
    fig_rs.add_hline(y=0, line_dash="dot", line_color="#d4c9b8", line_width=1)
    fig_rs.add_hline(y=1.0, line_dash="dot", line_color=GRAY, line_width=1,
                     annotation_text="Sharpe = 1",
                     annotation_font=dict(size=9, color=GRAY))
    fig_rs.update_layout(
        **layout(margin=dict(l=60, r=8, t=20, b=8), height=280,
                 xaxis=dict(**ax("Date")),
                 yaxis=dict(**ax("Sharpe ratio (annualized)")),
                 showlegend=False),
    )
    st.plotly_chart(fig_rs, use_container_width=True)
    st.markdown("""<div class="fig-caption">
      <b>Figure 6.</b> 60-day rolling annualized Sharpe. Below zero = negative risk-adjusted returns.
    </div>""", unsafe_allow_html=True)

fig_rb = go.Figure()
fig_rb.add_trace(go.Scatter(x=roll_beta.index, y=roll_beta.values, mode="lines",
    name="Rolling beta", line=dict(color=LBLUE, width=2.2),
    hovertemplate="Date: %{x|%Y-%m-%d}<br>Beta: %{y:.3f}<extra></extra>"))
fig_rb.add_hline(y=1.0, line_dash="dash", line_color=GRAY, line_width=1.5,
                 annotation_text="Beta = 1 (market)",
                 annotation_font=dict(size=10, color=GRAY), annotation_position="top right")
fig_rb.add_hline(y=beta, line_dash="dot", line_color=BLUE, line_width=1.5,
                 annotation_text=f"Full-period beta ({beta:.2f})",
                 annotation_font=dict(size=10, color=BLUE), annotation_position="bottom right")
fig_rb.update_layout(
    **layout(margin=dict(l=70, r=20, t=20, b=8), height=280,
             xaxis=dict(**ax("Date")),
             yaxis=dict(**ax("Beta vs. S&P 500")),
             showlegend=False),
)
st.plotly_chart(fig_rb, use_container_width=True)
st.markdown(f"""<div class="fig-caption">
  <b>Figure 7.</b> 60-day rolling market beta. Full-period beta of {beta:.2f} shown for
  reference. Excursions above 1.3 or below 0.7 indicate periods of materially elevated
  or reduced market sensitivity.
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 9 — VAR + CVAR
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">9. Value at Risk and Expected Shortfall</div>',
            unsafe_allow_html=True)

var_data = {
    "Method": ["Historical (empirical)", "Parametric (Gaussian)"],
    "VaR 95%": [f"{abs(var_hist_95):.2%}", f"{abs(var_param_95):.2%}"],
    "CVaR 95%": [f"{abs(cvar_hist_95):.2%}", f"{abs(cvar_param_95):.2%}"],
    "VaR 99%": [f"{abs(var_hist_99):.2%}", f"{abs(var_param_99):.2%}"],
    "CVaR 99%": [f"{abs(cvar_hist_99):.2%}", f"{abs(cvar_param_99):.2%}"],
    "Assumption": [
        "No distributional assumption; uses empirical return percentiles",
        "Returns normally distributed with sample mean and std dev",
    ],
}
st.dataframe(pd.DataFrame(var_data), use_container_width=True, hide_index=True)

hist_param_diff = abs(var_hist_95) - abs(var_param_95)
tail_note = (
    f"The historical estimate exceeds the parametric at 95% by {abs(hist_param_diff):.2%}, "
    f"suggesting fatter left tails than Gaussian."
    if hist_param_diff > 0 else
    "The parametric estimate exceeds the historical at 95%, suggesting the empirical "
    "distribution is somewhat thinner-tailed than Gaussian."
)
cvar_spread = abs(cvar_hist_95) - abs(var_hist_95)
st.markdown(f"""<div class="fig-caption">
  <b>Table 4.</b> VaR and CVaR (Expected Shortfall) at 95% and 99% confidence.
  z(0.95) = 1.645, z(0.99) = 2.326. {tail_note}
  The CVaR spread above VaR at 95% is {cvar_spread:.2%}, reflecting average severity
  of tail losses — a key input to Basel III/IV risk frameworks that VaR alone misses.
</div>""", unsafe_allow_html=True)
st.latex(r"\text{CVaR}_\alpha = -\,\mathbb{E}\!\left[r \mid r \leq \text{VaR}_\alpha\right]")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 10 — INDIVIDUAL ASSET RISK CONTRIBUTION
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">10. Individual asset risk contribution</div>',
            unsafe_allow_html=True)

asset_vols   = clean_df.std() * np.sqrt(252)
marginal_rc  = cov_matrix.values @ weights
component_rc = marginal_rc * weights
total_risk_sq = weights @ cov_matrix.values @ weights
pct_rc       = component_rc / total_risk_sq * 100

risk_df = pd.DataFrame({
    "Asset":                   [NAMES.get(t,t) for t in tickers_in],
    "Ticker":                  tickers_in,
    "Weight":                  [f"{w:.1%}" for w in weights],
    "Volatility (ann.)":       [f"{v:.2%}" for v in asset_vols],
    "% Variance contribution": [f"{p:.1f}%" for p in pct_rc],
    "Risk/Weight ratio":       [f"{p/(w*100):.2f}x" for p,w in zip(pct_rc, weights)],
}).sort_values("% Variance contribution", ascending=False,
               key=lambda x: x.str.rstrip('%x').astype(float))
st.dataframe(risk_df, use_container_width=True, hide_index=True)

fig_rc = go.Figure(go.Bar(
    x=[NAMES.get(t,t) for t in tickers_in], y=pct_rc,
    marker_color=[LBLUE if p >= 100/n_assets else LLBLUE for p in pct_rc],
    opacity=0.85,
    text=[f"{p:.1f}%" for p in pct_rc],
    textposition="outside", textfont=dict(size=10, color="#333"),
))
fig_rc.add_hline(y=100/n_assets, line_dash="dot", line_color=GRAY, line_width=1.2,
                 annotation_text=f"Equal contribution ({100/n_assets:.1f}%)",
                 annotation_font=dict(size=9, color=GRAY), annotation_position="top right")
fig_rc.update_layout(
    **layout(margin=dict(l=8, r=120, t=20, b=8), height=320),
    xaxis={**ax("Asset"), "tickangle": -35},
    yaxis=dict(**ax("% contribution to portfolio variance"), ticksuffix="%"),
    showlegend=False,
)
st.plotly_chart(fig_rc, use_container_width=True)
top_rc = tickers_in[np.argmax(pct_rc)]
st.markdown(f"""<div class="fig-caption">
  <b>Figure 8.</b> Component variance contribution per position. Darker blue exceeds the
  equal-contribution threshold. {NAMES.get(top_rc,top_rc)} is the largest single contributor.
  The Risk/Weight ratio in the table above shows how much each position punches above
  its capital weight — a ratio above 1.0x means it contributes more risk than capital.
</div>""", unsafe_allow_html=True)
st.latex(r"RC_i = \frac{w_i \, (\Sigma \mathbf{w})_i}{\mathbf{w}^T \Sigma \mathbf{w}}")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 11 — FAMA-FRENCH 3-FACTOR ATTRIBUTION
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">11. Fama-French three-factor attribution</div>',
            unsafe_allow_html=True)
st.markdown("""<div class="explainer-body">
  The Fama-French three-factor model extends the CAPM by adding size (SMB) and value (HML)
  factors. SMB (Small Minus Big) captures the historical small-cap premium.
  HML (High Minus Low) captures the value premium. Most large-cap growth portfolios have
  negative HML loadings. A positive and statistically significant FF alpha after controlling
  for all three factors suggests genuine skill or a structural edge.
</div>""", unsafe_allow_html=True)

ff3_loaded = ff3 is not None and not ff3.empty
if ff3_loaded:
    ff_aligned = ff3[["Mkt-RF","SMB","HML"]].reindex(port_aligned.index).dropna()
    port_ff    = port_aligned.reindex(ff_aligned.index) - RF_DAILY

    Y = port_ff.values
    X = np.column_stack([np.ones(len(ff_aligned)),
                         ff_aligned["Mkt-RF"].values,
                         ff_aligned["SMB"].values,
                         ff_aligned["HML"].values])
    ff_coefs, ff_r2 = ols(Y, X)
    ff_alpha, ff_mkt, ff_smb, ff_hml = ff_coefs[0]*252, ff_coefs[1], ff_coefs[2], ff_coefs[3]

    resid   = Y - X @ ff_coefs
    sig2    = (resid**2).sum() / (len(Y) - 4)
    se      = np.sqrt(np.diag(sig2 * np.linalg.inv(X.T @ X)))
    t_stats = ff_coefs / se
    t_alpha, t_mkt, t_smb, t_hml = t_stats

    ff1, ff2, ff3c, ff4 = st.columns(4)
    ff_cards = [
        ("FF alpha (ann.)", f"{sgn(ff_alpha)}", f"t = {t_alpha:.2f}",
         "pos" if ff_alpha > 0 else "neg"),
        ("Market (Mkt-RF)", f"{ff_mkt:.3f}", f"t = {t_mkt:.2f}", "neut"),
        ("Size (SMB)",      f"{ff_smb:.3f}", f"t = {t_smb:.2f}",
         "neut" if abs(ff_smb) < 0.1 else ("pos" if ff_smb > 0 else "warn")),
        ("Value (HML)",     f"{ff_hml:.3f}", f"t = {t_hml:.2f}",
         "neut" if abs(ff_hml) < 0.1 else ("pos" if ff_hml > 0 else "warn")),
    ]
    for col, (label, value, sub, cls) in zip([ff1, ff2, ff3c, ff4], ff_cards):
        with col:
            st.markdown(f"""<div class="kpi-card">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value {cls}">{value}</div>
              <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    factor_names  = ["Market (Mkt-RF)", "Size (SMB)", "Value (HML)"]
    factor_vals   = [ff_mkt, ff_smb, ff_hml]
    fig_ff = go.Figure(go.Bar(
        x=factor_names, y=factor_vals,
        marker_color=[LBLUE if v >= 0 else LRED for v in factor_vals], opacity=0.85,
        text=[f"{v:.3f}" for v in factor_vals],
        textposition="outside", textfont=dict(size=12, color="#333"),
        hovertemplate="%{x}: %{y:.4f}<extra></extra>",
    ))
    fig_ff.add_hline(y=0, line_color="#d4c9b8", line_width=1)
    fig_ff.update_layout(
        **layout(margin=dict(l=60, r=20, t=30, b=8), height=300),
        xaxis=dict(**ax("Factor", grid=False)),
        yaxis=dict(**ax_bare("Factor loading (beta)"),
                   gridcolor="#e8e0d0", linecolor="#d4c9b8", linewidth=1,
                   showline=True, showgrid=True, zeroline=True,
                   zerolinecolor="#d4c9b8", ticks="outside", ticklen=3),
        showlegend=False,
    )
    st.plotly_chart(fig_ff, use_container_width=True)

    # Rolling FF alpha
    roll_ff_alpha = pd.Series(index=port_aligned.index, dtype=float)
    for end_i in range(roll_win, len(port_aligned)):
        idx_slice = port_aligned.index[end_i-roll_win:end_i]
        pf_sl     = port_aligned.loc[idx_slice] - RF_DAILY
        ff_sl     = ff3[["Mkt-RF","SMB","HML"]].reindex(idx_slice).dropna()
        if len(ff_sl) < 20: continue
        pf_sl = pf_sl.reindex(ff_sl.index)
        Xw = np.column_stack([np.ones(len(ff_sl)), ff_sl["Mkt-RF"].values,
                              ff_sl["SMB"].values, ff_sl["HML"].values])
        bw, _ = ols(pf_sl.values, Xw)
        roll_ff_alpha.iloc[end_i] = bw[0] * 252

    roll_ff_alpha = roll_ff_alpha.dropna()
    if len(roll_ff_alpha) > 10:
        fig_rfa = go.Figure()
        fig_rfa.add_trace(go.Scatter(
            x=roll_ff_alpha.index, y=roll_ff_alpha.values*100,
            mode="lines", line=dict(color=GOLD, width=2.2),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Alpha: %{y:.2f}%<extra></extra>",
        ))
        fig_rfa.add_hline(y=0, line_dash="dot", line_color=GRAY, line_width=1)
        fig_rfa.update_layout(
            **layout(margin=dict(l=70, r=20, t=20, b=8), height=260,
                     xaxis=dict(**ax("Date")),
                     yaxis=dict(**ax("Annualized FF alpha (%)"), ticksuffix="%"),
                     showlegend=False),
        )
        st.plotly_chart(fig_rfa, use_container_width=True)

    smb_interp = ("small-cap tilt" if ff_smb > 0.1 else "large-cap tilt" if ff_smb < -0.1 else "size-neutral")
    hml_interp = ("value tilt" if ff_hml > 0.1 else "growth tilt" if ff_hml < -0.1 else "style-neutral")

    st.markdown(f"""<div class="fig-caption">
      <b>Figure 9.</b> Fama-French three-factor loadings estimated by OLS on daily excess returns.
      R² = {ff_r2:.1%}. Market loading of {ff_mkt:.3f} (t = {t_mkt:.2f}) is the primary driver.
      SMB of {ff_smb:.3f} (t = {t_smb:.2f}) indicates a {smb_interp};
      HML of {ff_hml:.3f} (t = {t_hml:.2f}) indicates a {hml_interp}.
      FF alpha of {sgn(ff_alpha)} annualized (t = {t_alpha:.2f}) represents return unexplained
      by all three factors.
    </div>""", unsafe_allow_html=True)
    st.latex(
        r"r_p - r_f = \alpha + \beta_{\text{Mkt}}(r_m - r_f) + "
        r"\beta_{\text{SMB}} \cdot \text{SMB} + \beta_{\text{HML}} \cdot \text{HML} + \varepsilon"
    )
    ff_results_df = pd.DataFrame({
        "Factor":  ["Alpha (ann.)", "Mkt-RF", "SMB", "HML"],
        "Loading": [f"{sgn(ff_alpha)}", f"{ff_mkt:.4f}", f"{ff_smb:.4f}", f"{ff_hml:.4f}"],
        "t-stat":  [f"{t_alpha:.2f}", f"{t_mkt:.2f}", f"{t_smb:.2f}", f"{t_hml:.2f}"],
        "Significant (5%)": ["Yes" if abs(t) > 1.96 else "No"
                              for t in [t_alpha, t_mkt, t_smb, t_hml]],
    })
    st.dataframe(ff_results_df, use_container_width=True, hide_index=True)
else:
    st.info("Fama-French data unavailable. All other sections remain fully functional.")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 12 — EFFICIENT FRONTIER & PORTFOLIO OPTIMIZATION  ★ NEW ★
# ══════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="sec-header">12. Mean-variance efficient frontier'
    '<span class="new-badge">New</span></div>',
    unsafe_allow_html=True
)
st.markdown("""<div class="explainer-body">
  The mean-variance efficient frontier, introduced by Harry Markowitz (1952), traces the set
  of portfolios offering the highest expected return for a given level of risk. Three portfolios
  are highlighted: the current equal-weight portfolio, the minimum-variance portfolio (lowest
  possible risk), and the maximum-Sharpe portfolio (highest risk-adjusted return given the
  risk-free rate). Individual assets are plotted for reference. This section answers a question
  the prior metrics could only imply: <em>how efficiently is capital allocated?</em>
</div>""", unsafe_allow_html=True)

# ── Build frontier ──────────────────────────────────────────────────────
mu_assets  = clean_df.mean().values * 252        # annualized expected returns
cov_ann    = cov_matrix.values                    # annualized covariance
n_a        = len(mu_assets)
bounds     = tuple((0.0, 1.0) for _ in range(n_a))
constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]

def port_stats(w):
    ret = float(w @ mu_assets)
    vol = float(np.sqrt(w @ cov_ann @ w))
    return ret, vol

def neg_sharpe(w):
    r, v = port_stats(w)
    return -(r - RF_ANNUAL) / v if v > 0 else 0.0

def min_vol_for_ret(target_ret):
    cons = constraints + [{"type": "eq", "fun": lambda w: w @ mu_assets - target_ret}]
    res = minimize(lambda w: np.sqrt(w @ cov_ann @ w),
                   np.ones(n_a)/n_a, method="SLSQP",
                   bounds=bounds, constraints=cons,
                   options={"ftol": 1e-12, "maxiter": 1000})
    return res

# Min-variance portfolio
res_mv = minimize(lambda w: np.sqrt(w @ cov_ann @ w),
                  np.ones(n_a)/n_a, method="SLSQP",
                  bounds=bounds, constraints=constraints,
                  options={"ftol": 1e-12, "maxiter": 1000})
w_mv     = res_mv.x
ret_mv, vol_mv = port_stats(w_mv)
sharpe_mv = (ret_mv - RF_ANNUAL) / vol_mv

# Max-Sharpe portfolio
res_ms = minimize(neg_sharpe, np.ones(n_a)/n_a, method="SLSQP",
                  bounds=bounds, constraints=constraints,
                  options={"ftol": 1e-12, "maxiter": 1000})
w_ms     = res_ms.x
ret_ms, vol_ms = port_stats(w_ms)
sharpe_ms = (ret_ms - RF_ANNUAL) / vol_ms

# Risk parity portfolio
def risk_parity_weights(cov_arr, n):
    def rp_obj(w):
        w = np.array(w)
        port_var = w @ cov_arr @ w
        rc = w * (cov_arr @ w) / port_var
        return np.sum((rc - 1.0/n)**2)
    res = minimize(rp_obj, np.ones(n)/n, method="SLSQP",
                   bounds=tuple((1e-4, 1.0) for _ in range(n)),
                   constraints=[{"type":"eq","fun": lambda w: w.sum()-1}],
                   options={"ftol":1e-14,"maxiter":2000})
    return res.x / res.x.sum()

w_rp         = risk_parity_weights(cov_ann, n_a)
ret_rp, vol_rp = port_stats(w_rp)
sharpe_rp    = (ret_rp - RF_ANNUAL) / vol_rp

# Frontier curve
ret_range = np.linspace(mu_assets.min()*0.9, mu_assets.max()*0.9, 80)
frontier_vols, frontier_rets = [], []
for rt in ret_range:
    res = min_vol_for_ret(rt)
    if res.success:
        frontier_vols.append(float(np.sqrt(res.x @ cov_ann @ res.x)))
        frontier_rets.append(rt)

# Current portfolio stats
ret_eq, vol_eq = port_stats(weights)
sharpe_eq      = (ret_eq - RF_ANNUAL) / vol_eq

# Individual asset stats
asset_rets_ann = mu_assets
asset_vols_ann = np.sqrt(np.diag(cov_ann))

# ── Plot ────────────────────────────────────────────────────────────────
fig_ef = go.Figure()

# Individual assets (scatter)
fig_ef.add_trace(go.Scatter(
    x=asset_vols_ann, y=asset_rets_ann, mode="markers",
    marker=dict(color=LGRAY, size=6, symbol="circle",
                line=dict(color=GRAY, width=0.75)),
    name="Individual assets",
    text=[NAMES.get(t,t) for t in tickers_in],
    hovertemplate="%{text}<br>Vol: %{x:.1%}<br>Return: %{y:.1%}<extra></extra>",
))

# Frontier
if frontier_vols:
    # Capital market line from risk-free to max-Sharpe
    cml_vols = [0, vol_ms * 1.5]
    cml_rets = [RF_ANNUAL, RF_ANNUAL + sharpe_ms * vol_ms * 1.5]
    fig_ef.add_trace(go.Scatter(
        x=cml_vols, y=cml_rets, mode="lines",
        line=dict(color=GOLD, width=1.5, dash="dot"),
        name="Capital Market Line", hoverinfo="skip",
    ))
    fig_ef.add_trace(go.Scatter(
        x=frontier_vols, y=frontier_rets, mode="lines",
        line=dict(color=GRAY, width=2), name="Efficient frontier",
        hovertemplate="Vol: %{x:.1%}<br>Return: %{y:.1%}<extra></extra>",
    ))

# Highlighted portfolios
for (rx, vx, label, color, sym, sz) in [
    (ret_eq, vol_eq,   f"Equal-weight (Sharpe {sharpe_eq:.2f})", RED,    "diamond", 14),
    (ret_mv, vol_mv,   f"Min-variance  (Sharpe {sharpe_mv:.2f})",  PURPLE, "star",    15),
    (ret_ms, vol_ms,   f"Max-Sharpe    (Sharpe {sharpe_ms:.2f})",  GREEN,  "star",    15),
    (ret_rp, vol_rp,   f"Risk parity   (Sharpe {sharpe_rp:.2f})",  GOLD,   "circle",  13),
]:
    fig_ef.add_trace(go.Scatter(
        x=[vx], y=[rx], mode="markers",
        marker=dict(color=color, size=sz, symbol=sym,
                    line=dict(color="#ffffff", width=1.5)),
        name=label,
        hovertemplate=f"<b>{label}</b><br>Vol: %{{x:.2%}}<br>Return: %{{y:.2%}}<extra></extra>",
    ))

fig_ef.update_layout(
    **layout(margin=dict(l=60, r=20, t=30, b=8),
       legend=dict(orientation="v", x=0.01, y=0.99, xanchor="left", yanchor="top",
                   font=dict(size=11, color="#1a1a1a"), bgcolor="rgba(245,240,232,0.85)",
                   bordercolor="#d4c9b8", borderwidth=0.75)),
    height=460,
    xaxis=dict(**ax("Annualized volatility"), tickformat=".0%"),
    yaxis=dict(**ax("Annualized expected return"), tickformat=".0%"),
)
st.plotly_chart(fig_ef, use_container_width=True)

st.markdown(f"""<div class="fig-caption">
  <b>Figure 10.</b> Mean-variance efficient frontier constructed from the 20-asset
  universe using daily return covariance estimates. Individual assets shown in gray.
  The Capital Market Line (gold dashed) passes through the risk-free rate and the
  max-Sharpe portfolio. The equal-weight portfolio (red diamond) sits below the frontier,
  confirming that analytically better allocations exist within this same universe.
</div>""", unsafe_allow_html=True)

# ── Comparison table ────────────────────────────────────────────────────
st.markdown('<div class="sec-header" style="margin-top:1rem;">Portfolio comparison</div>',
            unsafe_allow_html=True)

def pct_wt(w, top_n=3):
    idx = np.argsort(w)[::-1][:top_n]
    return ", ".join(f"{NAMES.get(tickers_in[i],tickers_in[i])} {w[i]:.0%}" for i in idx)

compare_data = {
    "Portfolio":           ["Equal-weight", "Min-variance", "Max-Sharpe", "Risk parity"],
    "Ann. return":         [f"{ret_eq:.1%}", f"{ret_mv:.1%}", f"{ret_ms:.1%}", f"{ret_rp:.1%}"],
    "Ann. volatility":     [f"{vol_eq:.1%}", f"{vol_mv:.1%}", f"{vol_ms:.1%}", f"{vol_rp:.1%}"],
    "Sharpe ratio":        [f"{sharpe_eq:.2f}", f"{sharpe_mv:.2f}", f"{sharpe_ms:.2f}", f"{sharpe_rp:.2f}"],
    "Top 3 positions":     [pct_wt(weights), pct_wt(w_mv), pct_wt(w_ms), pct_wt(w_rp)],
}
st.dataframe(pd.DataFrame(compare_data), use_container_width=True, hide_index=True)

# KPI callout cards — improvements over equal weight
e1, e2, e3 = st.columns(3)
vol_reduction = (vol_eq - vol_mv) / vol_eq
sharpe_improvement = sharpe_ms - sharpe_eq
rp_vol_reduction = (vol_eq - vol_rp) / vol_eq
for col, label, val, sub, cls in [
    (e1, "Min-var vol reduction",
     f"−{vol_reduction:.1%}", f"vs. equal-weight ({vol_eq:.1%} → {vol_mv:.1%})", "pos"),
    (e2, "Max-Sharpe improvement",
     f"+{sharpe_improvement:.2f}", f"Sharpe {sharpe_eq:.2f} → {sharpe_ms:.2f}", "pos"),
    (e3, "Risk parity vol reduction",
     f"−{rp_vol_reduction:.1%}", f"vol {vol_eq:.1%} → {vol_rp:.1%}", "pos"),
]:
    with col:
        st.markdown(f"""<div class="kpi-card-highlight">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value {cls}">{val}</div>
          <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown(f"""<div class="fig-caption" style="margin-top:0.75rem;">
  <b>Table 5.</b> Comparison of four weighting schemes on the same 20-asset universe.
  All metrics derived from the in-sample return and covariance estimates — forward-looking
  performance may differ. The min-variance and max-Sharpe portfolios assume long-only weights
  (no short positions) and are solved via constrained quadratic programming (scipy SLSQP).
  Risk parity targets equal variance contribution per asset, solved by minimizing the sum
  of squared deviations from equal risk shares.
</div>""", unsafe_allow_html=True)
st.latex(r"\min_{\mathbf{w}} \;\mathbf{w}^\top\Sigma\mathbf{w} \quad \text{s.t.} \quad \mathbf{1}^\top\mathbf{w}=1,\; \mathbf{w}\geq 0")

# ── Max-Sharpe weight breakdown ──────────────────────────────────────────
with st.expander("Max-Sharpe portfolio weights (full breakdown)"):
    ms_df = pd.DataFrame({
        "Ticker": tickers_in,
        "Company": [NAMES.get(t,t) for t in tickers_in],
        "Max-Sharpe weight": [f"{w:.1%}" for w in w_ms],
        "Equal weight": ["5.0%"] * n_a,
        "Δ vs equal weight": [f"{(w-0.05)*100:+.1f}pp" for w in w_ms],
    }).sort_values("Max-Sharpe weight", ascending=False,
                   key=lambda x: x.str.rstrip('%').astype(float))
    st.dataframe(ms_df, use_container_width=True, hide_index=True)

with st.expander("Risk parity portfolio weights (full breakdown)"):
    rp_df = pd.DataFrame({
        "Ticker": tickers_in,
        "Company": [NAMES.get(t,t) for t in tickers_in],
        "Risk parity weight": [f"{w:.1%}" for w in w_rp],
        "Equal weight": ["5.0%"] * n_a,
        "Δ vs equal weight": [f"{(w-0.05)*100:+.1f}pp" for w in w_rp],
    }).sort_values("Risk parity weight", ascending=False,
                   key=lambda x: x.str.rstrip('%').astype(float))
    st.dataframe(rp_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 13 — HISTORICAL STRESS TESTING  ★ NEW ★
# ══════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="sec-header">13. Historical stress testing'
    '<span class="new-badge">New</span></div>',
    unsafe_allow_html=True
)
st.markdown("""<div class="explainer-body">
  Stress testing answers: <em>how would this portfolio have performed during the worst
  historical market dislocations?</em> Because the portfolio holds only large-cap US stocks,
  performance during these periods is simulated by applying the portfolio's estimated beta
  to the realized S&P 500 return in each episode, then adding the estimated FF alpha intercept
  as a per-annum adjustment. This yields a model-implied portfolio return that accounts for
  both market exposure and historical factor tilts. The equal-weight, min-variance, and
  max-Sharpe portfolios are compared across all scenarios.
</div>""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def get_spx_scenario_returns(spx_prices, scenarios):
    """Compute S&P 500 return during each scenario window."""
    results = {}
    # yfinance may return a MultiIndex or simple DataFrame for single ticker
    if isinstance(spx_prices.columns, pd.MultiIndex):
        spx_prices = spx_prices.droplevel(1, axis=1)
    spx_col = BENCHMARK if BENCHMARK in spx_prices.columns else spx_prices.columns[0]
    spx_ret = spx_prices[spx_col].pct_change().dropna()
    for name, (start_str, end_str) in scenarios.items():
        try:
            slc = spx_ret.loc[start_str:end_str]
            if len(slc) < 5:
                results[name] = None
                continue
            cum = (1 + slc).prod() - 1
            n_days = len(slc)
            results[name] = {"spx_return": cum, "n_days": n_days,
                             "start": slc.index[0].strftime('%Y-%m-%d'),
                             "end":   slc.index[-1].strftime('%Y-%m-%d')}
        except Exception:
            results[name] = None
    return results

spx_scenarios = get_spx_scenario_returns(spx_ext, STRESS_SCENARIOS)

# Simulate portfolio returns using beta + alpha adjustment
def simulate_stress(beta_val, alpha_ann_val, spx_ret, n_days):
    """Model-implied portfolio return: beta * Rm + alpha*(n_days/252)"""
    alpha_period = alpha_ann_val * (n_days / 252)
    return beta_val * spx_ret + alpha_period

def simulate_stress_weights(w, mu_a, cov_a, spx_ret, n_days, bench_vol_ann):
    """Compute implied beta from weights and apply it."""
    # Use weighted average beta (approximation: each asset beta estimated from portfolio cov)
    port_v = float(np.sqrt(w @ cov_a @ w))
    bench_v = bench_vol_ann
    port_bench_cov = float(cov_a @ w @ np.ones(len(w)))  # rough; use direct calc below
    # Re-estimate: implied beta ≈ port_vol / bench_vol * correlation (Sharpe's approximation)
    # Better: use the stored OLS betas
    return port_v / bench_v * 0.98  # approximate; good enough for scenario comparison

# Use OLS betas computed from our data
stress_results = []
portfolios_stress = {
    "Equal-weight":  (beta,           alpha_ann,           weights),
    "Min-variance":  (None,           None,                w_mv),
    "Max-Sharpe":    (None,           None,                w_ms),
}

# Compute implied betas for mv and ms
def implied_beta_from_weights(w, cov_a, bench_daily_returns, daily_returns_df):
    port_r = daily_returns_df.dot(w)
    bench_r = bench_daily_returns.reindex(port_r.index).dropna()
    port_r  = port_r.reindex(bench_r.index)
    cov_val = np.cov(port_r.values, bench_r.values)[0,1]
    var_val = np.var(bench_r.values)
    return cov_val / var_val if var_val > 0 else 1.0

beta_mv = implied_beta_from_weights(w_mv, cov_ann, bench_aligned, clean_df)
beta_ms = implied_beta_from_weights(w_ms, cov_ann, bench_aligned, clean_df)

# Alpha for mv/ms from FF regression if available
ff_alpha_mv = alpha_ann  # fallback — in practice would need separate regression
ff_alpha_ms = alpha_ann

if ff3_loaded:
    for w_test, name_test in [(w_mv, "mv"), (w_ms, "ms")]:
        p_r = clean_df.dot(w_test)
        p_aligned = p_r.reindex(ff_aligned.index)
        p_excess  = p_aligned - RF_DAILY
        if len(p_excess.dropna()) > 30:
            Y2 = p_excess.dropna().values
            X2 = np.column_stack([np.ones(len(Y2)),
                                   ff_aligned.reindex(p_excess.dropna().index)["Mkt-RF"].values,
                                   ff_aligned.reindex(p_excess.dropna().index)["SMB"].values,
                                   ff_aligned.reindex(p_excess.dropna().index)["HML"].values])
            if X2.shape[0] == len(Y2):
                c2, _ = ols(Y2, X2)
                if name_test == "mv":
                    ff_alpha_mv = c2[0] * 252
                else:
                    ff_alpha_ms = c2[0] * 252

portfolios_stress = {
    "Equal-weight": (beta,    alpha_ann,    weights),
    "Min-variance": (beta_mv, ff_alpha_mv,  w_mv),
    "Max-Sharpe":   (beta_ms, ff_alpha_ms,  w_ms),
}

scenario_rows = []
for scenario_name, scenario_data in spx_scenarios.items():
    if scenario_data is None:
        continue
    row = {"Scenario": scenario_name,
           "S&P 500 return": scenario_data["spx_return"],
           "Period": f"{scenario_data['start']} → {scenario_data['end']}",
           "Days": scenario_data["n_days"]}
    for port_name, (b_val, a_val, _) in portfolios_stress.items():
        implied = simulate_stress(b_val, a_val, scenario_data["spx_return"],
                                  scenario_data["n_days"])
        row[port_name] = implied
    scenario_rows.append(row)

if scenario_rows:
    stress_df = pd.DataFrame(scenario_rows)

    # Display as a bar chart grouped by scenario
    fig_stress = go.Figure()
    port_colors = {"Equal-weight": LBLUE, "Min-variance": PURPLE, "Max-Sharpe": GREEN}
    for port_name, color in port_colors.items():
        fig_stress.add_trace(go.Bar(
            name=port_name,
            x=[r["Scenario"].split(" (")[0][:35] for r in scenario_rows],
            y=[r[port_name]*100 for r in scenario_rows],
            marker_color=color, opacity=0.82,
            text=[f"{r[port_name]:.1%}" for r in scenario_rows],
            textposition="outside", textfont=dict(size=9, color="#333"),
            hovertemplate="%{x}<br>" + port_name + ": %{y:.2f}%<extra></extra>",
        ))
    # S&P 500 actual
    fig_stress.add_trace(go.Scatter(
        name="S&P 500 (actual)",
        x=[r["Scenario"].split(" (")[0][:35] for r in scenario_rows],
        y=[r["S&P 500 return"]*100 for r in scenario_rows],
        mode="markers+lines",
        marker=dict(color=RED, size=9, symbol="x"),
        line=dict(color=RED, width=1, dash="dot"),
        hovertemplate="%{x}<br>S&P 500: %{y:.2f}%<extra></extra>",
    ))
    fig_stress.add_hline(y=0, line_color="#d4c9b8", line_width=1)
    fig_stress.update_layout(
        **layout(margin=dict(l=8, r=20, t=30, b=120), height=440, barmode="group",
                 legend=dict(orientation="h", y=1.05, x=0, font=dict(size=11), bgcolor="rgba(0,0,0,0)")),
        xaxis=dict(**ax_bare(""), tickangle=-30,
                   **AXIS_STYLE_NOGRID),
        yaxis=dict(**ax("Model-implied portfolio return (%)"), ticksuffix="%"),
    )
    st.plotly_chart(fig_stress, use_container_width=True)

    # Scenario detail table
    display_stress = stress_df.copy()
    display_stress["S&P 500"] = display_stress["S&P 500 return"].map(lambda x: f"{x:.1%}")
    for p in ["Equal-weight", "Min-variance", "Max-Sharpe"]:
        display_stress[p] = display_stress[p].map(lambda x: f"{x:.1%}")
    display_stress = display_stress[["Scenario","Period","Days","S&P 500",
                                     "Equal-weight","Min-variance","Max-Sharpe"]]
    st.dataframe(display_stress, use_container_width=True, hide_index=True)

    # Worst-case summary cards
    worst_eq = min(stress_df["Equal-weight"])
    worst_mv = min(stress_df["Min-variance"])
    worst_ms = min(stress_df["Max-Sharpe"])
    avg_eq   = stress_df["Equal-weight"].mean()
    avg_mv   = stress_df["Min-variance"].mean()
    mv_improvement = worst_mv - worst_eq

    sc1, sc2, sc3 = st.columns(3)
    for col, label, val, sub, cls in [
        (sc1, "Worst-case equal-weight",   f"{worst_eq:.1%}", "Across all 6 scenarios", "neg"),
        (sc2, "Worst-case min-variance",   f"{worst_mv:.1%}", f"{sgn(mv_improvement)} vs equal-weight", "warn"),
        (sc3, "Worst-case max-Sharpe",     f"{worst_ms:.1%}", "Highest-Sharpe portfolio", "warn"),
    ]:
        with col:
            st.markdown(f"""<div class="stress-card">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value {cls}">{val}</div>
              <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="fig-caption">
      <b>Figure 11.</b> Model-implied portfolio performance across six historical market
      dislocations, vs. realized S&P 500 return (red ×). Returns simulated using
      each portfolio's estimated market beta and annualized FF alpha: implied return =
      β × R<sub>market</sub> + α × (days/252). The min-variance portfolio's lower beta
      provides meaningful downside protection in severe drawdown episodes.
      All results are in-sample approximations; actual outcomes depend on intraperiod
      factor dynamics not captured by this model.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 14 — VOLATILITY REGIME DETECTION  ★ NEW ★
# ══════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="sec-header">14. Volatility regime detection'
    '<span class="new-badge">New</span></div>',
    unsafe_allow_html=True
)
st.markdown("""<div class="explainer-body">
  Market volatility clusters: low-volatility regimes dominate most of the calendar, punctuated
  by shorter high-volatility regimes during stress episodes. Understanding which regime the
  portfolio is currently in has direct implications for position sizing, hedging, and risk
  budget consumption. The approach here uses a rolling 20-day realized volatility and classifies
  each trading day into a low or high regime based on whether it exceeds the full-period
  annualized volatility threshold. This is a practical, interpretable approximation of a
  two-state Hidden Markov Model.
</div>""", unsafe_allow_html=True)

roll_20_vol = port_returns.rolling(20).std() * np.sqrt(252)
vol_threshold = port_vol   # full-period annualized vol as regime boundary
regime = roll_20_vol.apply(lambda v: "High" if (not np.isnan(v) and v > vol_threshold) else "Low")

# Assign colors
regime_colors_ts = regime.map({"Low": LGREEN, "High": LRED})

fig_regime = go.Figure()

# Background shading for high-vol periods
high_vol_mask = (regime == "High") & regime.shift(1).isin(["Low", None])
low_after_high = (regime == "Low") & regime.shift(1).eq("High")

# Draw rolling vol with regime color
low_vol_series  = roll_20_vol.where(regime == "Low")
high_vol_series = roll_20_vol.where(regime == "High")

fig_regime.add_trace(go.Scatter(
    x=roll_20_vol.index, y=roll_20_vol.values * 100,
    mode="lines", line=dict(color=LGRAY, width=1), showlegend=False,
    hoverinfo="skip",
))
fig_regime.add_trace(go.Scatter(
    x=low_vol_series.index, y=low_vol_series.values * 100,
    mode="lines", line=dict(color=GREEN, width=2.5), name="Low-vol regime",
    connectgaps=False,
    hovertemplate="Date: %{x|%Y-%m-%d}<br>20d vol: %{y:.1f}%<extra></extra>",
))
fig_regime.add_trace(go.Scatter(
    x=high_vol_series.index, y=high_vol_series.values * 100,
    mode="lines", line=dict(color=RED, width=2.5), name="High-vol regime",
    connectgaps=False,
    hovertemplate="Date: %{x|%Y-%m-%d}<br>20d vol: %{y:.1f}%<extra></extra>",
))
fig_regime.add_hline(
    y=vol_threshold * 100, line_dash="dash", line_color=GOLD, line_width=1.5,
    annotation_text=f"Regime threshold ({vol_threshold:.1%})",
    annotation_font=dict(size=9, color=GOLD), annotation_position="top right",
)
fig_regime.update_layout(
    **layout(margin=dict(l=70, r=20, t=20, b=8), height=300),
    xaxis=dict(**ax("Date")),
    yaxis=dict(**ax("20-day rolling volatility (annualized %)"), ticksuffix="%"),
)
st.plotly_chart(fig_regime, use_container_width=True)

# Regime statistics
regime_clean = regime.dropna()
low_days     = (regime_clean == "Low").sum()
high_days    = (regime_clean == "High").sum()
total_days   = len(regime_clean)
pct_low      = low_days / total_days
pct_high     = high_days / total_days

low_returns  = port_returns[regime_clean == "Low"]
high_returns = port_returns[regime_clean == "High"]
avg_low_ret  = low_returns.mean() * 252 if len(low_returns) > 0 else np.nan
avg_high_ret = high_returns.mean() * 252 if len(high_returns) > 0 else np.nan
avg_low_vol  = low_returns.std() * np.sqrt(252) if len(low_returns) > 0 else np.nan
avg_high_vol = high_returns.std() * np.sqrt(252) if len(high_returns) > 0 else np.nan

# Current regime
current_regime = regime_clean.iloc[-1] if len(regime_clean) > 0 else "Unknown"
current_vol    = roll_20_vol.iloc[-1] if not np.isnan(roll_20_vol.iloc[-1]) else vol_threshold

r1, r2, r3, r4 = st.columns(4)
for col, label, val, sub, cls in [
    (r1, "Current regime",
     current_regime,
     f"20d vol: {current_vol:.1%}",
     "pos" if current_regime == "Low" else "neg"),
    (r2, "Low-vol days",
     f"{low_days} ({pct_low:.0%})",
     f"Avg ann. return: {avg_low_ret:.1%}" if not np.isnan(avg_low_ret) else "",
     "pos"),
    (r3, "High-vol days",
     f"{high_days} ({pct_high:.0%})",
     f"Avg ann. return: {avg_high_ret:.1%}" if not np.isnan(avg_high_ret) else "",
     "neg"),
    (r4, "Regime threshold",
     f"{vol_threshold:.1%}",
     "Full-period portfolio volatility",
     "neut"),
]:
    with col:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value {cls}">{val}</div>
          <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

# Regime return comparison chart
if len(low_returns) > 5 and len(high_returns) > 5:
    fig_rr = go.Figure()
    kde_low_x,  kde_low_d  = kde_curve(low_returns.values)
    kde_high_x, kde_high_d = kde_curve(high_returns.values)
    if len(kde_low_x):
        bw = (low_returns.max()-low_returns.min()) / 40
        fig_rr.add_trace(go.Scatter(
            x=kde_low_x, y=kde_low_d * len(low_returns) * bw,
            mode="lines", line=dict(color=GREEN, width=2.5), name="Low-vol regime",
            fill="tozeroy", fillcolor="rgba(46,125,79,0.12)",
            hovertemplate="Return: %{x:.3%}<extra></extra>",
        ))
    if len(kde_high_x):
        bw2 = (high_returns.max()-high_returns.min()) / 40
        fig_rr.add_trace(go.Scatter(
            x=kde_high_x, y=kde_high_d * len(high_returns) * bw2,
            mode="lines", line=dict(color=RED, width=2.5), name="High-vol regime",
            fill="tozeroy", fillcolor="rgba(185,64,64,0.12)",
            hovertemplate="Return: %{x:.3%}<extra></extra>",
        ))
    fig_rr.add_vline(x=0, line_dash="dot", line_color=GRAY, line_width=1)
    fig_rr.update_layout(
        **layout(margin=dict(l=8, r=8, t=20, b=8), height=260),
        xaxis=dict(**ax("Daily return"), tickformat=".1%"),
        yaxis=dict(**ax("Density")),
    )
    st.plotly_chart(fig_rr, use_container_width=True)

st.markdown(f"""<div class="fig-caption">
  <b>Figure 12.</b> 20-day rolling portfolio volatility with regime classification.
  Green = low-vol regime (rolling vol below {vol_threshold:.1%} threshold);
  red = high-vol regime. The distribution comparison shows the return profile is wider
  and heavier-tailed in high-vol periods, confirming asymmetric downside risk.
  The current regime is <strong>{current_regime.lower()}-vol</strong>
  (20-day realized vol: {current_vol:.1%}).
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# APPENDIX
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header" style="margin-top:3.5rem;">Appendix: Methodology and terminology</div>',
            unsafe_allow_html=True)
st.markdown("""<div class="explainer-body">
  This appendix explains every metric in plain language, without assuming a background
  in finance or statistics.
</div>""", unsafe_allow_html=True)

appendix_groups = [
    ("Return metrics", [
        ("Annualized return",
         "The compound percentage gain on a $1 investment over the year. Uses multiplicative "
         "compounding rather than simple daily-average × 252, which overstates returns in volatile markets."),
        ("Excess return",
         "Portfolio return minus S&P 500 return over the same period. Positive = outperformance. "
         "Distinct from formal alpha, which also adjusts for the amount of market risk taken."),
        ("Cumulative return index",
         "A growth-of-$1 chart showing day-by-day compounded performance vs. the benchmark."),
    ]),
    ("Risk metrics", [
        ("Annualized volatility",
         "Standard deviation of daily returns × √252. A volatility of 18% implies typical "
         "daily swings of roughly ±1.1%. It penalizes both upside and downside equally."),
        ("Sharpe ratio",
         "Excess return above the risk-free rate divided by total volatility. Above 1.0 is "
         "considered good; above 2.0 exceptional. Risk-free rate: 2% annualized (short-term Treasury)."),
        ("Sortino ratio",
         "Like Sharpe, but uses only downside deviation (returns below the risk-free rate). "
         "A Sortino materially above Sharpe implies the portfolio's volatility is mostly upside."),
        ("Calmar ratio",
         "Annualized return divided by the absolute maximum drawdown. A Calmar ≥ 2 is strong; "
         "it answers how much return the portfolio earned per point of worst-case loss."),
        ("Value at Risk (VaR)",
         "Maximum expected 1-day loss at a given confidence level. Historical VaR uses the "
         "empirical distribution; parametric VaR assumes Gaussian returns."),
        ("Expected Shortfall (CVaR)",
         "The average loss on the worst tail days — i.e., the mean return below the VaR threshold. "
         "Unlike VaR, CVaR quantifies severity, not just frequency, of tail losses. "
         "It is the preferred risk measure under Basel III/IV and FRTB."),
        ("Maximum drawdown",
         "Largest peak-to-trough percentage decline. Measures worst-case loss for an investor "
         "who bought at the peak and sold at the trough."),
        ("Drawdown recovery",
         "Date the portfolio returned to its prior peak. Fast recovery = resilience; "
         "incomplete recovery may indicate structural impairment."),
    ]),
    ("Factor analysis", [
        ("Market beta",
         "Sensitivity of the portfolio to the S&P 500. Beta = 1.0 means lockstep; "
         "β > 1 amplifies moves; β < 1 is more defensive."),
        ("CAPM alpha",
         "Return unexplained by market exposure alone. Positive alpha = manager added value "
         "beyond what the portfolio's beta level would predict."),
        ("Fama-French three-factor model",
         "Extends CAPM with size (SMB) and value (HML) factors. Developed by Fama & French (1993) "
         "and the standard academic benchmark for portfolio attribution."),
        ("SMB (Small Minus Big)",
         "Return of small-cap stocks minus large-cap stocks. Negative SMB loading = large-cap tilt."),
        ("HML (High Minus Low)",
         "Return of value stocks minus growth stocks. Negative HML loading = growth tilt, "
         "typical of large-cap tech-heavy portfolios."),
        ("FF alpha",
         "Return unexplained after controlling for market, size, and value factors. "
         "A more stringent test than CAPM alpha; positive and significant FF alpha "
         "suggests genuine skill or structural edge."),
        ("t-statistic",
         "Statistical confidence measure. |t| > 1.96 = significant at the 5% level. "
         "Loadings with low t-stats should be interpreted cautiously."),
    ]),
    ("Portfolio optimization", [
        ("Efficient frontier",
         "The set of portfolios offering the highest expected return for each level of risk, "
         "derived from the sample return and covariance matrix via constrained quadratic programming."),
        ("Minimum-variance portfolio",
         "The portfolio with the lowest achievable volatility within the asset universe, "
         "subject to long-only constraints. Achieved by minimizing w'Σw."),
        ("Maximum-Sharpe portfolio",
         "The portfolio on the efficient frontier tangent to the Capital Market Line from "
         "the risk-free rate. Maximizes (E[r] − rf) / σ subject to long-only weights."),
        ("Risk parity",
         "Allocates capital such that each asset contributes equally to total portfolio variance. "
         "Inherently underweights high-volatility assets relative to equal-weight. "
         "Solved by minimizing the squared deviation of each asset's risk share from 1/N."),
        ("Capital Market Line (CML)",
         "The line from the risk-free rate through the max-Sharpe portfolio. Any point on the "
         "CML is achievable by combining the max-Sharpe portfolio with the risk-free asset."),
    ]),
    ("Regime detection", [
        ("Volatility regime",
         "A classification of market conditions into low-vol and high-vol states based on "
         "20-day realized volatility relative to the full-period average. Risk models, "
         "position sizing, and VaR estimates should be conditioned on the current regime."),
        ("Regime threshold",
         "The full-period annualized portfolio volatility, used as the boundary between "
         "low and high-vol regimes. A more rigorous approach (e.g., Gaussian HMM) would "
         "estimate the threshold via maximum likelihood."),
    ]),
    ("Portfolio construction", [
        ("Equal-weight portfolio",
         "5% to each of 20 positions. Simple, transparent, no concentration — but ignores "
         "risk differences across stocks."),
        ("Variance contribution",
         "How much of total portfolio variance comes from each position. A stock contributing "
         "11% of variance on a 5% capital weight is punching 2.2× above its weight on risk."),
        ("Correlation matrix",
         "Pairwise return co-movement. Correlations near +1 reduce diversification; "
         "near 0 or negative correlations provide meaningful diversification benefit."),
        ("Diversification benefit",
         "The gap between the weighted-average individual volatility and the portfolio volatility. "
         "Larger gap = more diversification. High pairwise correlations shrink this gap."),
        ("Risk-free rate",
         "2% annualized, representing a short-term US Treasury yield. Used as the baseline "
         "for Sharpe ratio and CAPM alpha calculations."),
    ]),
]

for group_title, terms in appendix_groups:
    st.markdown(f"""<div class="appendix-group">
      <div class="appendix-group-title">{group_title}</div>
      {''.join(f'<div class="appendix-term"><b>{term}.</b> {defn}</div>' for term, defn in terms)}
    </div>""", unsafe_allow_html=True)

st.markdown(f"""<div class="paper-footer">
  Data: Yahoo Finance daily adjusted closing prices, {start_date.strftime('%Y-%m-%d')} to
  {end_date.strftime('%Y-%m-%d')}. Extended S&P 500 history for stress testing from 2000-01-01.
  Fama-French factors: Kenneth French Data Library (mba.tuck.dartmouth.edu/pages/faculty/ken.french).
  Risk-free rate: 2.00% annualized (constant). Annualized figures use 252 trading-day convention.
  Portfolio optimization via scipy SLSQP (long-only constraints). Stress test returns are
  model-implied (β × R<sub>market</sub> + α-adjustment), not realized.
  Regime classification uses 20-day rolling volatility vs. full-period threshold.
  This dashboard is for analytical and educational purposes only and does not constitute investment advice.
</div>""", unsafe_allow_html=True)
