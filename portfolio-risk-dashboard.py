import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
import io

st.set_page_config(
    page_title="Equity Portfolio Risk Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;0,600;1,400&family=DM+Sans:wght@300;400;500;600&display=swap');

  .block-container { padding-top: 3.5rem; padding-bottom: 4rem; max-width: 1200px; }
  [data-testid="stSidebar"] { display: none; }
  [data-testid="collapsedControl"] { display: none; }

  .paper-title {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 31px; font-weight: 500; line-height: 1.25;
    color: var(--text-color); margin-bottom: 0.4rem; letter-spacing: -0.01em;
  }
  .paper-byline {
    font-family: 'DM Sans', sans-serif;
    font-size: 13px; color: var(--text-color); opacity: 0.5;
    margin-bottom: 1.5rem; letter-spacing: 0.01em;
  }
  .abstract-box {
    border-top: 1px solid rgba(128,128,128,0.25);
    border-bottom: 1px solid rgba(128,128,128,0.25);
    padding: 1.2rem 0; margin-bottom: 2rem;
  }
  .abstract-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 10px; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; opacity: 0.4; color: var(--text-color); margin-bottom: 7px;
  }
  .abstract-text {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 15.5px; line-height: 1.8; color: var(--text-color); max-width: 900px;
  }
  .sec-header {
    font-family: 'DM Sans', sans-serif;
    font-size: 10px; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--text-color); opacity: 0.4;
    margin: 2.5rem 0 0.75rem; padding-bottom: 5px;
    border-bottom: 1px solid rgba(128,128,128,0.15);
  }
  .kpi-card {
    border: 1px solid rgba(128,128,128,0.15);
    border-radius: 3px; padding: 1rem 1.1rem;
    background: rgba(128,128,128,0.03); margin-bottom: 0.5rem;
  }
  .kpi-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 9px; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; opacity: 0.45; color: var(--text-color); margin-bottom: 6px;
  }
  .kpi-value {
    font-family: 'DM Sans', sans-serif;
    font-size: 26px; font-weight: 500; line-height: 1.1; color: var(--text-color);
  }
  .kpi-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 12px; margin-top: 4px; opacity: 0.55; color: var(--text-color);
  }
  .pos  { color: #2e7d4f !important; font-weight: 600; }
  .neg  { color: #b94040 !important; font-weight: 600; }
  .neut { color: #1a4f82 !important; font-weight: 600; }
  .warn { color: #c47a00 !important; font-weight: 600; }

  .fig-caption {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 13.5px; line-height: 1.75; color: var(--text-color);
    opacity: 0.8; margin-top: 0.1rem; margin-bottom: 1.5rem; font-style: italic;
  }
  .fig-caption b { font-style: normal; font-weight: 600; color: var(--text-color); }

  .explainer-body {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 15px; line-height: 1.8; color: var(--text-color); opacity: 0.85;
    margin-bottom: 0.75rem;
  }
  .appendix-group {
    margin-bottom: 2rem;
  }
  .appendix-group-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 10px; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--text-color); opacity: 0.5;
    margin-bottom: 0.6rem; padding-bottom: 4px;
    border-bottom: 1px solid rgba(128,128,128,0.12);
  }
  .appendix-term {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 15px; line-height: 1.85; color: var(--text-color);
    margin-bottom: 0.7rem;
  }
  .appendix-term b {
    font-weight: 600; font-style: normal;
  }
  .paper-footer {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 12px; color: var(--text-color); opacity: 0.35;
    margin-top: 4rem; padding-top: 1rem;
    border-top: 1px solid rgba(128,128,128,0.15); line-height: 1.7;
  }
  label, .stSlider label, .stNumberInput label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 11px !important; font-weight: 500 !important;
    letter-spacing: 0.02em; opacity: 0.7;
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

BLUE   = "#1a4f82"; LBLUE  = "#3d7ab5"; LLBLUE = "#b8cfe0"
GREEN  = "#2e7d4f"; LGREEN = "#6ab06a"
RED    = "#b94040"; LRED   = "#c47a7a"
GOLD   = "#c47a00"; GRAY   = "#888888"

def ax(title, grid=True):
    return dict(
        title=dict(text=title, font=dict(size=12, color="#333333")),
        tickfont=dict(size=11, color="#444444"),
        gridcolor="#e8e0d0" if grid else "rgba(0,0,0,0)",
        linecolor="#d4c9b8", linewidth=1, showline=True,
        showgrid=grid, zeroline=False, ticks="outside", ticklen=3,
    )

def fmt_pct(v, decimals=2): return f"{v:.{decimals}%}"
def fmt_f(v, decimals=2):   return f"{v:.{decimals}f}"
def sgn(v):                 return f"+{v:.2%}" if v >= 0 else f"{v:.2%}"

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
    """Pure-numpy OLS. X should include a constant column. Returns (coeffs, r2)."""
    b  = np.linalg.lstsq(X, y, rcond=None)[0]
    yh = X @ b
    ss_res = ((y - yh)**2).sum()
    ss_tot = ((y - y.mean())**2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return b, r2

# ══════════════════════════════════════════════════════════════════════════
# TICKERS & NAMES
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

# FF3 factor tickers (Ken French factors via yfinance proxies)
# We'll fetch actual FF data from the web as a CSV — Kenneth French's public library
FF3_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)

# ══════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="paper-title">Equity Portfolio Risk Dashboard</div>
<div class="paper-byline">
  Risk attribution, factor exposure, return distribution, and drawdown analysis
  for a 20-stock US equity portfolio
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="abstract-box">
  <div class="abstract-label">Overview</div>
  <div class="abstract-text">
    This dashboard computes and attributes risk across a 20-stock US large-cap equity portfolio,
    benchmarked against the S&P 500. Metrics include annualized return, volatility, Sharpe ratio,
    Sortino ratio, Calmar ratio, Value at Risk under both historical and parametric assumptions,
    market beta, and CAPM alpha. Rolling 60-day windows for volatility, Sharpe, and beta show
    how the portfolio's risk profile has shifted over time. Section 11 performs Fama-French
    three-factor attribution, decomposing excess returns into market, size, and value loadings.
    A plain-language methodology appendix defines every metric for non-technical audiences.
    Weights are adjustable inline; a custom return dataset can be uploaded in CSV format.
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">Data</div>', unsafe_allow_html=True)

end_date   = datetime.today()
start_date = end_date - timedelta(days=365)

@st.cache_data(show_spinner=False)
def load_prices(tickers, start, end):
    raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
    return raw['Close'].dropna()

@st.cache_data(show_spinner=False)
def load_ff3(start, end):
    """Fetch daily Fama-French 3 factors from Ken French's data library."""
    try:
        ff = pd.read_csv(
            FF3_URL, compression="zip", skiprows=3,
            names=["Date","Mkt-RF","SMB","HML","RF"],
            index_col=0, parse_dates=True,
        )
        # Rows after the daily section contain monthly data — drop them
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
      is fetched from Kenneth French's public data library. To use your own return data, upload
      a CSV with dates as the index and one column per asset containing daily returns (not prices).
      The benchmark column is not required in a custom upload.
    </div>""", unsafe_allow_html=True)

with dl_col2:
    uploaded_file = st.file_uploader("Upload returns CSV (optional)", type=["csv"])

with st.spinner("Fetching price data..."):
    all_prices = load_prices(TICKERS + [BENCHMARK], start_date, end_date)

with st.spinner("Fetching Fama-French factors..."):
    ff3 = load_ff3(start_date, end_date)

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
  Each cell is the simple daily return for that asset, calculated as shown below.
  Returns are used directly for all downstream calculations;
  no log-return transformation is applied.
</div>""", unsafe_allow_html=True)
st.latex(r"r_t = \frac{P_t - P_{t-1}}{P_{t-1}}")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 2 — PORTFOLIO WEIGHTS
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">2. Portfolio weights</div>', unsafe_allow_html=True)

st.markdown("""<div class="explainer-body" style="margin-bottom:1rem;">
  Edit the Weight column directly. Weights are normalized to sum to 100% before any
  calculation, so only relative magnitudes matter. The default is equal-weight at 5% per stock.
  The chart on the right updates immediately and shows each position's normalized share of the
  portfolio alongside the equal-weight threshold.
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
    bar_colors = [
        f"rgba({int(26+(180-26)*i/max(n-1,1))},"
        f"{int(79+(207-79)*i/max(n-1,1))},"
        f"{int(130+(224-130)*i/max(n-1,1))},0.85)"
        for i in range(n)
    ]
    fig_alloc = go.Figure(go.Bar(
        x=sorted_weights, y=sorted_names, orientation="h",
        marker_color=bar_colors,
        text=[f"{w:.1f}%" for w in sorted_weights],
        textposition="outside", textfont=dict(size=9, color="#333"),
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    equal_w = round(100/n_assets, 1)
    fig_alloc.add_vline(x=equal_w, line_dash="dot", line_color=GRAY, line_width=1,
                        annotation_text=f"Equal weight ({equal_w:.1f}%)",
                        annotation_font=dict(size=9, color=GRAY),
                        annotation_position="top right")
    fig_alloc.update_layout(
        **{**BASE, "margin": dict(l=8, r=55, t=10, b=8)},
        height=max(320, n_assets*22),
        xaxis=dict(**ax("Normalized weight (%)"), ticksuffix="%"),
        yaxis=dict(tickfont=dict(size=10, color="#444"), showgrid=False,
                   linecolor="#d4c9b8", linewidth=1, autorange="reversed"),
        showlegend=False,
    )
    st.plotly_chart(fig_alloc, use_container_width=True)

st.markdown("""<div class="fig-caption">
  <b>Portfolio allocation.</b> Normalized weight of each position after adjustment.
  The dotted line marks the equal-weight threshold. Positions to the right carry more
  capital than equal-weight; positions to the left carry less.
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# CORE CALCULATIONS
# ══════════════════════════════════════════════════════════════════════════
clean_df = numeric_df.dropna()
if clean_df.shape[0] < 30:
    st.error("Insufficient clean data (fewer than 30 trading days).")
    st.stop()

port_returns = clean_df.dot(weights)
rf_daily     = 0.02 / 252
cov_matrix   = clean_df.cov() * 252
port_vol     = np.sqrt(weights @ cov_matrix.values @ weights)

# Sharpe
excess  = port_returns - rf_daily
sharpe  = (excess.mean() / excess.std()) * np.sqrt(252)

# Sortino — downside deviation only (returns below rf_daily)
downside      = excess.copy(); downside[downside > 0] = 0
downside_std  = np.sqrt((downside**2).mean()) * np.sqrt(252)
sortino       = (excess.mean() * 252) / downside_std if downside_std > 0 else np.nan

# VaR
pr_clean    = port_returns.dropna().values
var_hist_95 = np.percentile(pr_clean, 5)
var_hist_99 = np.percentile(pr_clean, 1)
mu_daily    = port_returns.mean()
sig_daily   = port_returns.std()
z95, z99    = 1.6449, 2.3263
var_param_95 = mu_daily - z95*sig_daily
var_param_99 = mu_daily - z99*sig_daily

# Beta / alpha
bench_aligned = bench_returns.reindex(port_returns.index).dropna()
port_aligned  = port_returns.reindex(bench_aligned.index)
cov_pm        = np.cov(port_aligned.values, bench_aligned.values)[0,1]
var_m         = np.var(bench_aligned.values)
beta          = cov_pm / var_m if var_m > 0 else np.nan
alpha_ann     = ((port_aligned.mean()-rf_daily) - beta*(bench_aligned.mean()-rf_daily)) * 252

bench_vol     = bench_aligned.std() * np.sqrt(252)
bench_sharpe  = ((bench_aligned.mean()-rf_daily)/bench_aligned.std()) * np.sqrt(252)
bench_ret_ann = bench_aligned.mean() * 252

# Rolling metrics
roll_win    = 60
roll_vol    = port_returns.rolling(roll_win).std() * np.sqrt(252)
roll_sharpe = (port_returns.rolling(roll_win).mean() /
               port_returns.rolling(roll_win).std()) * np.sqrt(252)
roll_beta   = (port_aligned.rolling(roll_win).cov(bench_aligned) /
               bench_aligned.rolling(roll_win).var())

# Cumulative / drawdown
cum_port        = (1 + port_returns).cumprod()
cum_bench       = (1 + bench_aligned).cumprod()
total_ret       = cum_port.iloc[-1] - 1
bench_total_ret = cum_bench.iloc[-1] - 1

rolling_max    = cum_port.cummax()
drawdown       = (cum_port - rolling_max) / rolling_max
max_dd         = drawdown.min()
trough_date    = drawdown.idxmin()
pre_trough     = cum_port.loc[:trough_date]
peak_date      = pre_trough.idxmax()
post_trough_dd = drawdown.loc[trough_date:]
recovered      = post_trough_dd[post_trough_dd >= 0]
recovery_date  = recovered.index[0] if len(recovered) > 0 else None

# Calmar ratio = annualized return / abs(max drawdown)
calmar = (total_ret / abs(max_dd)) if max_dd != 0 else np.nan

# ══════════════════════════════════════════════════════════════════════════
# PORTFOLIO SUMMARY
# ══════════════════════════════════════════════════════════════════════════
port_ret_ann = port_returns.mean() * 252
excess_ret   = total_ret - bench_total_ret
alpha_str    = sgn(alpha_ann)
sharpe_diff  = sharpe - bench_sharpe
vol_diff     = port_vol - bench_vol

if is_equal_weight:
    weight_desc = f"At equal weight across {n_assets} positions"
else:
    top3_idx    = np.argsort(weights)[::-1][:3]
    top3_names  = [NAMES.get(tickers_in[i], tickers_in[i]) for i in top3_idx]
    top3_wts    = [weights[i]*100 for i in top3_idx]
    weight_desc = (
        f"With a tilted allocation (largest positions: "
        f"{top3_names[0]} {top3_wts[0]:.1f}%, "
        f"{top3_names[1]} {top3_wts[1]:.1f}%, "
        f"{top3_names[2]} {top3_wts[2]:.1f}%)"
    )

if sharpe_diff > 0.1:
    sharpe_prose = (f"The Sharpe ratio of {sharpe:.2f} exceeds the S&P 500's "
                    f"{bench_sharpe:.2f} by {sharpe_diff:.2f}, indicating materially "
                    f"better risk-adjusted returns.")
elif sharpe_diff > 0:
    sharpe_prose = (f"The Sharpe ratio of {sharpe:.2f} modestly exceeds "
                    f"the S&P 500's {bench_sharpe:.2f}.")
else:
    sharpe_prose = (f"The Sharpe ratio of {sharpe:.2f} trails "
                    f"the S&P 500's {bench_sharpe:.2f}.")

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
    Calmar ratio of {calmar:.2f} expresses the return earned per unit of maximum drawdown risk.
    With a market beta of {beta:.2f}, the portfolio {beta_prose}.
    CAPM alpha of {alpha_str} annualized represents residual return beyond what market exposure
    alone predicts. The maximum drawdown of {max_dd:.2%} reached its trough in
    {trough_date.strftime('%B %Y')}.
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 3 — KPI CARDS (3 rows × 3 cols)
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">3. Portfolio metrics</div>', unsafe_allow_html=True)

kpi_rows = [
    [
        ("Annualized return",    fmt_pct(total_ret),
         f"S&P 500: {fmt_pct(bench_total_ret)}",
         "pos" if total_ret > bench_total_ret else "neg"),
        ("Annualized volatility", fmt_pct(port_vol),
         f"S&P 500: {fmt_pct(bench_vol)}",
         "warn" if port_vol > bench_vol*1.1 else "neut"),
        ("Max drawdown",         fmt_pct(max_dd),
         f"Trough: {trough_date.strftime('%b %Y')}",
         "neg" if abs(max_dd) > 0.1 else "warn"),
    ],
    [
        ("Sharpe ratio",         fmt_f(sharpe),
         f"S&P 500: {fmt_f(bench_sharpe)}",
         "pos" if sharpe > bench_sharpe else "neut"),
        ("Sortino ratio",        fmt_f(sortino),
         "Downside deviation only",
         "pos" if sortino > 1.0 else "neut"),
        ("Calmar ratio",         fmt_f(calmar),
         "Return / |max drawdown|",
         "pos" if calmar > 1.0 else "neut"),
    ],
    [
        ("Historical VaR (95%)", fmt_pct(abs(var_hist_95)),
         f"99%: {fmt_pct(abs(var_hist_99))}",
         "neg" if abs(var_hist_95) > 0.02 else "neut"),
        ("Parametric VaR (95%)", fmt_pct(abs(var_param_95)),
         f"Gaussian, 99%: {fmt_pct(abs(var_param_99))}", "neut"),
        ("CAPM alpha (ann.)",    alpha_str,
         f"Beta: {beta:.2f} vs. S&P 500",
         "pos" if alpha_ann > 0 else "neg"),
    ],
]

for row in kpi_rows:
    cols = st.columns(3)
    for col, (label, value, sub, cls) in zip(cols, row):
        with col:
            st.markdown(f"""<div class="kpi-card">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value {cls}">{value}</div>
              <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

weight_note = "equal-weight" if is_equal_weight else "current-weight"
st.markdown(f"""<div class="fig-caption" style="margin-top:0.75rem;">
  <b>Table 2.</b> Summary risk and return metrics for the {weight_note} portfolio.
  Annualized return is the compound 1-year return.
  Sharpe uses total volatility; Sortino uses only downside deviation (returns below the
  risk-free rate); Calmar uses the maximum drawdown as the risk denominator.
  Historical VaR uses the empirical 5th percentile; parametric VaR assumes Gaussian returns.
  Beta is estimated by OLS on daily returns; alpha is the annualized regression intercept
  net of the risk-free rate.
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
    **{**BASE, "margin": dict(l=8, r=60, t=20, b=8)}, height=420,
    xaxis=dict(tickfont=dict(size=10, color="#444"), showgrid=False,
               linecolor="#d4c9b8", linewidth=1),
    yaxis=dict(tickfont=dict(size=10, color="#444"), showgrid=False,
               linecolor="#d4c9b8", linewidth=1, autorange="reversed"),
)
st.plotly_chart(fig_corr, use_container_width=True)
st.markdown(f"""<div class="fig-caption">
  <b>Figure 1.</b> Pearson correlation matrix across all {n_assets} constituents.
  Blue indicates positive co-movement; red indicates near-zero or negative correlation.
  Cells are not annotated at 20x20 — hover for exact values.
  High pairwise correlations reduce diversification benefit.
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
  <b>Table 3.</b> Fifteen highest pairwise return correlations by absolute value.
  The Mastercard/Visa pair reflects near-identical business models and macro sensitivities.
  High within-sector correlations are the primary constraint on diversification.
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
  with Gaussian KDE overlay (Silverman bandwidth). Daily mean {port_returns.mean():.3%},
  standard deviation {port_returns.std():.3%}. Skewness {skew_val:.3f} ({skew_desc}).
  {kurt_note} The dashed red line marks the empirical 95% VaR.
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
  The portfolio returned {total_ret:.2%} against {bench_total_ret:.2%} for the benchmark,
  an outperformance of {sgn(total_ret-bench_total_ret)}.
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 7 — DRAWDOWN ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">7. Drawdown analysis</div>', unsafe_allow_html=True)

fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(
    x=drawdown.index, y=drawdown.values*100, mode="lines", fill="tozeroy",
    fillcolor="rgba(185,64,64,0.18)", line=dict(color=RED, width=1.5),
    hovertemplate="Date: %{x|%Y-%m-%d}<br>Drawdown: %{y:.2f}%<extra></extra>",
))
fig_dd.add_hline(y=max_dd*100, line_dash="dot", line_color=RED, line_width=1,
                 annotation_text=f"Max DD {max_dd:.2%}",
                 annotation_font=dict(size=9, color=RED),
                 annotation_position="bottom right")
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
  <b>Figure 4.</b> Portfolio drawdown as a percentage decline from the rolling peak.
  The maximum drawdown of {max_dd:.2%} reached its trough on
  {trough_date.strftime('%B %d, %Y')}, following a peak on {peak_date.strftime('%B %d, %Y')}.
  {recovery_prose} The formula is shown below.
</div>""", unsafe_allow_html=True)
st.latex(r"D_t = \frac{V_t - \max_{s \leq t} V_s}{\max_{s \leq t} V_s}")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 8 — ROLLING RISK METRICS
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">8. Rolling risk metrics (60-day window)</div>',
            unsafe_allow_html=True)
st.markdown("""<div class="explainer-body">
  Rolling metrics reveal how the portfolio's risk profile has shifted over time.
  A rising rolling volatility during a drawdown confirms risk escalation;
  a declining rolling Sharpe confirms deteriorating risk-adjusted returns.
  Rolling beta identifies when the portfolio became more or less sensitive to the market.
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
        **{**BASE, "margin": dict(l=60, r=8, t=20, b=8)}, height=280,
        xaxis=dict(**ax("Date")),
        yaxis=dict(**ax("Annualized volatility (%)"), ticksuffix="%"),
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
                     annotation_text="Sharpe = 1", annotation_font=dict(size=9, color=GRAY))
    fig_rs.update_layout(
        **{**BASE, "margin": dict(l=60, r=8, t=20, b=8)}, height=280,
        xaxis=dict(**ax("Date")),
        yaxis=dict(**ax("Sharpe ratio (annualized)")), showlegend=False,
    )
    st.plotly_chart(fig_rs, use_container_width=True)
    st.markdown("""<div class="fig-caption">
      <b>Figure 6.</b> 60-day rolling annualized Sharpe ratio. Periods below zero indicate
      negative risk-adjusted returns. Sharpe = 1 is a conventional adequacy threshold.
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
    **{**BASE, "margin": dict(l=70, r=20, t=20, b=8)}, height=280,
    xaxis=dict(**ax("Date")),
    yaxis=dict(**ax("Beta vs. S&P 500")), showlegend=False,
)
st.plotly_chart(fig_rb, use_container_width=True)
st.markdown(f"""<div class="fig-caption">
  <b>Figure 7.</b> 60-day rolling market beta. Full-period beta of {beta:.2f} shown for
  reference. Excursions above 1.3 or below 0.7 indicate periods of materially elevated or
  reduced market sensitivity.
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 9 — VaR COMPARISON
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">9. Value at Risk: historical vs. parametric</div>',
            unsafe_allow_html=True)

var_data = {
    "Method": ["Historical (empirical)", "Parametric (Gaussian)"],
    "VaR 95%": [f"{abs(var_hist_95):.2%}", f"{abs(var_param_95):.2%}"],
    "VaR 99%": [f"{abs(var_hist_99):.2%}", f"{abs(var_param_99):.2%}"],
    "Assumption": [
        "No distributional assumption; uses empirical return percentiles",
        "Returns are normally distributed with sample mean and std dev",
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
st.markdown(f"""<div class="fig-caption">
  <b>Table 4.</b> Historical and parametric VaR at 95% and 99% confidence levels.
  z(0.95) = 1.645, z(0.99) = 2.326. {tail_note}
</div>""", unsafe_allow_html=True)
st.latex(r"\text{VaR}_\alpha = -(\mu - z_\alpha \, \sigma)")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 10 — INDIVIDUAL ASSET RISK CONTRIBUTION
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">10. Individual asset risk contribution</div>',
            unsafe_allow_html=True)

asset_vols    = clean_df.std() * np.sqrt(252)
marginal_rc   = cov_matrix.values @ weights
component_rc  = marginal_rc * weights
total_risk_sq = weights @ cov_matrix.values @ weights
pct_rc        = component_rc / total_risk_sq * 100

risk_df = pd.DataFrame({
    "Asset":                   [NAMES.get(t,t) for t in tickers_in],
    "Ticker":                  tickers_in,
    "Weight":                  [f"{w:.1%}" for w in weights],
    "Volatility (ann.)":       [f"{v:.2%}" for v in asset_vols],
    "% Variance contribution": [f"{p:.1f}%" for p in pct_rc],
}).sort_values("% Variance contribution", ascending=False,
               key=lambda x: x.str.rstrip('%').astype(float))
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
    **{**BASE, "margin": dict(l=8, r=120, t=20, b=8)}, height=320,
    xaxis={**ax("Asset"), "tickangle": -35},
    yaxis=dict(**ax("% contribution to portfolio variance"), ticksuffix="%"),
    showlegend=False,
)
st.plotly_chart(fig_rc, use_container_width=True)
top_rc = tickers_in[np.argmax(pct_rc)]
st.markdown(f"""<div class="fig-caption">
  <b>Figure 8.</b> Component variance contribution per position. Darker blue exceeds the
  equal-contribution threshold; lighter blue is below it. {NAMES.get(top_rc,top_rc)} is the
  largest single contributor. The formula is shown below.
</div>""", unsafe_allow_html=True)
st.latex(r"RC_i = \frac{w_i \, (\Sigma \mathbf{w})_i}{\mathbf{w}^T \Sigma \mathbf{w}}")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 11 — FAMA-FRENCH 3-FACTOR ATTRIBUTION
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">11. Fama-French three-factor attribution</div>',
            unsafe_allow_html=True)

st.markdown("""<div class="explainer-body">
  The Fama-French three-factor model extends the CAPM by adding size (SMB) and value (HML)
  factors. SMB (Small Minus Big) captures the tendency of small-cap stocks to outperform
  large-caps over time. HML (High Minus Low) captures the tendency of value stocks
  (high book-to-market) to outperform growth stocks. A positive SMB loading indicates
  the portfolio tilts toward smaller companies; a positive HML loading indicates a value tilt.
  Most large-cap growth portfolios have negative HML loadings.
</div>""", unsafe_allow_html=True)

if ff3 is not None and not ff3.empty:
    # Align portfolio excess returns with FF factors
    port_excess_daily = port_aligned - bench_aligned.reindex(port_aligned.index).fillna(0)*0 - rf_daily
    ff_aligned = ff3[["Mkt-RF","SMB","HML"]].reindex(port_aligned.index).dropna()
    port_ff    = port_aligned.reindex(ff_aligned.index) - rf_daily

    # OLS: r_p - rf = alpha + b1*MktRF + b2*SMB + b3*HML + e
    Y = port_ff.values
    X = np.column_stack([
        np.ones(len(ff_aligned)),
        ff_aligned["Mkt-RF"].values,
        ff_aligned["SMB"].values,
        ff_aligned["HML"].values,
    ])
    ff_coefs, ff_r2 = ols(Y, X)
    ff_alpha   = ff_coefs[0] * 252
    ff_mkt     = ff_coefs[1]
    ff_smb     = ff_coefs[2]
    ff_hml     = ff_coefs[3]

    # t-stats (approximate, assuming homoskedastic)
    resid   = Y - X @ ff_coefs
    sig2    = (resid**2).sum() / (len(Y) - 4)
    se      = np.sqrt(np.diag(sig2 * np.linalg.inv(X.T @ X)))
    t_stats = ff_coefs / se
    t_alpha, t_mkt, t_smb, t_hml = t_stats

    # KPI cards for factor loadings
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

    # Factor loading bar chart
    factor_names  = ["Market (Mkt-RF)", "Size (SMB)", "Value (HML)"]
    factor_vals   = [ff_mkt, ff_smb, ff_hml]
    factor_colors = [LBLUE if v >= 0 else LRED for v in factor_vals]

    fig_ff = go.Figure(go.Bar(
        x=factor_names, y=factor_vals,
        marker_color=factor_colors, opacity=0.85,
        text=[f"{v:.3f}" for v in factor_vals],
        textposition="outside", textfont=dict(size=12, color="#333"),
        hovertemplate="%{x}: %{y:.4f}<extra></extra>",
    ))
    fig_ff.add_hline(y=0, line_color="#d4c9b8", line_width=1)
    fig_ff.update_layout(
        **{**BASE, "margin": dict(l=60, r=20, t=30, b=8)}, height=300,
        xaxis=dict(**ax("Factor", grid=False)),
        yaxis={**ax("Factor loading (beta)"),
               "zeroline": True, "zerolinecolor": "#d4c9b8", "zerolinewidth": 1},
        showlegend=False,
    )
    st.plotly_chart(fig_ff, use_container_width=True)

    # Rolling 60-day FF alpha
    roll_ff_alpha = pd.Series(index=port_aligned.index, dtype=float)
    for end_i in range(roll_win, len(port_aligned)):
        idx_slice  = port_aligned.index[end_i-roll_win:end_i]
        pf_sl      = port_aligned.loc[idx_slice] - rf_daily
        ff_sl      = ff3[["Mkt-RF","SMB","HML"]].reindex(idx_slice).dropna()
        if len(ff_sl) < 20: continue
        pf_sl      = pf_sl.reindex(ff_sl.index)
        Xw = np.column_stack([np.ones(len(ff_sl)),
                              ff_sl["Mkt-RF"].values,
                              ff_sl["SMB"].values,
                              ff_sl["HML"].values])
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
            **{**BASE, "margin": dict(l=70, r=20, t=20, b=8)}, height=260,
            xaxis=dict(**ax("Date")),
            yaxis=dict(**ax("Annualized FF alpha (%)"), ticksuffix="%"),
            showlegend=False,
        )
        st.plotly_chart(fig_rfa, use_container_width=True)

    # Interpret SMB/HML
    smb_interp = ("small-cap tilt" if ff_smb > 0.1
                  else "large-cap tilt" if ff_smb < -0.1
                  else "size-neutral")
    hml_interp = ("value tilt" if ff_hml > 0.1
                  else "growth tilt" if ff_hml < -0.1
                  else "style-neutral")

    st.markdown(f"""<div class="fig-caption">
      <b>Figure 9.</b> Fama-French three-factor loadings estimated by OLS on daily excess returns.
      The model explains {ff_r2:.1%} of portfolio return variance (R²).
      The market loading of {ff_mkt:.3f} (t = {t_mkt:.2f}) is the primary driver — consistent
      with a large-cap equity portfolio. SMB loading of {ff_smb:.3f} (t = {t_smb:.2f})
      indicates a {smb_interp}; HML loading of {ff_hml:.3f} (t = {t_hml:.2f}) indicates a
      {hml_interp}. FF alpha of {sgn(ff_alpha)} annualized (t = {t_alpha:.2f}) represents
      return unexplained by all three factors. The rolling 60-day alpha chart shows how this
      residual has varied over time.
    </div>""", unsafe_allow_html=True)
    st.latex(
        r"r_p - r_f = \alpha + \beta_{\text{Mkt}}(r_m - r_f) + \beta_{\text{SMB}} \cdot "
        r"\text{SMB} + \beta_{\text{HML}} \cdot \text{HML} + \varepsilon"
    )
    st.markdown(f"""<div class="fig-caption" style="margin-top:0.5rem;">
      <b>Table 5.</b> Full-period Fama-French regression results.
      Data: Kenneth French Data Library daily factors, aligned to the portfolio's return dates.
      |t| > 1.96 is conventionally significant at the 5% level.
    </div>""", unsafe_allow_html=True)
    ff_results_df = pd.DataFrame({
        "Factor":  ["Alpha (ann.)", "Mkt-RF", "SMB", "HML"],
        "Loading": [f"{sgn(ff_alpha)}", f"{ff_mkt:.4f}", f"{ff_smb:.4f}", f"{ff_hml:.4f}"],
        "t-stat":  [f"{t_alpha:.2f}", f"{t_mkt:.2f}", f"{t_smb:.2f}", f"{t_hml:.2f}"],
        "Significant (5%)": [
            "Yes" if abs(t_alpha) > 1.96 else "No",
            "Yes" if abs(t_mkt)   > 1.96 else "No",
            "Yes" if abs(t_smb)   > 1.96 else "No",
            "Yes" if abs(t_hml)   > 1.96 else "No",
        ],
    })
    st.dataframe(ff_results_df, use_container_width=True, hide_index=True)
else:
    st.info("Fama-French factor data could not be fetched. "
            "Check your network connection or try again — the Kenneth French data library "
            "is occasionally unavailable. All other sections remain fully functional.")

# ══════════════════════════════════════════════════════════════════════════
# APPENDIX — METHODOLOGY & TERMINOLOGY
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header" style="margin-top:3.5rem;">Appendix: Methodology and terminology</div>',
            unsafe_allow_html=True)

st.markdown("""<div class="explainer-body">
  This appendix explains every metric used in the dashboard in plain language,
  without assuming a background in finance or statistics.
  Each definition describes what the number means and why it matters.
</div>""", unsafe_allow_html=True)

appendix_groups = [
    ("Return metrics", [
        ("Annualized return",
         "The percentage gain on a $1 investment over the year. A return of 26% means $1 "
         "grew to $1.26. This dashboard uses the compound (multiplicative) return rather than "
         "simply multiplying the daily average by 252, which would slightly overstate returns "
         "in volatile markets."),
        ("Excess return",
         "The portfolio's return minus the S&P 500's return over the same period. Positive "
         "excess return means the portfolio beat its benchmark; negative means it lagged. This "
         "is also called active return or alpha in casual usage, though formal alpha (see below) "
         "adjusts for the amount of market risk taken."),
        ("Cumulative return index",
         "A growth-of-$1 chart that shows how $1 invested at the start would have grown, "
         "day by day, through the end of the period. It is the most intuitive way to compare "
         "portfolio performance to a benchmark over time."),
    ]),
    ("Risk metrics", [
        ("Annualized volatility",
         "A measure of how much the portfolio's daily returns fluctuate, scaled to an annual "
         "figure. Higher volatility means bigger swings — both up and down. It is computed as "
         "the standard deviation of daily returns multiplied by the square root of 252 (the "
         "number of trading days in a year). A volatility of 18% means daily moves of roughly "
         "plus or minus 1.1% are typical."),
        ("Sharpe ratio",
         "Return earned per unit of total risk taken, expressed as a ratio. It is calculated "
         "as the portfolio's excess return above the risk-free rate divided by its volatility. "
         "A Sharpe ratio above 1.0 is generally considered good; above 2.0 is exceptional. "
         "The risk-free rate used here is 2% per year, representing a short-term Treasury "
         "yield benchmark."),
        ("Sortino ratio",
         "Similar to the Sharpe ratio, but it only penalizes downside volatility — movements "
         "below the risk-free rate — rather than all volatility. The logic is that upward "
         "swings are not actually risky from an investor's perspective. A higher Sortino ratio "
         "than Sharpe ratio suggests the portfolio's volatility is mostly on the upside."),
        ("Calmar ratio",
         "Annual return divided by the absolute maximum drawdown. It answers: how much return "
         "did the portfolio earn for every percentage point of its worst decline? A Calmar ratio "
         "of 2.0 means the portfolio earned 2% of annual return for every 1% it fell at its "
         "worst. It is most useful for comparing strategies with very different drawdown profiles."),
        ("Value at Risk (VaR)",
         "The maximum expected loss on a single day, at a given confidence level. A 95% "
         "historical VaR of 1.3% means that on 95 out of 100 trading days, the portfolio "
         "lost less than 1.3%. On the remaining 5 days — the tail — losses were larger. VaR "
         "does not predict how large those tail losses are, only that they exceeded the threshold."),
        ("Historical vs. parametric VaR",
         "Historical VaR uses the actual distribution of past returns to find the cutoff — "
         "no assumptions required. Parametric VaR assumes returns follow a normal (bell-curve) "
         "distribution and calculates the cutoff mathematically. If they differ significantly, "
         "the empirical distribution has meaningfully fatter or thinner tails than Gaussian."),
        ("Maximum drawdown",
         "The largest peak-to-trough percentage decline in portfolio value over the analysis "
         "period. A drawdown of -13% means the portfolio fell 13% from its highest point before "
         "recovering. It measures the worst-case loss an investor would have experienced if they "
         "bought at the peak and sold at the trough."),
        ("Drawdown recovery",
         "The date on which the portfolio returned to its prior peak value after the maximum "
         "drawdown. A fast recovery indicates resilience; a slow or incomplete recovery may "
         "indicate structural damage to the portfolio's return-generating ability."),
    ]),
    ("Factor analysis", [
        ("Market beta",
         "A measure of how much the portfolio moves relative to the overall stock market. "
         "A beta of 1.0 means the portfolio moves in lockstep with the S&P 500. A beta of "
         "1.2 means it tends to rise 1.2% when the market rises 1%, and fall 1.2% when the "
         "market falls 1%. A beta below 1.0 indicates a more defensive portfolio that "
         "amplifies market moves less."),
        ("CAPM alpha",
         "The portion of portfolio return that cannot be explained by its market exposure. "
         "After accounting for the fact that a higher-beta portfolio should earn higher returns "
         "just by being riskier, alpha measures what the manager added — or lost — through "
         "stock selection and timing. Positive alpha means the portfolio outperformed what its "
         "risk level alone would have predicted."),
        ("Fama-French three-factor model",
         "An extension of the CAPM that recognizes two additional drivers of stock returns "
         "beyond market risk. The model was developed by economists Eugene Fama and Kenneth "
         "French and has become the standard academic benchmark for portfolio performance. "
         "Its three factors are: the market premium (Mkt-RF), the size premium (SMB), "
         "and the value premium (HML)."),
        ("SMB — Small Minus Big",
         "The return difference between small-cap and large-cap stocks on a given day. "
         "A positive SMB loading means the portfolio behaves more like small-cap stocks "
         "(more volatile, historically higher long-run return). A negative loading — typical "
         "of large-cap portfolios — means the portfolio tilts toward large established companies. "
         "SMB captures the historical tendency of smaller firms to outperform over long periods."),
        ("HML — High Minus Low",
         "The return difference between value stocks (cheap relative to book value) and growth "
         "stocks (expensive relative to book value). A positive HML loading indicates a value "
         "tilt; negative indicates a growth tilt. Most large-cap US technology portfolios have "
         "negative HML because they are growth-oriented. HML captures the historical value premium."),
        ("FF alpha",
         "After controlling for all three Fama-French factors — market, size, and value — any "
         "remaining unexplained return is FF alpha. It is a more stringent test than CAPM alpha "
         "because it rules out the possibility that outperformance was simply due to holding "
         "small or cheap stocks. A positive and statistically significant FF alpha suggests "
         "genuine skill or structural advantage."),
        ("t-statistic",
         "A measure of statistical confidence. A t-statistic above 1.96 (or below -1.96) "
         "means the result is statistically significant at the 5% level — in plain terms, "
         "unlikely to have occurred by chance if the true value were zero. Factor loadings "
         "with low t-statistics should be interpreted cautiously, as they may not reflect "
         "a real structural tilt."),
    ]),
    ("Portfolio construction", [
        ("Equal-weight portfolio",
         "A portfolio where every stock receives the same percentage allocation — here, 5% "
         "each across 20 positions. Equal weighting is simple, transparent, and avoids "
         "concentration in any single stock. Its main drawback is that it ignores differences "
         "in risk: a highly volatile stock gets the same capital as a stable one."),
        ("Variance contribution",
         "How much of the portfolio's total risk (measured as variance) comes from each "
         "individual position. A stock can contribute more than its capital weight suggests "
         "if it is highly volatile or highly correlated with other positions. A stock that "
         "contributes 11% of variance but holds only 5% of capital is effectively punching "
         "above its weight on risk."),
        ("Correlation matrix",
         "A table showing how closely each pair of stocks moves together. A correlation of "
         "+1.0 means two stocks always move in the same direction by the same amount. A "
         "correlation of 0 means they move independently. A correlation of -1.0 means they "
         "always move in opposite directions. High correlations within a portfolio reduce the "
         "benefit of diversification."),
        ("Diversification",
         "The benefit of holding multiple assets that do not all move together. When stocks "
         "are imperfectly correlated, losses in one can be partially offset by gains in another. "
         "The portfolio's overall volatility will be lower than the average volatility of its "
         "individual holdings — the gap between them is the diversification benefit."),
        ("Risk-free rate",
         "The return available on a riskless investment, used as the baseline for measuring "
         "whether a risky investment is worth taking. This dashboard uses 2% annualized, "
         "representing a short-term US Treasury yield. The Sharpe ratio measures return "
         "above this baseline per unit of risk; CAPM alpha measures outperformance after "
         "adjusting for beta-driven return above the risk-free rate."),
    ]),
]

for group_title, terms in appendix_groups:
    st.markdown(f"""<div class="appendix-group">
      <div class="appendix-group-title">{group_title}</div>
      {''.join(f'<div class="appendix-term"><b>{term}.</b> {defn}</div>' for term, defn in terms)}
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════
st.markdown(f"""<div class="paper-footer">
  Data: Yahoo Finance daily adjusted closing prices, {start_date.strftime('%Y-%m-%d')} to
  {end_date.strftime('%Y-%m-%d')}. Fama-French factors: Kenneth French Data Library
  (mba.tuck.dartmouth.edu/pages/faculty/ken.french). Risk-free rate: 2.00% annualized (constant).
  Annualized figures use a 252 trading-day convention.
  Beta and FF loadings estimated by OLS on daily excess returns.
  VaR figures are 1-day estimates. Rolling window: {roll_win} trading days.
  This dashboard is for analytical and educational purposes only and does not constitute
  investment advice.
</div>""", unsafe_allow_html=True)
