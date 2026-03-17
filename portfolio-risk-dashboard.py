import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import io

st.set_page_config(
    page_title="Equity Portfolio Risk Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════
# CSS — academic whitepaper style matching LBO framework
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
    background: rgba(128,128,128,0.03);
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
  .param-table {
    width: 100%; border-collapse: collapse;
    font-family: 'DM Sans', sans-serif; font-size: 12px;
    margin-bottom: 0.5rem;
  }
  .param-table th {
    font-size: 9px; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; opacity: 0.4; color: var(--text-color);
    border-bottom: 1px solid rgba(128,128,128,0.2);
    padding: 6px 10px 6px 0; text-align: left;
  }
  .param-table td {
    padding: 5px 10px 5px 0;
    border-bottom: 1px solid rgba(128,128,128,0.07);
    color: var(--text-color); font-size: 12px; vertical-align: top;
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
# CHART CONSTANTS
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
GRAY   = "#888888"

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

# ══════════════════════════════════════════════════════════════════════════
# KDE (no scipy)
# ══════════════════════════════════════════════════════════════════════════
def kde_curve(data, n_pts=300, bw_factor=1.0):
    data = np.asarray(data)
    n, std = len(data), np.std(data)
    if std == 0 or n < 2: return np.array([]), np.array([])
    bw = bw_factor * 1.06 * std * n**(-0.2)
    x  = np.linspace(data.min() - 3*bw, data.max() + 3*bw, n_pts)
    z  = (x[:, None] - data[None, :]) / bw
    density = np.exp(-0.5*z**2).sum(axis=1) / (n*bw*np.sqrt(2*np.pi))
    return x, density

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

# ══════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="paper-title">Equity Portfolio Risk Dashboard</div>
<div class="paper-byline">
  Risk attribution, factor exposure, return distribution, and drawdown analysis for a 20-stock US equity portfolio
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="abstract-box">
  <div class="abstract-label">Overview</div>
  <div class="abstract-text">
    This dashboard computes and attributes risk across a 20-stock US large-cap equity portfolio,
    benchmarked against the S&P 500. Metrics include annualized volatility, Value at Risk under
    both historical and parametric assumptions, the Sharpe ratio, market beta, and CAPM alpha.
    Rolling 60-day windows for volatility, Sharpe, and beta show how the portfolio's risk profile
    has shifted over time. Weights are adjustable inline; a custom return dataset can be uploaded
    in CSV format.
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

dl_col1, dl_col2 = st.columns([2, 1])
with dl_col1:
    st.markdown("""<div class="explainer-body">
      By default the dashboard pulls one year of adjusted daily closing prices from Yahoo Finance
      for the 20 stocks listed below, plus the S&P 500 as benchmark. To use your own data, upload
      a CSV with dates as the index and one column per asset containing daily returns
      (not prices). The benchmark column is not required in a custom upload.
    </div>""", unsafe_allow_html=True)

with dl_col2:
    uploaded_file = st.file_uploader("Upload returns CSV (optional)", type=["csv"])

with st.spinner("Fetching price data..."):
    all_prices = load_prices(TICKERS + [BENCHMARK], start_date, end_date)

all_returns    = all_prices.pct_change().dropna()
bench_returns  = all_returns[BENCHMARK].copy()
asset_returns  = all_returns.drop(columns=[BENCHMARK]).copy()

if uploaded_file:
    custom = pd.read_csv(uploaded_file, index_col=0, parse_dates=True).dropna()
    df = custom.copy()
    st.success("Custom dataset loaded.")
else:
    df = asset_returns.copy()

# Rename columns to full names for display where possible
display_names = {t: NAMES.get(t, t) for t in df.columns}

# Download button
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
  Each cell represents the simple daily return for that asset,
  calculated as shown below. Returns are used directly for all downstream
  calculations; no log-return transformation is applied.
</div>""", unsafe_allow_html=True)
st.latex(r"r_t = \frac{P_t - P_{t-1}}{P_{t-1}}")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 2 — PORTFOLIO WEIGHTS (inline, labeled)
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">2. Portfolio weights</div>', unsafe_allow_html=True)

st.markdown("""<div class="explainer-body" style="margin-bottom:1rem;">
  Edit the weight column directly. Weights are normalized to sum to 100% before any
  calculation, so only relative magnitudes matter. The default is equal-weight at 5% per stock.
  Sort by any column; use the Ticker or Company columns to find positions quickly.
</div>""", unsafe_allow_html=True)

numeric_df = df.select_dtypes(include=np.number)
n_assets   = numeric_df.shape[1]
tickers_in = list(numeric_df.columns)

# Build editable weight dataframe
weight_init = pd.DataFrame({
    "Ticker":  tickers_in,
    "Company": [NAMES.get(t, t) for t in tickers_in],
    "Weight":  [5.0] * n_assets,
})

wt_col, bar_col = st.columns([1, 1])

with wt_col:
    edited = st.data_editor(
        weight_init,
        use_container_width=True,
        hide_index=True,
        disabled=["Ticker", "Company"],
        column_config={
            "Ticker":  st.column_config.TextColumn("Ticker", width="small"),
            "Company": st.column_config.TextColumn("Company", width="medium"),
            "Weight":  st.column_config.NumberColumn(
                "Weight", min_value=0.0, max_value=100.0,
                step=0.5, format="%.1f", width="small"
            ),
        },
        key="weight_editor",
    )

raw_weights_arr = edited["Weight"].values.astype(float)
if raw_weights_arr.sum() == 0:
    st.error("All weights are zero.")
    st.stop()
weights = raw_weights_arr / raw_weights_arr.sum()

with bar_col:
    # Live allocation donut / bar
    sorted_idx = np.argsort(weights)[::-1]
    sorted_names  = [NAMES.get(tickers_in[i], tickers_in[i]) for i in sorted_idx]
    sorted_weights = weights[sorted_idx] * 100

    # Color palette cycling through brand palette
    bar_colors = [BLUE, LBLUE, GREEN, LGREEN, LRED, RED, "#c47a00", LLBLUE,
                  "#6ab06a", "#3d7ab5", "#b94040", "#888888", "#c8a86e",
                  "#5a8a5a", "#8a5a5a", "#5a5a8a", "#8a8a5a", "#5a8a8a",
                  "#8a5a8a", "#aaaaaa"]
    colors_used = [bar_colors[i % len(bar_colors)] for i in range(len(sorted_names))]

    fig_alloc = go.Figure(go.Bar(
        x=sorted_weights,
        y=sorted_names,
        orientation="h",
        marker_color=colors_used,
        opacity=0.82,
        text=[f"{w:.1f}%" for w in sorted_weights],
        textposition="outside",
        textfont=dict(size=9, color="#333"),
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig_alloc.add_vline(x=100/n_assets, line_dash="dot",
                        line_color=GRAY, line_width=1,
                        annotation_text="Equal weight",
                        annotation_font=dict(size=9, color=GRAY),
                        annotation_position="top right")
    fig_alloc.update_layout(
        **{**BASE, "margin": dict(l=8, r=50, t=10, b=8)},
        height=max(300, n_assets * 22),
        xaxis=dict(**ax("Normalized weight (%)"), ticksuffix="%"),
        yaxis=dict(tickfont=dict(size=10, color="#444"),
                   showgrid=False, linecolor="#d4c9b8", linewidth=1,
                   autorange="reversed"),
        showlegend=False,
    )
    st.plotly_chart(fig_alloc, use_container_width=True)

st.markdown("""<div class="fig-caption">
  <b>Figure 0.</b> Normalized portfolio allocation after weight adjustment.
  The dotted line marks the equal-weight threshold. Positions to the right are
  overweight relative to equal-weight; positions to the left are underweight.
  The chart updates immediately when weights are changed in the table.
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# CORE CALCULATIONS
# ══════════════════════════════════════════════════════════════════════════
# Drop any rows with NaN before computing portfolio returns —
# yfinance occasionally returns sparse data for individual tickers
clean_df     = numeric_df.dropna()
if clean_df.shape[0] < 30:
    st.error("Insufficient clean data (fewer than 30 trading days). Check the tickers or upload a complete dataset.")
    st.stop()

port_returns  = clean_df.dot(weights)
rf_daily      = 0.02 / 252
cov_matrix    = clean_df.cov() * 252
port_vol      = np.sqrt(weights @ cov_matrix.values @ weights)
excess        = port_returns - rf_daily
sharpe        = (excess.mean() / excess.std()) * np.sqrt(252)

# Drop any remaining NaNs before percentile (safety guard)
pr_clean      = port_returns.dropna().values
var_hist_95   = np.percentile(pr_clean, 5)
var_hist_99   = np.percentile(pr_clean, 1)

# Parametric VaR (Gaussian)
mu_daily      = port_returns.mean()
sig_daily     = port_returns.std()
z95, z99      = 1.6449, 2.3263
var_param_95  = mu_daily - z95 * sig_daily
var_param_99  = mu_daily - z99 * sig_daily

# Beta and CAPM alpha (OLS: r_p = alpha + beta * r_m + e)
bench_aligned = bench_returns.reindex(port_returns.index).dropna()
port_aligned  = port_returns.reindex(bench_aligned.index)
cov_pm        = np.cov(port_aligned.values, bench_aligned.values)[0, 1]
var_m         = np.var(bench_aligned.values)
beta          = cov_pm / var_m if var_m > 0 else np.nan
rf_ann        = 0.02
alpha_ann     = ((port_aligned.mean() - rf_daily) - beta * (bench_aligned.mean() - rf_daily)) * 252

# Benchmark metrics
bench_vol     = bench_aligned.std() * np.sqrt(252)
bench_sharpe  = ((bench_aligned.mean() - rf_daily) / bench_aligned.std()) * np.sqrt(252)
bench_ret_ann = bench_aligned.mean() * 252

# Rolling metrics (60-day)
roll_win = 60
roll_vol   = port_returns.rolling(roll_win).std() * np.sqrt(252)
roll_sharpe= (port_returns.rolling(roll_win).mean() / port_returns.rolling(roll_win).std()) * np.sqrt(252)
roll_beta  = port_aligned.rolling(roll_win).cov(bench_aligned) / bench_aligned.rolling(roll_win).var()

# Cumulative returns
cum_port  = (1 + port_returns).cumprod()
cum_bench = (1 + bench_aligned).cumprod()
total_ret = cum_port.iloc[-1] - 1
bench_total_ret = cum_bench.iloc[-1] - 1

# Drawdown — corrected
rolling_max     = cum_port.cummax()
drawdown        = (cum_port - rolling_max) / rolling_max
max_dd          = drawdown.min()
trough_date     = drawdown.idxmin()
# Peak is the last date before the trough where cumulative return was at its max
pre_trough      = cum_port.loc[:trough_date]
peak_date       = pre_trough.idxmax()
# Recovery: first date after trough where we exceed the rolling max at trough
post_trough_dd  = drawdown.loc[trough_date:]
recovered       = post_trough_dd[post_trough_dd >= 0]
recovery_date   = recovered.index[0] if len(recovered) > 0 else None

# ══════════════════════════════════════════════════════════════════════════
# PORTFOLIO SUMMARY
# ══════════════════════════════════════════════════════════════════════════
port_ret_ann = port_returns.mean() * 252
excess_ret   = port_ret_ann - bench_ret_ann
alpha_str    = f"+{alpha_ann:.2%}" if alpha_ann >= 0 else f"{alpha_ann:.2%}"
excess_str   = f"+{excess_ret:.2%}" if excess_ret >= 0 else f"{excess_ret:.2%}"
vol_diff     = port_vol - bench_vol
vol_diff_str = f"+{vol_diff:.2%}" if vol_diff >= 0 else f"{vol_diff:.2%}"

st.markdown(f"""
<div class="abstract-box" style="margin-top:2rem;">
  <div class="abstract-label">Portfolio summary</div>
  <div class="abstract-text">
    At equal weight across {n_assets} positions, the portfolio earned {total_ret:.2%} over the
    past year against {bench_total_ret:.2%} for the S&P 500, a {excess_str} excess return.
    Annualized volatility of {port_vol:.2%} compares to {bench_vol:.2%} for the benchmark
    ({vol_diff_str}). The Sharpe ratio of {sharpe:.2f} exceeds the S&P 500's {bench_sharpe:.2f},
    reflecting better risk-adjusted performance despite similar absolute volatility.
    Market beta is {beta:.2f} and CAPM alpha is {alpha_str} annualized, indicating the portfolio
    {'captures more than proportional market moves and earns positive residual return' if beta > 1 and alpha_ann > 0
     else 'tracks the market closely with modest residual return' if abs(beta - 1) < 0.1
     else 'provides some defensive characteristics relative to the index'}.
    The maximum drawdown of {max_dd:.2%} occurred around {trough_date.strftime('%B %Y')}.
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 3 — KPI CARDS
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">3. Portfolio metrics</div>', unsafe_allow_html=True)

k1, k2, k3, k4, k5, k6 = st.columns(6)
kpi_data = [
    ("Annualized volatility", fmt_pct(port_vol),
     f"S&P 500: {fmt_pct(bench_vol)}", "warn" if port_vol > bench_vol * 1.1 else "neut"),
    ("Historical VaR (95%)", fmt_pct(abs(var_hist_95)),
     f"99%: {fmt_pct(abs(var_hist_99))}", "neg" if abs(var_hist_95) > 0.02 else "neut"),
    ("Parametric VaR (95%)", fmt_pct(abs(var_param_95)),
     f"Gaussian assumption, 99%: {fmt_pct(abs(var_param_99))}", "neut"),
    ("Sharpe ratio", fmt_f(sharpe),
     f"S&P 500: {fmt_f(bench_sharpe)}", "pos" if sharpe > bench_sharpe else "neut"),
    ("Market beta", fmt_f(beta),
     f"vs. S&P 500 daily returns", "neut"),
    ("CAPM alpha (ann.)", alpha_str,
     f"Residual over CAPM prediction", "pos" if alpha_ann > 0 else "neg"),
]
for col, (label, value, sub, cls) in zip([k1,k2,k3,k4,k5,k6], kpi_data):
    with col:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value {cls}">{value}</div>
          <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown(f"""<div class="fig-caption" style="margin-top:0.75rem;">
  <b>Table 2.</b> Summary risk and return metrics for the equal-weight portfolio.
  Historical VaR uses the empirical 5th percentile of daily returns; parametric VaR assumes
  normally distributed returns with the same mean and standard deviation.
  Beta is estimated by OLS regression of portfolio returns on benchmark returns.
  Alpha is the annualized intercept from that regression, net of the risk-free rate.
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 4 — CORRELATION MATRIX
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">4. Correlation matrix</div>', unsafe_allow_html=True)

corr = clean_df.corr()
# Rename to short tickers for heatmap readability
corr_display = corr.copy()

# Custom colorscale: red (negative) -> cream (zero) -> blue (positive)
colorscale = [
    [0.0,  "#b94040"],
    [0.4,  "#e8c8c8"],
    [0.5,  CREAM],
    [0.65, "#b8cfe0"],
    [1.0,  "#1a4f82"],
]

fig_corr = go.Figure(go.Heatmap(
    z=corr_display.values,
    x=list(corr_display.columns),
    y=list(corr_display.index),
    colorscale=colorscale,
    zmid=0,
    zmin=-1, zmax=1,
    colorbar=dict(
        title=dict(text="ρ", font=dict(size=12, color="#333")),
        tickfont=dict(size=10, color="#444"),
        thickness=10, len=0.8,
    ),
    hovertemplate="%{y} / %{x}: %{z:.3f}<extra></extra>",
    showscale=True,
))
fig_corr.update_layout(
    **{**BASE, "margin": dict(l=8, r=60, t=20, b=8)},
    height=420,
    xaxis=dict(tickfont=dict(size=10, color="#444"), showgrid=False,
               linecolor="#d4c9b8", linewidth=1),
    yaxis=dict(tickfont=dict(size=10, color="#444"), showgrid=False,
               linecolor="#d4c9b8", linewidth=1, autorange="reversed"),
)
st.plotly_chart(fig_corr, use_container_width=True)

st.markdown(f"""<div class="fig-caption">
  <b>Figure 1.</b> Pearson correlation matrix of daily returns across all {n_assets} portfolio
  constituents. Blue cells indicate positive co-movement; red cells indicate negative or near-zero
  correlation. Cells are not annotated to preserve readability at 20x20; hover for exact values.
  High pairwise correlations reduce diversification benefit; the pairwise table below identifies
  the strongest relationships.
</div>""", unsafe_allow_html=True)

# Top pairwise correlations table
corr_pairs = corr.unstack().rename_axis(['T1','T2']).reset_index(name='Correlation')
corr_pairs = corr_pairs[corr_pairs['T1'] != corr_pairs['T2']]
corr_pairs['Pair'] = corr_pairs.apply(lambda r: tuple(sorted([r['T1'], r['T2']])), axis=1)
corr_pairs = corr_pairs.drop_duplicates('Pair').drop(columns='Pair')
corr_pairs = corr_pairs.reindex(corr_pairs['Correlation'].abs().sort_values(ascending=False).index)
corr_pairs['Company 1'] = corr_pairs['T1'].map(NAMES)
corr_pairs['Company 2'] = corr_pairs['T2'].map(NAMES)
top_pairs = corr_pairs[['Company 1','Company 2','Correlation']].head(15)

st.markdown('<div class="sec-header" style="margin-top:1rem;">Top 15 pairwise correlations</div>', unsafe_allow_html=True)
st.dataframe(top_pairs.style.format({"Correlation": "{:.3f}"}), use_container_width=True)
st.markdown(f"""<div class="fig-caption">
  <b>Table 3.</b> Fifteen highest pairwise return correlations by absolute value.
  The Mastercard/Visa pair at the top reflects near-identical business models and identical macro
  sensitivities. High within-sector correlations (payments, mega-cap tech) are the primary
  constraint on diversification in this portfolio.
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 5 — RETURN DISTRIBUTION WITH KDE
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">5. Return distribution</div>', unsafe_allow_html=True)

fig_dist = go.Figure()
fig_dist.add_trace(go.Histogram(
    x=port_returns.values, nbinsx=50,
    marker_color=LBLUE, opacity=0.55, name="Portfolio",
    hovertemplate="Return: %{x:.3%}<br>Count: %{y}<extra></extra>",
))
# KDE overlay
kde_x, kde_d = kde_curve(port_returns.values, bw_factor=0.9)
if len(kde_x):
    bin_w   = (port_returns.max() - port_returns.min()) / 50
    kde_cnt = kde_d * len(port_returns) * bin_w
    fig_dist.add_trace(go.Scatter(
        x=kde_x, y=kde_cnt, mode="lines",
        line=dict(color="#2a2a2a", width=2.5), name="KDE", hoverinfo="skip",
    ))
# Benchmark distribution
fig_dist.add_trace(go.Histogram(
    x=bench_aligned.values, nbinsx=50,
    marker_color=LRED, opacity=0.35, name="S&P 500",
    hovertemplate="S&P 500: %{x:.3%}<br>Count: %{y}<extra></extra>",
))
# VaR lines
fig_dist.add_vline(x=var_hist_95, line_dash="dash", line_color=RED, line_width=1.5,
                   annotation_text=f"VaR 95% ({var_hist_95:.2%})",
                   annotation_font=dict(size=9, color=RED),
                   annotation_position="top left")
fig_dist.add_vline(x=0, line_dash="dot", line_color=GRAY, line_width=1)
fig_dist.update_layout(
    **BASE, height=320, barmode="overlay",
    xaxis=dict(**ax("Daily return"), tickformat=".1%"),
    yaxis=dict(**ax("Count")),
)
st.plotly_chart(fig_dist, use_container_width=True)

skew_val = float(port_returns.skew())
kurt_val = float(port_returns.kurtosis())  # excess kurtosis
skew_desc = 'slight left tail' if skew_val < -0.1 else 'slight right tail' if skew_val > 0.1 else 'approximately symmetric'
kurt_note = (
    f"Excess kurtosis of {kurt_val:.2f} is very high; over a one-year window this "
    f"is typically driven by one or two extreme return days rather than a structurally "
    f"fat-tailed distribution. Moment estimates should be interpreted cautiously."
    if kurt_val > 5 else
    f"Excess kurtosis of {kurt_val:.2f} indicates "
    f"{'mildly fat tails relative to a normal distribution' if kurt_val > 0.5 else 'near-normal tail behavior'}."
)
st.markdown(f"""<div class="fig-caption">
  <b>Figure 2.</b> Daily return distribution for the portfolio (blue) and S&P 500 (red), with
  Gaussian KDE overlay (Silverman bandwidth). The portfolio daily mean is {port_returns.mean():.3%}
  with a standard deviation of {port_returns.std():.3%}. Skewness is {skew_val:.3f} ({skew_desc}).
  {kurt_note} The dashed red line marks the empirical 95% VaR.
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 6 — CUMULATIVE RETURNS
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">6. Cumulative returns</div>', unsafe_allow_html=True)

fig_cum = go.Figure()
fig_cum.add_trace(go.Scatter(
    x=cum_port.index, y=cum_port.values,
    mode="lines", name="Portfolio",
    line=dict(color=BLUE, width=2.5),
    hovertemplate="Date: %{x|%Y-%m-%d}<br>Value: %{y:.3f}<extra></extra>",
))
fig_cum.add_trace(go.Scatter(
    x=cum_bench.index, y=cum_bench.values,
    mode="lines", name="S&P 500",
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
  <b>Figure 3.</b> Growth of $1 invested in the portfolio vs. the S&P 500 over the past year.
  The portfolio returned {total_ret:.2%} against {bench_total_ret:.2%} for the benchmark,
  an outperformance of {excess_str}. Periods of divergence reflect both the portfolio's higher
  beta ({beta:.2f}) and its sector concentrations in mega-cap technology and payments.
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 7 — DRAWDOWN ANALYSIS (corrected)
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">7. Drawdown analysis</div>', unsafe_allow_html=True)

fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(
    x=drawdown.index, y=drawdown.values * 100,
    mode="lines", fill="tozeroy",
    fillcolor=f"rgba(185,64,64,0.18)",
    line=dict(color=RED, width=1.5),
    name="Drawdown",
    hovertemplate="Date: %{x|%Y-%m-%d}<br>Drawdown: %{y:.2f}%<extra></extra>",
))
fig_dd.add_hline(y=max_dd * 100, line_dash="dot", line_color=RED, line_width=1,
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
with dd1:
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-label">Maximum drawdown</div>
      <div class="kpi-value neg">{max_dd:.2%}</div>
      <div class="kpi-sub">Peak-to-trough decline</div>
    </div>""", unsafe_allow_html=True)
with dd2:
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-label">Drawdown peak</div>
      <div class="kpi-value neut">{peak_date.strftime('%b %d, %Y')}</div>
      <div class="kpi-sub">Last high before trough</div>
    </div>""", unsafe_allow_html=True)
with dd3:
    recovery_str = recovery_date.strftime('%b %d, %Y') if recovery_date else "Not yet recovered"
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-label">Recovery date</div>
      <div class="kpi-value {'pos' if recovery_date else 'warn'}">{recovery_str}</div>
      <div class="kpi-sub">First date drawdown reached 0</div>
    </div>""", unsafe_allow_html=True)

recovery_str = (
    'The portfolio subsequently recovered to a new high by ' + recovery_date.strftime('%B %d, %Y') + '.'
    if recovery_date else
    'The portfolio had not fully recovered to its prior peak within the analysis window.'
)
st.markdown(f"""<div class="fig-caption">
  <b>Figure 4.</b> Portfolio drawdown expressed as percentage decline from the rolling peak.
  The maximum drawdown of {max_dd:.2%} reached its trough on {trough_date.strftime('%B %d, %Y')},
  following a peak on {peak_date.strftime('%B %d, %Y')}. {recovery_str}
  Drawdown at time t is the percentage below the prior running maximum, as defined below.
</div>""", unsafe_allow_html=True)
st.latex(r"D_t = \frac{V_t - \max_{s \leq t} V_s}{\max_{s \leq t} V_s}")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 8 — ROLLING RISK METRICS
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">8. Rolling risk metrics (60-day window)</div>', unsafe_allow_html=True)

st.markdown("""<div class="explainer-body">
  Rolling metrics reveal how the portfolio's risk profile has changed over time, which a single
  point-in-time estimate cannot show. A rising rolling volatility during a drawdown confirms
  risk escalation; a declining rolling Sharpe during underperformance confirms deteriorating
  risk-adjusted returns. Rolling beta shows when the portfolio became more or less sensitive
  to broad market moves.
</div>""", unsafe_allow_html=True)

rc1, rc2 = st.columns(2)

with rc1:
    # Rolling volatility
    fig_rvol = go.Figure()
    fig_rvol.add_trace(go.Scatter(
        x=roll_vol.index, y=roll_vol.values * 100,
        mode="lines", name="Portfolio",
        line=dict(color=BLUE, width=2.2),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Vol: %{y:.2f}%<extra></extra>",
    ))
    bench_roll_vol = bench_aligned.rolling(roll_win).std() * np.sqrt(252)
    fig_rvol.add_trace(go.Scatter(
        x=bench_roll_vol.index, y=bench_roll_vol.values * 100,
        mode="lines", name="S&P 500",
        line=dict(color=GRAY, width=1.5, dash="dash"),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>S&P 500 Vol: %{y:.2f}%<extra></extra>",
    ))
    fig_rvol.update_layout(
        **BASE, height=280,
        xaxis=dict(**ax("Date")),
        yaxis=dict(**ax("Annualized volatility (%)"), ticksuffix="%"),
    )
    st.plotly_chart(fig_rvol, use_container_width=True)
    st.markdown("""<div class="fig-caption">
      <b>Figure 5.</b> 60-day rolling annualized volatility for the portfolio vs. S&P 500.
      Periods where the portfolio line exceeds the benchmark indicate elevated idiosyncratic
      or concentrated risk.
    </div>""", unsafe_allow_html=True)

with rc2:
    # Rolling Sharpe
    fig_rs = go.Figure()
    fig_rs.add_trace(go.Scatter(
        x=roll_sharpe.index, y=roll_sharpe.values,
        mode="lines", name="Rolling Sharpe",
        line=dict(color=GREEN, width=2.2),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Sharpe: %{y:.2f}<extra></extra>",
    ))
    fig_rs.add_hline(y=0, line_dash="dot", line_color="#d4c9b8", line_width=1)
    fig_rs.add_hline(y=1.0, line_dash="dot", line_color=GRAY, line_width=1,
                     annotation_text="Sharpe = 1",
                     annotation_font=dict(size=9, color=GRAY))
    fig_rs.update_layout(
        **BASE, height=280,
        xaxis=dict(**ax("Date")),
        yaxis=dict(**ax("Sharpe ratio (annualized)")),
        showlegend=False,
    )
    st.plotly_chart(fig_rs, use_container_width=True)
    st.markdown("""<div class="fig-caption">
      <b>Figure 6.</b> 60-day rolling annualized Sharpe ratio. Periods below zero indicate
      negative risk-adjusted returns over that window. The dashed line marks Sharpe = 1,
      a conventional threshold for adequate compensation per unit of risk.
    </div>""", unsafe_allow_html=True)

# Rolling beta — full width
fig_rb = go.Figure()
fig_rb.add_trace(go.Scatter(
    x=roll_beta.index, y=roll_beta.values,
    mode="lines", name="Rolling beta",
    line=dict(color=LBLUE, width=2.2),
    hovertemplate="Date: %{x|%Y-%m-%d}<br>Beta: %{y:.3f}<extra></extra>",
))
fig_rb.add_hline(y=1.0, line_dash="dash", line_color=GRAY, line_width=1.5,
                 annotation_text="Beta = 1 (market)",
                 annotation_font=dict(size=10, color=GRAY),
                 annotation_position="top right")
fig_rb.add_hline(y=beta, line_dash="dot", line_color=BLUE, line_width=1.5,
                 annotation_text=f"Full-period beta ({beta:.2f})",
                 annotation_font=dict(size=10, color=BLUE),
                 annotation_position="bottom right")
fig_rb.update_layout(
    **{**BASE, "margin": dict(l=70, r=20, t=20, b=8)},
    height=280,
    xaxis=dict(**ax("Date")),
    yaxis=dict(**ax("Beta vs. S&P 500")),
    showlegend=False,
)
st.plotly_chart(fig_rb, use_container_width=True)
st.markdown(f"""<div class="fig-caption">
  <b>Figure 7.</b> 60-day rolling market beta computed by rolling OLS covariance over benchmark
  variance. The full-period beta of {beta:.2f} (dotted blue) is shown for reference.
  Beta excursions above 1.3 or below 0.7 indicate periods of materially elevated or reduced
  market sensitivity, often coinciding with large-cap tech momentum or defensive rotation.
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 9 — VaR METHODOLOGY COMPARISON
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">9. Value at Risk: historical vs. parametric</div>', unsafe_allow_html=True)

var_data = {
    "Method": ["Historical (empirical)", "Parametric (Gaussian)"],
    "VaR 95%": [f"{abs(var_hist_95):.2%}", f"{abs(var_param_95):.2%}"],
    "VaR 99%": [f"{abs(var_hist_99):.2%}", f"{abs(var_param_99):.2%}"],
    "Assumption": ["No distributional assumption; uses empirical return percentiles",
                   "Returns are normally distributed with sample mean and std dev"],
}
st.dataframe(pd.DataFrame(var_data), use_container_width=True, hide_index=True)

hist_param_diff = abs(var_hist_95) - abs(var_param_95)
tail_note = (
    f"The historical estimate exceeds the parametric estimate at 95% by "
    f"{abs(hist_param_diff):.2%}, suggesting the empirical distribution has "
    f"fatter left tails than a normal distribution predicts."
    if hist_param_diff > 0 else
    "The parametric estimate exceeds the historical at 95%, suggesting the "
    "empirical distribution is somewhat thinner-tailed than Gaussian."
)
st.markdown(f"""<div class="fig-caption">
  <b>Table 4.</b> Comparison of historical and parametric VaR at 95% and 99% confidence levels.
  Historical VaR uses the empirical 5th and 1st percentiles directly from the return series.
  Parametric VaR uses the Gaussian quantile formula shown below, where
  z(0.95) = 1.645 and z(0.99) = 2.326. {tail_note}
</div>""", unsafe_allow_html=True)
st.latex(r"\text{VaR}_\alpha = -(\mu - z_\alpha \, \sigma)")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 10 — INDIVIDUAL ASSET RISK
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">10. Individual asset risk contribution</div>', unsafe_allow_html=True)

# Marginal and component risk contribution
asset_vols   = clean_df.std() * np.sqrt(252)
marginal_rc  = cov_matrix.values @ weights
component_rc = marginal_rc * weights
total_risk_sq = weights @ cov_matrix.values @ weights
pct_rc        = component_rc / total_risk_sq * 100  # % contribution to variance

risk_df = pd.DataFrame({
    "Asset": [NAMES.get(t, t) for t in tickers_in],
    "Ticker": tickers_in,
    "Weight": [f"{w:.1%}" for w in weights],
    "Volatility (ann.)": [f"{v:.2%}" for v in asset_vols],
    "% Variance contribution": [f"{p:.1f}%" for p in pct_rc],
}).sort_values("% Variance contribution", ascending=False, key=lambda x: x.str.rstrip('%').astype(float))

st.dataframe(risk_df, use_container_width=True, hide_index=True)

# Bar chart of variance contributions
fig_rc = go.Figure(go.Bar(
    x=[NAMES.get(t, t) for t in tickers_in],
    y=pct_rc,
    marker_color=[LBLUE if p > 0 else LRED for p in pct_rc],
    opacity=0.85,
    text=[f"{p:.1f}%" for p in pct_rc],
    textposition="outside",
    textfont=dict(size=10, color="#333"),
))
fig_rc.add_hline(y=100/n_assets, line_dash="dot", line_color=GRAY, line_width=1.2,
                 annotation_text="Equal contribution",
                 annotation_font=dict(size=9, color=GRAY),
                 annotation_position="top right")
fig_rc.update_layout(
    **BASE, height=320,
    xaxis={**ax("Asset"), "tickangle": -35},
    yaxis=dict(**ax("% contribution to portfolio variance"), ticksuffix="%"),
    showlegend=False,
)
st.plotly_chart(fig_rc, use_container_width=True)

top_rc = tickers_in[np.argmax(pct_rc)]
st.markdown(f"""<div class="fig-caption">
  <b>Figure 8.</b> Component variance contribution of each position, expressed as a percentage
  of total portfolio variance. Component contribution is computed as
  RC_i = w_i * (Sigma * w)_i / (w^T * Sigma * w),
  accounting for each asset's weight, own volatility, and correlation with all other positions.
  The dashed line marks the equal-contribution threshold of {100/n_assets:.1f}%.
  {NAMES.get(top_rc, top_rc)} is the largest single contributor to portfolio variance;
  positions above the dashed line are overweight from a risk perspective
  relative to their capital allocation.
</div>""", unsafe_allow_html=True)
st.latex(r"RC_i = \frac{w_i \, (\Sigma \mathbf{w})_i}{\mathbf{w}^T \Sigma \mathbf{w}}")

# ══════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════
st.markdown(f"""<div class="paper-footer">
  Data: Yahoo Finance daily adjusted closing prices, {start_date.strftime('%Y-%m-%d')} to
  {end_date.strftime('%Y-%m-%d')}. Risk-free rate: 2.00% annualized (constant).
  Annualized figures use a 252 trading-day convention.
  Beta estimated by OLS: r_p = alpha + beta * r_m + epsilon.
  VaR figures are 1-day estimates. Rolling window: {roll_win} trading days.
  This dashboard is for analytical and educational purposes only and does not constitute
  investment advice.
</div>""", unsafe_allow_html=True)
