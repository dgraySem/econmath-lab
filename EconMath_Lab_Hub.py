# EconMath_Lab_Hub.py
# EconMath Lab: Math You Can Use — Comprehensive single-file Streamlit dashboard.
# Cheyney University of Pennsylvania — Econ I.
# Design principle: Story → Numbers → Graph → Equation → Meaning.
# Uses st.secrets for FRED / Anthropic API key handling (hidden keys).

import datetime as dt
import math
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

# --- Configuration ---
st.set_page_config(
    page_title="EconMath Lab: Math You Can Use",
    page_icon="📊",
    layout="wide",
    menu_items={"About": "EconMath Lab — interactive economics + math labs for Econ I at Cheyney University."},
)

# --- ACCESSIBILITY INJECTION: Global Font Size 22px + Brand fonts ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=IBM+Plex+Mono:wght@400;600&display=swap');
html, body, p, li, label, .stMetricValue > div, .stMetricLabel > div, .stMarkdown div {
    font-size: 22px !important;
    line-height: 1.45 !important;
    font-family: 'DM Sans', sans-serif;
}
code, pre { font-family: 'IBM Plex Mono', monospace; }
h1 { font-size: 40px !important; color: #0B1F3A; margin-bottom: 0.4rem !important; }
h2 { font-size: 32px !important; color: #0B1F3A; }
h3 { font-size: 27px !important; color: #0B1F3A; }
/* Reclaim the dead space above the page and stop lines running edge-to-edge */
.block-container { padding-top: 1.2rem !important; max-width: 1150px; }
/* Inner sub-tabs (Simple / Math / Challenge): slightly smaller so all three always fit */
button[data-baseweb="tab"] p { font-size: 19px !important; }
button[data-baseweb="tab"] { padding: 6px 14px !important; }
/* Sidebar lab navigation: readable but compact */
section[data-testid="stSidebar"] div[role="radiogroup"] label p { font-size: 20px !important; }
section[data-testid="stSidebar"] .stMarkdown div, section[data-testid="stSidebar"] label { font-size: 19px !important; }
.stButton>button { background: #0B1F3A; color: #FAF7F0; border: 1px solid #C9A227; border-radius: 6px; }
.stButton>button:hover { background: #C9A227; color: #0B1F3A; }
</style>
""", unsafe_allow_html=True)

# ======================
# Utilities
# ======================

CONF_PATH = "confidence_log.csv"

# Month-end frequency alias: "M" was renamed to "ME" in newer pandas.
try:
    pd.tseries.frequencies.to_offset("ME")
    MONTH_END = "ME"
except ValueError:
    MONTH_END = "M"

FACES = {1: "😟 1", 2: "😕 2", 3: "😐 3", 4: "🙂 4", 5: "😃 5"}

def three_levels(new_fn, math_fn, challenge_fn, key: str):
    """Render the 'Tell Me Like I'm New → Show Me the Math → Challenge Me' sub-tabs."""
    t1, t2, t3 = st.tabs(["🌱 Simple", "🧮 The Math", "🔥 Challenge"])
    with t1:
        new_fn()
    with t2:
        math_fn()
    with t3:
        challenge_fn()

def say_it(latex: str, phonetic: str, meaning: str):
    st.latex(latex)
    with st.expander("🗣️ How do I say this out loud?"):
        st.markdown(f"**Say it:** \u201c{phonetic}\u201d")
        st.markdown(f"**What it means:** {meaning}")

def why_care(text: str, key: str):
    with st.expander("💡 Why should I care?"):
        st.markdown(text)

def std_layout(fig, title=None, height=480, xaxis="Date", yaxis=None):
    """Central layout manager to prevent jumbled overlapping text in Plotly charts."""
    fig.update_layout(
        template="plotly_white", 
        height=height, 
        title=dict(text=title, font=dict(size=18, color="#0B1F3A")), 
        # Smaller base font for ticks to prevent overlap
        font=dict(family="DM Sans, sans-serif", size=14, color="#333333"),
        legend=dict(
            orientation="h", 
            yanchor="bottom", y=1.05, # Push above the chart
            xanchor="right", x=1,     # Align to the right to avoid title collision
            bgcolor="rgba(255,255,255,0.8)"
        ),
        # Extra left and bottom margin to prevent axis numbers/dates from cutting off
        margin=dict(l=70, r=30, t=80, b=60), 
        xaxis_title=dict(text=xaxis, font=dict(size=15)), 
        yaxis_title=dict(text=yaxis, font=dict(size=15)),
        hovermode="x unified" # Combines overlapping tooltips into one clean box
    )
    # Lighten grid lines for a cleaner classroom presentation
    fig.update_xaxes(showline=True, linewidth=1, linecolor='lightgray', gridcolor='rgba(200,200,200,0.2)')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='lightgray', gridcolor='rgba(200,200,200,0.2)')
    return fig

def make_dual_axis(x, left_y, left_name, right_y, right_name,
                   left_color="#0068C9", right_color="#ff7f0e", title=None):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=x, y=left_y, name=left_name, mode="lines",
                             line=dict(width=3, color=left_color)), secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=right_y, name=right_name, mode="lines",
                             line=dict(width=3, color=right_color)), secondary_y=True)
    fig.update_yaxes(title_text=left_name, secondary_y=False, title_font=dict(color=left_color))
    fig.update_yaxes(title_text=right_name, secondary_y=True, title_font=dict(color=right_color))
    return std_layout(fig, title=title, height=550)

@st.cache_data(ttl=3600)
def fred_series(series_id: str, start: dt.date, end: dt.date, api_key: str) -> pd.Series:
    """Generic FRED pull; returns Series indexed by date. Empty series on any failure."""
    empty = pd.Series(dtype=float, name=series_id, index=pd.DatetimeIndex([]))
    if not api_key:
        return empty
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = dict(series_id=series_id, api_key=api_key, file_type="json",
                  observation_start=start.isoformat(), observation_end=end.isoformat())
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        obs = r.json().get("observations", [])
        if not obs:
            return empty
        df = pd.DataFrame(obs)
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        s = df.set_index("date")["value"].dropna()
        s.name = series_id
        return s
    except Exception:
        return empty

@st.cache_data(ttl=3600)
def load_fred_data(start_date: dt.date, end_date: dt.date, api_key: str) -> pd.DataFrame:
    """Fetches and cleans all series used across tabs."""
    series_map = {
        "UNRATE": "UNRATE",                    # National unemployment
        "CPIAUCSL": "CPIAUCSL",                # National CPI-U
        "CPI_Philly": "CUURS12BSA0",           # CPI-U Philadelphia-Camden-Wilmington
        "Philly_Unemp": "PAPHIL5URN",          # Unemployment rate, Philadelphia County/City
        "Philly_Payrolls": "PHIL942NA",        # Nonfarm payrolls, Philadelphia MSA
        "Avg_Hourly_Earnings": "CES0500000003",# Avg hourly earnings, total private
        "FedFunds": "FEDFUNDS",                # Policy interest rate
    }
    data = {k: fred_series(v, start_date, end_date, api_key) for k, v in series_map.items()}
    # Drop series that came back empty — a single empty series would otherwise
    # degrade the combined index from DatetimeIndex to plain Index and break .resample().
    data = {k: s for k, s in data.items() if s is not None and not s.empty}
    if not data:
        return pd.DataFrame()  # nothing loaded — app runs in simulation-only mode
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().resample(MONTH_END).mean()
    # ffill only slower-frequency series so gaps in monthly data stay visible
    for col in ["CPI_Philly"]:
        if col in df.columns:
            df[col] = df[col].ffill()
    df = df.dropna(how="all")
    if "CPIAUCSL" in df.columns:
        df["Inflation (YoY %)"] = df["CPIAUCSL"].pct_change(periods=12) * 100
    if "CPI_Philly" in df.columns:
        df["Philly Inflation (YoY %)"] = df["CPI_Philly"].pct_change(periods=12) * 100
    if "Avg_Hourly_Earnings" in df.columns:
        df["Wage Growth (YoY %)"] = df["Avg_Hourly_Earnings"].pct_change(periods=12) * 100
    return df


# ----------------------
# Robust Memory + CSV + Sheets Logging
# ----------------------
SHEET_HEADER = ["timestamp", "date", "lab", "when", "score"]

@st.cache_resource
def get_memory_log():
    """Returns a global mutable list shared across ALL active user sessions on the server.
    This guarantees logs aren't lost on Streamlit Cloud until the container reboots."""
    return []

def sheets_configured() -> bool:
    """True when Google Sheets credentials are present in secrets."""
    try:
        return "gcp_service_account" in st.secrets and "GSHEET_ID" in st.secrets
    except Exception:
        return False

@st.cache_resource
def get_worksheet():
    """Authorize the service account once and return the 'confidence' worksheet."""
    import gspread
    from google.oauth2.service_account import Credentials
    creds = Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]),
        scopes=["https://www.googleapis.com/auth/spreadsheets"],
    )
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(st.secrets["GSHEET_ID"])
    try:
        ws = sh.worksheet("confidence")
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title="confidence", rows=2000, cols=len(SHEET_HEADER))
    if not ws.get_values("A1:A1"):
        ws.append_row(SHEET_HEADER, value_input_option="USER_ENTERED")
    return ws

def log_confidence(lab: str, when: str, score: int) -> str:
    """Logs to Memory (always), tries Sheets (if configured), and backups to CSV."""
    row_dict = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "date": dt.date.today().isoformat(),
        "lab": lab, 
        "when": when, 
        "score": int(score)
    }
    
    # 1. Always append to the server-wide memory pool
    get_memory_log().append(row_dict)

    # 2. Try Google Sheets (Highly Persistent)
    if sheets_configured():
        try:
            ws = get_worksheet()
            ws.append_row(list(row_dict.values()), value_input_option="USER_ENTERED")
            return "sheets"
        except Exception:
            pass  # Fall through to CSV backup if API fails

    # 3. Try Local CSV (Ephemeral on Cloud, but good for local/temporary backup)
    try:
        row_df = pd.DataFrame([row_dict])
        row_df.to_csv(CONF_PATH, mode="a", header=not os.path.exists(CONF_PATH), index=False)
        return "csv/memory"
    except Exception:
        return "memory" # Successfully logged to memory pool, even if file system failed

def load_confidence():
    """Reads from Sheets first. If missing, combines the Memory Pool + Local CSV."""
    # Try Google Sheets
    if sheets_configured():
        try:
            df = pd.DataFrame(get_worksheet().get_all_records())
            if not df.empty:
                df["score"] = pd.to_numeric(df["score"], errors="coerce")
                df = df.dropna(subset=["score"])
            return df, "sheets"
        except Exception:
            pass

    # Fallback: Combine the Local CSV and the Server Memory Pool
    df_csv = pd.DataFrame(columns=SHEET_HEADER)
    if os.path.exists(CONF_PATH):
        try:
            df_tmp = pd.read_csv(CONF_PATH)
            if not df_tmp.empty:
                df_csv = df_tmp
        except Exception:
            pass
            
    df_mem = pd.DataFrame(get_memory_log(), columns=SHEET_HEADER)
    
    if df_csv.empty and df_mem.empty:
        return pd.DataFrame(columns=SHEET_HEADER), "none"
        
    # Merge and remove duplicates (in case CSV and Memory recorded the exact same event)
    df_combined = pd.concat([df_csv, df_mem], ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset=["timestamp", "lab", "when"])
    
    df_combined["score"] = pd.to_numeric(df_combined["score"], errors="coerce")
    return df_combined, "csv/memory"


# ======================
# Sidebar: Lab Navigation, Confidence, Settings — & Global Data Loading
# ======================
PAGE_TITLES = [
    "🏠 Start",
    "💵 Paycheck",
    "🛒 Inflation",
    "📈 Supply & Demand",
    "⚖️ Marginal",
    "💳 Credit",
    "🎓 College",
    "🏙️ Philadelphia",
    "💰 Wealth",
    "🧮 Translator",
    "🎮 Economy",
    "🧑‍🏫 Instructor",
]

with st.sidebar:
    st.markdown("## 📚 Labs")
    page = st.radio("Choose a lab", PAGE_TITLES, label_visibility="collapsed", key="nav")

    st.markdown("---")
    st.markdown("### 🧠 Math Confidence")
    current_lab = page.split(" ", 1)[1] if " " in page else page
    
    if current_lab not in ("Start", "Instructor"):
        st.caption(f"Logging for: **{current_lab}** · anonymous")
        conf_when = st.radio("When?", ["Start of session", "End of session"], horizontal=True, key="conf_when")
        conf_score = st.select_slider("How comfortable are you with today's math?",
            options=list(FACES), format_func=FACES.get, value=3, key="conf_score")
        if st.button("Log my confidence", key="conf_btn", use_container_width=True):
            dest = log_confidence(current_lab, "before" if conf_when.startswith("Start") else "after", conf_score)
            if dest == "sheets":
                st.success("✅ Logged securely!")
            else:
                st.success("✅ Logged for this session!")
    else:
        st.info("👆 **Please do this first:**\n\nClick the **Paycheck** tab (or any other lab) above to log your starting score.")

    st.markdown("---")
    with st.expander("⚙️ Data settings"):
        FRED_API_KEY = st.secrets.get("FRED_API_KEY", os.getenv("FRED_API_KEY", ""))
        ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY", ""))
        if not FRED_API_KEY:
            st.warning("⚠️ FRED API Key not found — simulation-only mode.")
        else:
            st.success("✅ FRED API Key detected.")
        if ANTHROPIC_API_KEY:
            st.success("✅ AI Translator enabled.")
        end_date_input = st.date_input("End date", value=dt.date.today())
        years_back = st.slider("History (years)", 5, 40, 15)
        start_date = end_date_input - dt.timedelta(days=365 * years_back)
    st.caption(f"Data: {start_date} → {end_date_input}")

df_macro = load_fred_data(start_date, end_date_input, FRED_API_KEY)
HAS_DATA = not df_macro.empty

# ----------------------
# Tab 0: Start Here
# ----------------------
if page == "🏠 Start":
    st.title("📊 EconMath Lab: Math You Can Use")
    st.markdown("""
> Economics is about **choices**: rent, wages, food, cars, jobs, credit, college, inflation, and wealth.
> Mathematics gives us a way to **measure** those choices.
>
> You do **not** need to be a "math person."
> In every lab we start with the **story**, see the **numbers**, draw the **picture**,
> and only then meet the **equation**.

**Our path in every lab:**

$$\\text{Story} \\rightarrow \\text{Numbers} \\rightarrow \\text{Graph} \\rightarrow \\text{Equation} \\rightarrow \\text{Meaning}$$
""")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**💵 Money Labs**\n\nPaycheck • Inflation • Credit • College")
    with c2:
        st.markdown("**📈 Market Labs**\n\nSupply & Demand • Marginal Thinking • Build an Economy")
    with c3:
        st.markdown("**🏙️ Your City, Your Data**\n\nPhiladelphia Lab • Wealth Simulator • Math Translator")
        
    st.info("👆 **Please do this first:** Click the **💵 Paycheck** tab (or any other lab in the sidebar) to log your starting score. Log it again when you finish! It's anonymous, and it helps make this course a **gateway**, not a filter.")
    
    if not HAS_DATA:
        st.warning("Live data is offline right now — every lab still runs fully on its built-in simulations.")

# ----------------------
# Tab 1: Paycheck (Arithmetic, Percentages, Slope)
# ----------------------
if page == "💵 Paycheck":
    st.title("💵 My Paycheck")
    st.markdown("**Economic question:** *How much do I actually earn?* — and the math hiding inside it: arithmetic, percentages, and **slope**.")

    cA, cB = st.columns(2)
    with cA:
        wage = st.slider("Hourly wage ($)", 7.25, 40.0, 15.0, 0.25, key="pay_wage")
    with cB:
        hours = st.slider("Hours per week", 1, 60, 20, key="pay_hours")

    weekly = wage * hours
    monthly = wage * hours * 52 / 12
    tbl = pd.DataFrame({"Hours": range(1, max(hours, 6) + 1)})
    tbl["Pay ($)"] = (tbl["Hours"] * wage).round(2)

    def pay_new():
        st.markdown(f"""
You earn **\${wage:.2f} per hour**. Work one more hour and your pay rises by **exactly \${wage:.2f}** —
no more, no less. That steady, repeatable increase has a math name: the **slope**.
""")
        st.dataframe(tbl.head(6), hide_index=True, use_container_width=True)

    def pay_math():
        fig = go.Figure(go.Scatter(x=tbl["Hours"], y=tbl["Pay ($)"], mode="lines+markers", line=dict(width=3)))
        st.plotly_chart(std_layout(fig, "Pay vs Hours — a perfectly straight line", 420, "Hours", "Pay ($)"),
                        use_container_width=True, key="pay_fig")
        say_it(r"\text{Pay} = %.2f \times \text{Hours}" % wage,
               f"Pay equals {wage:.2f} times hours",
               "Every extra hour adds the same amount. The wage IS the slope of this line.")
        say_it(r"m = \frac{\Delta y}{\Delta x}",
               "m equals the change in y divided by the change in x",
               f"Here: change in pay ÷ change in hours = \${wage:.2f} per hour. Δ (delta) just means 'change in.'")

    def pay_challenge():
        st.markdown(f"""
* Weekly pay: **\${weekly:,.2f}**. Monthly pay: **\${monthly:,.2f}**.
* Why multiply by **52/12** instead of 4? Because a month is not 4 weeks — it's 52 weeks ÷ 12 months ≈ **4.33 weeks**.
""")
        st.latex(r"\text{Monthly Income} = \text{Wage} \times \text{Hours} \times \tfrac{52}{12}")
        guess = st.number_input("Challenge: your wage rises 8%. What's the new hourly wage?", 0.0, 100.0, 0.0, key="pay_guess")
        if guess:
            true = wage * 1.08
            if abs(guess - true) < 0.02:
                st.success(f"Correct — \${true:.2f}. An 8% raise means multiply by **1.08** (the whole wage plus 8% more).")
            else:
                st.info(f"Not quite. 8% of \${wage:.2f} is \${wage*0.08:.2f}. Add it back: **\${true:.2f}**. Shortcut: × 1.08.")

    three_levels(pay_new, pay_math, pay_challenge, "pay")

    if HAS_DATA and "Avg_Hourly_Earnings" in df_macro.columns and df_macro["Avg_Hourly_Earnings"].notna().sum() > 12:
        st.subheader("Reality check: what does the average American private-sector worker earn?")
        ahe = df_macro["Avg_Hourly_Earnings"].dropna()
        latest_ahe = float(ahe.iloc[-1])
        fig_ahe = go.Figure(go.Scatter(x=ahe.index, y=ahe.values, mode="lines", line=dict(width=3, color="#0068C9")))
        fig_ahe.add_hline(y=wage, line_dash="dash", line_color="#C9A227",
                          annotation_text=f"Your slider wage (${wage:.2f})")
        st.plotly_chart(std_layout(fig_ahe, "Average Hourly Earnings, Total Private (FRED: CES0500000003)",
                                   450, "Date", "$ per hour"), use_container_width=True, key="pay_ahe")
        pos = "below" if wage < latest_ahe else "above"
        st.markdown(f"""
**How to read this chart**

* The blue line is the **national average hourly wage** for private-sector workers — currently about **\${latest_ahe:.2f}/hr**.
* The gold dashed line is **your slider wage** (\${wage:.2f}/hr), which sits **{pos}** the national average.
* Notice the line rises over time. Ask: is that a **raise** in real life, or just **inflation**? That question is exactly where the next tab begins.
""")

    why_care("Slope shows up in wages, overtime, taxes, surge pricing, and phone plans. "
             "If you can read a slope, you can read a contract — and catch when one is bad for you.", "pay")

# ----------------------
# Tab 2: Inflation (Percent change, indexes, purchasing power)
# ----------------------
if page == "🛒 Inflation":
    st.title("🛒 Why Did Everything Get So Expensive?")
    st.markdown("**Economic question:** *What does inflation do to my purchasing power?* — the math: **percent change** and **index numbers**.")

    st.subheader("1️⃣ The rent problem (percent change)")
    c1, c2 = st.columns(2)
    with c1:
        p0 = st.number_input("Old rent ($)", 100.0, 5000.0, 1200.0, 50.0, key="infl_p0")
    with c2:
        p1 = st.number_input("New rent ($)", 100.0, 6000.0, 1350.0, 50.0, key="infl_p1")
    dollars, pct = p1 - p0, (p1 - p0) / p0 * 100

    def infl_new():
        st.markdown(f"""
Rent went up **\${dollars:,.0f}**. But is that a lot? It depends what you compare it to.
Compare the **\${dollars:,.0f} increase** with the **original \${p0:,.0f} rent**:
the increase is about **{pct:.1f}%** of what you used to pay. That comparison IS percent change.
""")

    def infl_math():
        say_it(r"\%\Delta P = \frac{P_1 - P_0}{P_0} \times 100",
               "Percent change in P equals P-one minus P-zero, divided by P-zero, times one hundred",
               "New minus old, divided by old — then scale to 'per hundred.' P₀ = before, P₁ = after.")
        st.metric("Your rent increase", f"{pct:.1f}%")

    def infl_challenge():
        yrs = st.slider("Years of steady 4% inflation", 1, 30, 5, key="infl_yrs")
        total = ((1.04 ** yrs) - 1) * 100
        st.markdown(f"""
If prices rise **4% per year for {yrs} years**, they don't rise {4*yrs}%.
They multiply: (1.04)^{yrs} ≈ {(1.04**yrs):.3f}, a **{total:.1f}%** total increase.
That extra {total - 4*yrs:.1f} points is **compounding** — inflation charging interest on itself.
""")
        st.latex(r"P_t = P_0(1+\pi)^t")

    three_levels(infl_new, infl_math, infl_challenge, "infl")

    if HAS_DATA and "Inflation (YoY %)" in df_macro.columns and df_macro["Inflation (YoY %)"].notna().sum() > 12:
        st.subheader("2️⃣ Live inflation: the nation vs. Philadelphia")
        infl_us = df_macro["Inflation (YoY %)"].dropna()
        fig_i = go.Figure()
        fig_i.add_trace(go.Scatter(x=infl_us.index, y=infl_us.values, name="U.S. CPI Inflation (YoY %)",
                                   mode="lines", line=dict(width=3, color="#0068C9")))
        has_philly_cpi = "Philly Inflation (YoY %)" in df_macro.columns and df_macro["Philly Inflation (YoY %)"].notna().sum() > 6
        if has_philly_cpi:
            infl_ph = df_macro["Philly Inflation (YoY %)"].dropna()
            fig_i.add_trace(go.Scatter(x=infl_ph.index, y=infl_ph.values, name="Philadelphia CPI Inflation (YoY %)",
                                       mode="lines", line=dict(width=3, color="#d62728")))
        fig_i.add_hline(y=2.0, line_dash="dash", annotation_text="Fed's 2% target")
        st.plotly_chart(std_layout(fig_i, "Inflation, Year over Year (FRED: CPIAUCSL, CUURS12BSA0)",
                                   500, "Date", "YoY % change"), use_container_width=True, key="infl_live")
        latest_us = float(infl_us.iloc[-1])
        lines = [f"* National inflation is currently about **{latest_us:.1f}%** — prices are {abs(latest_us):.1f}% "
                 f"{'higher' if latest_us >= 0 else 'lower'} than one year ago."]
        if has_philly_cpi:
            latest_ph = float(df_macro['Philly Inflation (YoY %)'].dropna().iloc[-1])
            gap_ph = latest_ph - latest_us
            lines.append(f"* Philadelphia-area inflation is about **{latest_ph:.1f}%** — "
                         f"**{abs(gap_ph):.1f} points {'above' if gap_ph > 0 else 'below'}** the national rate. "
                         "Inflation is an average; *your* inflation depends on *your* basket and *your* city.")
        lines.append("* The dashed line is the Federal Reserve's **2% target** — the pace the Fed considers healthy.")
        st.markdown("**How to read this chart**\n\n" + "\n".join(lines))

        st.subheader("3️⃣ Purchasing-power calculator")
        amt = st.number_input("Dollars in your pocket today ($)", 10.0, 100000.0, 1000.0, 10.0, key="infl_amt")
        pi = st.slider("Assumed inflation rate (%)", 0.0, 12.0, max(0.0, round(latest_us, 1)), 0.5, key="infl_pi") / 100
        t_ = st.slider("Years from now", 1, 30, 10, key="infl_t")
        real_val = amt / (1 + pi) ** t_
        st.latex(r"\text{Real Value} = \frac{\text{Nominal}}{(1+\pi)^t}")
        st.metric(f"What \${amt:,.0f} buys in {t_} years (today's dollars)", f"${real_val:,.0f}",
                  delta=f"-{(1 - real_val/amt)*100:.0f}% purchasing power")

    why_care("Percent change is inflation, a raise, a markup, a discount, an interest rate, and the first step "
             "toward elasticity. One formula, half of economics.", "infl")

# ----------------------
# Tab 3: Supply & Demand (Graphs, slopes, simultaneous equations)
# ----------------------
if page == "📈 Supply & Demand":
    st.title("📈 Supply and Demand")
    st.markdown("**Economic question:** *Why do prices move?* — the math: graphs, slopes, and solving **two equations at once**.")

    cA, cB = st.columns(2)
    with cA:
        a_d = st.slider("Demand intercept a — buyers' overall appetite", 40, 200, 100, 5, key="sd_a")
        b_d = st.slider("Demand slope b — buyers' price sensitivity", 1, 10, 4, key="sd_b")
    with cB:
        c_s = st.slider("Supply intercept c — sellers' base willingness", 0, 60, 10, 5, key="sd_c")
        d_s = st.slider("Supply slope d — sellers' price responsiveness", 1, 10, 3, key="sd_d")

    p_star = (a_d - c_s) / (b_d + d_s)
    q_star = a_d - b_d * p_star
    P = np.linspace(0, max(30.0, p_star * 1.7), 120)

    fig_sd = go.Figure()
    fig_sd.add_trace(go.Scatter(x=np.clip(a_d - b_d * P, 0, None), y=P, name="Demand: Q = a − bP",
                                mode="lines", line=dict(width=3, color="#0068C9")))
    fig_sd.add_trace(go.Scatter(x=np.clip(c_s + d_s * P, 0, None), y=P, name="Supply: Q = c + dP",
                                mode="lines", line=dict(width=3, color="#2ca02c")))
    
    # Hid redundant text from the legend for a cleaner appearance
    fig_sd.add_trace(go.Scatter(x=[q_star], y=[p_star], mode="markers+text", text=["Equilibrium"],
                                textposition="top center", marker=dict(size=14, color="#C9A227"), 
                                name="Equilibrium", showlegend=False))
                                
    st.plotly_chart(std_layout(fig_sd, "The Market for Sneakers (move the sliders!)", 520, "Quantity", "Price ($)"),
                    use_container_width=True, key="sd_fig")

    st.markdown(f"""
**How to read this chart**

* The **blue line (demand)** slopes down: at higher prices, buyers want fewer sneakers.
* The **green line (supply)** slopes up: at higher prices, sellers are happy to make more.
* They cross at the **gold dot**: the market clears at **P\\* ≈ \${p_star:.2f}** and **Q\\* ≈ {q_star:.0f} units**.
* Raise **a** (buyers got raises → appetite up) and watch the whole blue line shift **right**: at the *old* price,
  buyers now want more shoes, so the price gets bid **up**.
""")

    def sd_new():
        st.markdown("""
A market is a tug-of-war. Buyers pull prices **down**; sellers pull prices **up**.
The equilibrium is where the tug-of-war balances — nobody is left holding unsold shoes,
and nobody is left unable to buy at the going price.
""")

    def sd_math():
        say_it(r"Q_d = a - bP", "Q sub d equals a minus b times P",
               f"Every \\$1 rise in price → buyers want {b_d} fewer units. The equation *describes behavior*.")
        say_it(r"Q_s = c + dP", "Q sub s equals c plus d times P",
               "Higher prices coax more production out of sellers.")
        st.markdown("At equilibrium, quantity demanded = quantity supplied. Set the equations equal and solve:")
        st.latex(r"a - bP = c + dP \;\Rightarrow\; P^{*} = \frac{a - c}{b + d} = \frac{%d - %d}{%d + %d} \approx %.2f"
                 % (a_d, c_s, b_d, d_s, p_star))
        st.markdown("**That's it — two equations, one unknown price. You just solved a simultaneous system.**")

    def sd_challenge():
        st.markdown("""
* **Predict before you slide:** if base cost `c` falls (cheaper materials), which way do P\\* and Q\\* move? Test it.
* What does a **huge b** say about buyers? (They flee at the smallest price rise — that's *elastic* demand, coming soon.)
* Real-world: concert tickets have nearly **vertical supply** (fixed seats). Sketch what that does to price when demand jumps.
""")

    three_levels(sd_new, sd_math, sd_challenge, "sd")
    why_care("An equation is not just something you solve. It **describes behavior** — gas prices, sneaker resale, "
             "and Philly rent all obey these two lines.", "sd")

# ----------------------
# Tab 4: Marginal Thinking (Δ → derivatives, sneaking up on calculus)
# ----------------------
if page == "⚖️ Marginal":
    st.title("⚖️ What Happens Next?")
    st.markdown("**Economic question:** *Should the firm make one more unit? Should I work one more hour?* — "
                "the math: **Δ (change)**, then, very quietly, **the derivative**.")

    q0 = st.slider("Units currently produced (Q)", 10, 300, 100, key="marg_q")
    TC = lambda Q: 500 + 10 * Q + 0.05 * np.asarray(Q, dtype=float) ** 2
    delta_tc = float(TC(q0 + 1) - TC(q0))
    mc_exact = 10 + 0.10 * q0

    def marg_new():
        st.markdown(f"""
A company's total cost at **Q = {q0}** units is **\${float(TC(q0)):,.2f}**.
Make **one more** unit and total cost rises to **\${float(TC(q0+1)):,.2f}** —
an increase of **\${delta_tc:.2f}**.

That increase — the cost of *the next one* — is the **marginal cost**.
Businesses live and die by that number, not by the total.
""")

    def marg_math():
        Q = np.arange(0, 301)
        fig_m = go.Figure()
        fig_m.add_trace(go.Scatter(x=Q, y=TC(Q), mode="lines", name="Total Cost TC(Q)", line=dict(width=3, color="#0068C9")))
        fig_m.add_trace(go.Scatter(x=[q0, q0 + 1], y=[float(TC(q0)), float(TC(q0+1))], mode="markers",
                                   marker=dict(size=12, color="#d62728"), name="One more unit"))
        st.plotly_chart(std_layout(fig_m, "Total Cost curve — zoom in on 'one more unit'", 470, "Q (units)", "Total Cost ($)"),
                        use_container_width=True, key="marg_fig")
        say_it(r"MC = \frac{\Delta TC}{\Delta Q}",
               "Marginal cost equals the change in total cost divided by the change in quantity",
               f"Here: \${delta_tc:.2f} ÷ 1 unit = \${delta_tc:.2f} for the next unit.")

    def marg_challenge():
        st.markdown("The **derivative** is a more precise way of asking the *same question* — "
                    "'what happens when we make one more?' — as the step size shrinks to zero.")
        say_it(r"MC = \frac{dTC}{dQ} = 10 + 0.10\,Q",
               "Marginal cost equals d-T-C by d-Q, which equals ten plus zero point one zero Q",
               f"At Q = {q0}: exact MC = **\${mc_exact:.2f}**. Your Δ answer was **\${delta_tc:.2f}**. "
               "Nearly identical — the derivative is just Δ done perfectly.")
        price_mkt = st.slider("Market price per unit ($)", 10.0, 50.0, 28.0, 1.0, key="marg_price")
        q_opt = max(0.0, (price_mkt - 10) / 0.10)
        st.markdown(f"**Profit rule:** keep expanding while price > MC. Setting MC = price: "
                    f"10 + 0.10Q = {price_mkt:.0f} → **Q\\* = {q_opt:.0f} units**. "
                    "You just did an optimization problem — the heart of business calculus.")

    three_levels(marg_new, marg_math, marg_challenge, "marg")
    why_care("Marginal thinking decides overtime shifts, airline seat pricing, and whether a firm hires one more worker. "
             "It is the single most-used piece of calculus in economics — and you just did it with subtraction.", "marg")

# ----------------------
# Tab 5: Credit, APR & Compounding (Exponents)
# ----------------------
if page == "💳 Credit":
    st.title("💳 Credit, APR & Compounding")
    st.markdown("**Economic question:** *What does borrowing really cost?* — the math: **exponents** and exponential growth.")

    cA, cB, cC = st.columns(3)
    with cA:
        pv_c = st.slider("Amount borrowed ($)", 500, 30000, 5000, 100, key="cr_pv")
    with cB:
        apr = st.slider("APR (%)", 1.0, 36.0, 24.0, 0.5, key="cr_apr") / 100
    with cC:
        t_c = st.slider("Years the balance rides", 1, 10, 3, key="cr_t")

    fv_c = pv_c * (1 + apr) ** t_c

    def cr_new():
        st.markdown(f"""
Borrow **\${pv_c:,}** at **{apr*100:.0f}% APR** and let it ride for **{t_c} years**:
you'd owe about **\${fv_c:,.0f}**.

Notice the debt doesn't grow in a straight line. It **curves upward**, because every year
you pay interest on last year's interest. That curve is the whole story of credit.
""")

    def cr_math():
        say_it(r"FV = PV(1+r)^t",
               "F-V equals P-V times, open parenthesis, one plus r, close parenthesis, to the power t",
               "The exponent t counts how many times the whole balance gets multiplied by (1+r).")
        yrs = np.arange(0, t_c + 1)
        fig_c = go.Figure(go.Scatter(x=yrs, y=pv_c * (1 + apr) ** yrs, mode="lines+markers", line=dict(width=3, color="#d62728")))
        st.plotly_chart(std_layout(fig_c, "Your balance, year by year", 430, "Years", "Balance ($)"),
                        use_container_width=True, key="cr_fig")

    def cr_challenge():
        yrs = np.arange(0, 11)
        fig_cmp = go.Figure()
        for rate, name, color in [(0.06, "6% APR (good credit)", "#2ca02c"), (0.29, "29% APR (store card)", "#d62728")]:
            fig_cmp.add_trace(go.Scatter(x=yrs, y=pv_c * (1 + rate) ** yrs, name=name, mode="lines", line=dict(width=3, color=color)))
        st.plotly_chart(std_layout(fig_cmp, "Same loan, two universes", 450, "Years", "Balance ($)"),
                        use_container_width=True, key="cr_cmp")
        st.markdown(f"""
**How to read this chart**

* Both lines start at the same **\${pv_c:,}**. After 10 years, 6% grows to **\${pv_c*1.06**10:,.0f}**,
  while 29% explodes to **\${pv_c*1.29**10:,.0f}**.
* **Rule of 72:** doubling time ≈ 72 ÷ rate. At 29%: 72 ÷ 29 ≈ **2.5 years** to double. At 6%: **12 years**.
* Same equation, FV = PV(1+r)^t. The exponent doesn't care about your feelings — only about r and t.
""")

    three_levels(cr_new, cr_math, cr_challenge, "cr")
    why_care("This one equation explains credit-card debt, savings, retirement accounts, mortgages, student loans, "
             "and why borrowing at 29% is a different universe from borrowing at 6%. It also runs the Wealth tab — "
             "the same math that traps borrowers builds investors.", "cr")

# ----------------------
# Tab 6: College & Opportunity Cost (Present value)
# ----------------------
if page == "🎓 College":
    st.title("🎓 College & Opportunity Cost")
    st.markdown("**Economic question:** *Is college \u201cfree\u201d if tuition is covered?* — the math: opportunity cost and **present value**.")

    cA, cB = st.columns(2)
    with cA:
        wage_now = st.slider("Wage you could earn right now ($/hr)", 8.0, 30.0, 15.0, key="col_wage")
        hrs_school = st.slider("Hours/week devoted to classes & study", 5, 40, 25, key="col_hrs")
    with cB:
        yrs_school = st.slider("Years in school", 1, 6, 4, key="col_yrs")
        premium = st.slider("Extra yearly income the degree adds ($)", 0, 40000, 15000, 1000, key="col_prem")

    forgone = wage_now * hrs_school * 52 * yrs_school

    def col_new():
        st.markdown(f"""
Even with tuition fully covered, going to school means **not working** those hours.
You give up roughly **\${forgone:,.0f}** in wages over {yrs_school} years.

That is the **opportunity cost**: the value of the best alternative you gave up.
Nothing is free — not even free things.
""")

    def col_math():
        st.latex(r"\text{Opportunity Cost} = \text{Value of the Best Alternative Forgone}")
        yrs_out = np.arange(0, 31)
        no_deg = wage_now * 40 * 52 * yrs_out
        deg = np.where(yrs_out <= yrs_school, 0, (yrs_out - yrs_school) * (wage_now * 40 * 52 + premium))
        fig_col = go.Figure()
        fig_col.add_trace(go.Scatter(x=yrs_out, y=no_deg, name="Work now", mode="lines", line=dict(width=3, color="#ff7f0e")))
        fig_col.add_trace(go.Scatter(x=yrs_out, y=deg, name="Degree first", mode="lines", line=dict(width=3, color="#0068C9")))
        cross = yrs_out[np.argmax(deg > no_deg)] if np.any(deg > no_deg) else None
        st.plotly_chart(std_layout(fig_col, "Cumulative earnings: work now vs degree first", 470,
                                   "Years from today", "Cumulative earnings ($)"),
                        use_container_width=True, key="col_fig")
        if cross:
            st.markdown(f"**How to read this chart:** the lines cross at about **year {int(cross)}** — the break-even point. "
                        "Before it, working wins. After it, the degree wins **every single year**.")
        else:
            st.markdown("**How to read this chart:** with these settings, the degree path never catches up in 30 years — "
                        "raise the premium or shorten the schooling and watch the break-even appear.")

    def col_challenge():
        st.markdown("A dollar in 20 years is worth less than a dollar today. Discount future dollars back:")
        say_it(r"PV = \frac{FV}{(1+r)^t}",
               "P-V equals F-V divided by, open parenthesis, one plus r, close parenthesis, to the t",
               "Divide by the same growth factor that compounding multiplies by. PV and FV are mirror images.")
        r_d = st.slider("Discount rate (%)", 1, 10, 4, key="col_r") / 100
        pv_premium = sum(premium / (1 + r_d) ** t for t in range(yrs_school + 1, 41))
        st.metric("Present value of the degree premium (through ~age 58)", f"${pv_premium:,.0f}")
        st.markdown(f"Compare that to the **\${forgone:,.0f}** opportunity cost. Now the college decision is one line of arithmetic.")

    three_levels(col_new, col_math, col_challenge, "col")
    why_care("Opportunity cost is the single most-used idea in all of economics. It prices your **time**, "
             "not just your money — and present value lets you compare dollars across decades fairly.", "col")

# ----------------------
# Tab 7: Philadelphia Data Lab (Rates, averages, live data)
# ----------------------
if page == "🏙️ Philadelphia":
    st.title("🏙️ Philadelphia Data Lab")
    st.markdown("**Economic question:** *Can I afford this apartment — and what is my city's economy actually doing?* — "
                "the math: rates, averages, and reading real data.")

    st.subheader("1️⃣ Can I afford this apartment?")
    c1, c2, c3 = st.columns(3)
    with c1:
        ph_wage = st.slider("Hourly wage ($)", 7.25, 40.0, 16.0, 0.25, key="ph_wage")
        ph_hours = st.slider("Hours per week", 10, 60, 35, key="ph_hours")
    with c2:
        ph_rent = st.slider("Rent ($/mo)", 700, 2600, 1350, 25, key="ph_rent")
        ph_transit = st.slider("Transportation ($/mo)", 0, 400, 96, key="ph_transit")
    with c3:
        ph_food = st.slider("Food ($/mo)", 100, 900, 400, 25, key="ph_food")
        ph_loans = st.slider("Student loan payment ($/mo)", 0, 800, 150, 25, key="ph_loans")
    ph_tax = st.slider("Approx. combined tax rate (%)", 0, 35, 18, key="ph_tax") / 100

    gross_mo = ph_wage * ph_hours * 52 / 12
    net_mo = gross_mo * (1 - ph_tax)
    fixed = ph_rent + ph_transit + ph_food + ph_loans
    left = net_mo - fixed
    rent_share = ph_rent / gross_mo * 100

    st.latex(r"\text{Monthly Income} = \text{Wage} \times \text{Hours} \times \tfrac{52}{12}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Gross / month", f"${gross_mo:,.0f}")
    m2.metric("Take-home / month", f"${net_mo:,.0f}")
    m3.metric("Fixed costs / month", f"${fixed:,.0f}")
    m4.metric("Left over", f"${left:,.0f}")

    if left < 0:
        st.error(f"This budget is short by **\${-left:,.0f}/month**. What's the smallest change that fixes it — "
                 "hours, rent, or wage? (That question is marginal thinking from the ⚖️ tab.)")
    elif rent_share > 30:
        st.warning(f"Rent is **{rent_share:.0f}%** of gross income. Landlords and lenders flag anything over **~30%** — "
                   "you'd likely need a co-signer or a cheaper unit.")
    else:
        st.success(f"Rent is **{rent_share:.0f}%** of gross income — under the 30% guideline. This budget clears.")

    if HAS_DATA:
        st.subheader("2️⃣ Philadelphia vs. the nation: unemployment")
        ph_u = df_macro.get("Philly_Unemp", pd.Series(dtype=float)).dropna()
        us_u = df_macro.get("UNRATE", pd.Series(dtype=float)).dropna()
        if not ph_u.empty and not us_u.empty:
            fig_u = go.Figure()
            fig_u.add_trace(go.Scatter(x=us_u.index, y=us_u.values, name="U.S. (UNRATE)", mode="lines",
                                       line=dict(width=3, color="#0068C9")))
            fig_u.add_trace(go.Scatter(x=ph_u.index, y=ph_u.values, name="Philadelphia County (PAPHIL5URN)", mode="lines",
                                       line=dict(width=3, color="#d62728")))
            st.plotly_chart(std_layout(fig_u, "Unemployment Rate: Philadelphia County vs. United States", 500,
                                       "Date", "Unemployment rate (%)"), use_container_width=True, key="ph_unemp")
            last_common = ph_u.index.intersection(us_u.index).max()
            if pd.notna(last_common):
                ph_last, us_last = float(ph_u.loc[last_common]), float(us_u.loc[last_common])
                gap_u = ph_last - us_last
                st.markdown(f"""
**How to read this chart**

* As of **{last_common.date()}**, Philadelphia County unemployment is about **{ph_last:.1f}%**, versus **{us_last:.1f}%** nationally —
  a gap of **{gap_u:+.1f} percentage points**.
* A rate is a fraction: unemployed ÷ labor force × 100. **Who counts** in each part matters — discouraged workers
  who stopped searching disappear from the numerator *and* the denominator.
* Citywide averages hide neighborhood variation. The number for your block may look nothing like {ph_last:.1f}%.
""")
        else:
            st.info("Philadelphia unemployment series unavailable in this window — widen the date range or check the FRED key.")

        payrolls = df_macro.get("Philly_Payrolls", pd.Series(dtype=float)).dropna()
        if not payrolls.empty and len(payrolls) > 13:
            st.subheader("3️⃣ Philadelphia-area jobs (nonfarm payrolls)")
            fig_p = go.Figure(go.Scatter(x=payrolls.index, y=payrolls.values, mode="lines", line=dict(width=3, color="#2ca02c")))
            st.plotly_chart(std_layout(fig_p, "Nonfarm Payrolls, Philadelphia MSA (FRED: PHIL942NA)", 450,
                                       "Date", "Thousands of jobs"), use_container_width=True, key="ph_pay")
            yoy_jobs = (payrolls.iloc[-1] / payrolls.iloc[-13] - 1) * 100
            st.markdown(f"**How to read this chart:** the metro area currently has about **{payrolls.iloc[-1]:,.0f} thousand jobs**, "
                        f"a change of **{yoy_jobs:+.1f}%** over the past year. Percent change again — the same formula as your rent.")
    else:
        st.info("Add a FRED_API_KEY in secrets to unlock live Philadelphia unemployment, payroll, and CPI data.")

    why_care("This is your budget, your city, your data. Rates, averages, and percent-of-income rules "
             "decide real leases and real loans — and reading them is a job skill.", "ph")

# ----------------------
# Tab 8: Wealth Gap Simulator (Annuities, geometric series)
# ----------------------
if page == "💰 Wealth":
    st.title("💰 The Wealth Gap Simulator")
    st.markdown("**Two 18-year-olds. Identical incomes. One difference: *when they start.*** — the math: the annuity formula (a geometric series in disguise).")

    cA, cB = st.columns(2)
    with cA:
        pmt_w = st.slider("Monthly contribution ($)", 10, 500, 50, 10, key="w_pmt")
        delay_yrs = st.slider("Student B starts how many years later?", 1, 20, 10, key="w_delay")
    with cB:
        r_w = st.slider("Annual investment return (%)", 1.0, 12.0, 7.0, 0.5, key="w_r") / 100
        infl_w = st.slider("Inflation (%)", 0.0, 6.0, 2.5, 0.5, key="w_infl") / 100

    months = np.arange(0, 47 * 12)  # to age 65
    rm = r_w / 12

    def wealth_path(delay_months):
        w = np.zeros_like(months, dtype=float)
        for t in range(1, len(months)):
            w[t] = w[t - 1] * (1 + rm) + (pmt_w if t >= delay_months else 0)
        return w

    wa, wb = wealth_path(0), wealth_path(delay_yrs * 12)
    deflator = (1 + infl_w) ** (months / 12)
    fig_w = go.Figure()
    fig_w.add_trace(go.Scatter(x=18 + months / 12, y=wa / deflator, name="Student A (starts at 18)",
                               mode="lines", line=dict(width=3, color="#0068C9")))
    fig_w.add_trace(go.Scatter(x=18 + months / 12, y=wb / deflator, name=f"Student B (starts at {18+delay_yrs})",
                               mode="lines", line=dict(width=3, color="#d62728")))
    st.plotly_chart(std_layout(fig_w, "Real wealth to age 65 (today's dollars)", 520, "Age", "Real wealth ($)"),
                    use_container_width=True, key="w_fig")

    gap_w = (wa[-1] - wb[-1]) / deflator[-1]
    contrib_a = pmt_w * len(months)
    contrib_b = pmt_w * (len(months) - delay_yrs * 12)
    st.markdown(f"""
**How to read this chart**

* Student A contributes **\${contrib_a:,.0f}** total; Student B contributes **\${contrib_b:,.0f}** — only
  **\${contrib_a-contrib_b:,.0f} less**.
* Yet at 65 the head start is worth **\${gap_w:,.0f}** in today's dollars. The gap comes overwhelmingly from **time**,
  not from money: early dollars get compounded the most times.
* Now connect this to history: communities **blocked** from early asset-building (redlining, exclusion from GI Bill benefits)
  didn't just lose the blocked dollars. They lost every compounding period those dollars would have earned.
""")

    def w_new():
        st.markdown("Wait — B put in *almost as much money*. Where did the gap come from? **Time.** "
                    "Compounding pays interest on interest, and the earliest dollars collect the longest.")

    def w_math():
        say_it(r"FV = PMT\left(\frac{(1+r)^n - 1}{r}\right)",
               "F-V equals P-M-T times, open parenthesis, one plus r to the n, minus one, all over r, close parenthesis",
               "\u201cWait — so *that's* how the computer calculated it?\u201d Exactly. Every monthly deposit is its own "
               "little FV = PV(1+r)^t problem; this formula adds them all up in one line.")

    def w_challenge():
        target = wa[-1]
        lo, hi = pmt_w, pmt_w * 50
        for _ in range(60):
            mid = (lo + hi) / 2
            w = 0.0
            for t in range(1, len(months)):
                w = w * (1 + rm) + (mid if t >= delay_yrs * 12 else 0)
            if w < target:
                lo = mid
            else:
                hi = mid
        st.markdown(f"""
For Student B to **catch** Student A by 65, B must contribute about **\${hi:,.0f}/month** — that is
**{hi/pmt_w:.1f}×** Student A's contribution. That multiple is the **price of waiting {delay_yrs} years**.
(The app found it by *binary search* — guess, check, halve the interval. That's an algorithm.)
""")

    three_levels(w_new, w_math, w_challenge, "w")
    why_care("Starting age, not income, drives most of this gap — one honest mathematical lens on how "
             "wealth gaps compound across generations, and why starting small NOW beats starting big later.", "w")

# ----------------------
# Tab 9: Math Translator (notation → plain English, optional AI)
# ----------------------
if page == "🧮 Translator":
    st.title("🧮 Math Translator")
    st.markdown("Point at notation. Get **plain English**, the **meaning**, and **how to say it out loud**.")

    BUILTIN = {
        "Q_d = 100 − 4P": (r"Q_d = 100 - 4P",
            "Quantity demanded equals 100 minus four times the price.",
            "Every time price rises by \\$1, consumers buy 4 fewer units.",
            "\u201cQ sub d equals one hundred minus four P.\u201d"),
        "ΔQ_d / ΔP": (r"\frac{\Delta Q_d}{\Delta P}",
            "Change in quantity demanded divided by change in price.",
            "How strongly buyers react when the price moves — the seed of elasticity.",
            "\u201cDelta Q sub d over delta P.\u201d"),
        "FV = PV(1+r)^t": (r"FV = PV(1+r)^t",
            "Future value equals present value times one-plus-the-rate, raised to the number of periods.",
            "Money left alone grows (or debt grows) by the same multiplier every period.",
            "\u201cF-V equals P-V times one plus r, to the t.\u201d"),
        "MC = dTC/dQ": (r"MC = \frac{dTC}{dQ}",
            "Marginal cost equals the derivative of total cost with respect to quantity.",
            "The cost of making exactly one more unit.",
            "\u201cM-C equals d-T-C by d-Q.\u201d"),
        "%ΔP = (P₁−P₀)/P₀ × 100": (r"\%\Delta P = \frac{P_1 - P_0}{P_0}\times 100",
            "Percent change equals new minus old, divided by old, times one hundred.",
            "Change relative to the starting point, per hundred. Inflation, raises, markups — all this.",
            "\u201cPercent change in P equals P-one minus P-naught, over P-naught, times a hundred.\u201d"),
        "PV = FV/(1+r)^t": (r"PV = \frac{FV}{(1+r)^t}",
            "Present value equals future value divided by one-plus-the-rate to the t.",
            "Shrink future dollars back to today. The mirror image of compounding.",
            "\u201cP-V equals F-V over one plus r, to the t.\u201d"),
    }

    choice = st.selectbox("Pick an expression", list(BUILTIN), key="tr_choice")
    latex_str, words, meaning, spoken = BUILTIN[choice]
    st.latex(latex_str)
    if st.button("Translate This", key="tr_btn"):
        st.markdown(f"**In words:** {words}")
        st.markdown(f"**What it means:** {meaning}")
        st.markdown(f"**How to say it out loud:** {spoken}")

    st.divider()
    st.subheader("Ask about ANY expression (AI-powered)")
    if ANTHROPIC_API_KEY:
        custom = st.text_input("Type any economic expression — e.g.  P = MC,  or  e = %ΔQ / %ΔP", key="tr_custom")
        level = st.radio("Explain it like…", ["I'm brand new", "Show me the math", "Challenge me"],
                         horizontal=True, key="tr_level")
        if custom and st.button("Ask the Translator", key="tr_ai_btn"):
            with st.spinner("Translating…"):
                try:
                    resp = requests.post(
                        "https://api.anthropic.com/v1/messages",
                        headers={"x-api-key": ANTHROPIC_API_KEY,
                                 "anthropic-version": "2023-06-01",
                                 "content-type": "application/json"},
                        json={"model": "claude-sonnet-4-6", "max_tokens": 700,
                              "system": ("You are the Math Translator inside an Econ I teaching app at Cheyney "
                                         "University, an HBCU. Explain economic/mathematical notation in plain "
                                         "language at the requested level (brand new = 7th–9th grade). Always give: "
                                         "the phonetic reading, the meaning, and one everyday example a Philadelphia "
                                         "college student would recognize. Never solve homework problems or produce "
                                         "graded answers — explain notation only."),
                              "messages": [{"role": "user", "content": f"Level: {level}\n\nExplain: {custom}"}]},
                        timeout=30)
                    resp.raise_for_status()
                    out = "".join(b.get("text", "") for b in resp.json().get("content", []))
                    st.markdown(out if out else "The translator returned nothing — try the built-in expressions above.")
                except Exception:
                    st.warning("The translator is unavailable right now — the built-in expressions above still work.")
    else:
        st.caption("🟡 Add ANTHROPIC_API_KEY in secrets to unlock free-form translation of any expression.")

    why_care("Notation anxiety is the #1 wall between students and economics. Once you can **read** an equation "
             "out loud, it stops being scary — it becomes a sentence.", "tr")

# ----------------------
# Tab 10: Build an Economy (Functions feeding functions, simulation)
# ----------------------
if page == "🎮 Economy":
    st.title("🎮 Build an Economy")
    st.markdown("**Economic question:** *What happens if wages, taxes, prices, or interest rates change?* — "
                "the math: **functions feeding functions** (a simulation).")

    c1, c2, c3 = st.columns(3)
    with c1:
        wage_g = st.slider("Annual wage growth (%)", 0.0, 8.0, 3.0, 0.5, key="ec_wg") / 100
        infl_e = st.slider("Inflation (%)", 0.0, 10.0, 3.0, 0.5, key="ec_infl") / 100
    with c2:
        tax_e = st.slider("Flat tax rate (%)", 0, 40, 15, key="ec_tax") / 100
        rate_e = st.slider("Interest on savings (%)", 0.0, 8.0, 4.0, 0.5, key="ec_rate") / 100
    with c3:
        save_share = st.slider("Share of after-tax income saved (%)", 0, 30, 10, key="ec_save") / 100
        start_income = st.slider("Starting income ($/yr)", 25000, 80000, 40000, 5000, key="ec_inc")

    yrs_e = np.arange(0, 21)
    income_e = start_income * (1 + wage_g) ** yrs_e
    after_tax = income_e * (1 - tax_e)
    real_after_tax = after_tax / (1 + infl_e) ** yrs_e
    wealth_e = np.zeros_like(yrs_e, dtype=float)
    for t in yrs_e[1:]:
        wealth_e[t] = wealth_e[t - 1] * (1 + rate_e) + after_tax[t] * save_share

    fig_e = go.Figure()
    fig_e.add_trace(go.Scatter(x=yrs_e, y=after_tax, name="After-tax income (nominal $)", mode="lines", line=dict(width=3, color="#0068C9")))
    fig_e.add_trace(go.Scatter(x=yrs_e, y=real_after_tax, name="REAL after-tax income (today's $)", mode="lines", line=dict(width=3, color="#d62728")))
    fig_e.add_trace(go.Scatter(x=yrs_e, y=wealth_e, name="Accumulated savings", mode="lines", line=dict(width=3, color="#2ca02c")))
    st.plotly_chart(std_layout(fig_e, "Your 20-year economy", 520, "Years", "$"), use_container_width=True, key="ec_fig")

    real_growth = (1 + wage_g) / (1 + infl_e) - 1
    verdict = ("**gaining** ground — wages are outrunning prices" if real_growth > 0.001
               else "**losing** ground — prices are outrunning wages" if real_growth < -0.001
               else "**treading water** — wages and prices are tied")
    st.markdown(f"""
**How to read this chart**

* The **blue line** (nominal income) almost always rises — that's just the number on the paycheck.
* The **red line** is what that paycheck actually *buys*. With wage growth of {wage_g*100:.1f}% and inflation of
  {infl_e*100:.1f}%, real income grows about **{real_growth*100:+.1f}%/year** — you are {verdict}.
* The **green line** compounds: savings feed on the interest rate AND on income growth. Every slider here is a
  **function**, and the chart is functions feeding functions. That is how the CBO, the Fed, and every fintech app
  model the future.
* Try it: set inflation above wage growth and watch the red line sink while the blue line rises. That single picture
  explains most political anger about the economy.
""")

    export_df = pd.DataFrame({"Year": yrs_e, "Nominal After-Tax Income": after_tax.round(0),
                              "Real After-Tax Income": real_after_tax.round(0), "Savings": wealth_e.round(0)})
    st.download_button("⬇️ Download my economy (CSV)", export_df.to_csv(index=False).encode("utf-8"),
                       "my_economy.csv", "text/csv", key="ec_dl")
    why_care("Congress, the Fed, and every retirement calculator run versions of exactly this loop. "
             "You just built a structural model — the kind professionals get paid for.", "ec")

# ----------------------
# Tab 11: Instructor Dashboard
# ----------------------
if page == "🧑‍🏫 Instructor":
    st.title("🧑‍🏫 Instructor Dashboard")
    pw = st.text_input("Instructor passcode", type="password", key="instr_pw")
    expected = st.secrets.get("INSTRUCTOR_PASSCODE", os.getenv("INSTRUCTOR_PASSCODE", ""))
    if not expected:
        st.warning("Set INSTRUCTOR_PASSCODE in secrets to protect this page.")
        authorized = True
    else:
        authorized = (pw == expected)
        if pw and not authorized:
            st.error("Incorrect passcode.")

    if authorized:
        df_conf, source = load_confidence()
         if authorized:
        df_conf, source = load_confidence()

        with st.expander("🔧 Test Google Sheets connection"):
            if not sheets_configured():
                st.error("Secrets missing: need GSHEET_ID and [gcp_service_account].")
            else:
                try:
                    ws = get_worksheet()
                    st.success(f"Connected to tab '{ws.title}'.")
                except Exception as e:
                    st.error(f"{type(e).__name__}: {e}")
        
        # Safely extract the source badge using .get() to prevent KeyErrors
        badge_options = {   
        
        # Safely extract the source badge using .get() to prevent KeyErrors
        badge_options = {
            "sheets": "🟢 Live from Google Sheets",
            "csv/memory": "🟡 Local Memory & CSV (ephemeral on Streamlit Cloud)",
            "csv": "🟡 Local CSV (ephemeral on Streamlit Cloud)",
            "memory": "🟡 Local Memory (ephemeral on Streamlit Cloud)",
            "none": "⚪ No data source configured"
        }
        badge = badge_options.get(source, "⚪ No data source configured")
        
        cS1, cS2 = st.columns([3, 1])
        cS1.caption(badge)
        if cS2.button("🔄 Refresh", key="instr_refresh"):
            st.rerun()
            
        if df_conf.empty:
            st.info("No confidence data logged yet. Data appears here as students use the sidebar tracker.")
        else:
            st.metric("Total responses", len(df_conf))
            pivot = df_conf.groupby(["lab", "when"])["score"].agg(["mean", "count"])
            means = pivot["mean"].unstack().reindex(columns=["before", "after"])
            if {"before", "after"}.issubset(means.columns):
                means["Δ (after − before)"] = means["after"] - means["before"]
            st.dataframe(means.round(2), use_container_width=True)

            labs_present = sorted(df_conf["lab"].unique())
            fig_conf = go.Figure()
            for when, color in [("before", "#ff7f0e"), ("after", "#2ca02c")]:
                sub = df_conf[df_conf["when"] == when].groupby("lab")["score"].mean().reindex(labs_present)
                fig_conf.add_trace(go.Bar(name=when.title(), x=labs_present, y=sub.values, marker_color=color))
            fig_conf.update_layout(barmode="group", template="plotly_white", height=480, font=dict(size=14),
                                   title=dict(text="Mean Math Confidence: before vs after, by lab", font=dict(size=18)),
                                   yaxis_title="Mean confidence (1–5)",
                                   legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1))
            st.plotly_chart(fig_conf, use_container_width=True, key="instr_fig")

            st.download_button("⬇️ Download raw confidence CSV", df_conf.to_csv(index=False).encode("utf-8"),
                               "confidence_log.csv", "text/csv", key="instr_dl")
        st.caption("Research note: with Google Sheets configured, responses persist across redeploys and you can "
                   "watch them arrive live in the Sheet. Before treating before/after deltas as publishable SoTL "
                   "data, secure IRB approval; anonymous formative course-improvement use is standard practice.")
