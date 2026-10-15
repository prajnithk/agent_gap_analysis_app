"""
Agent Coaching Insights - Streamlit app.

    streamlit run app.py

Everything runs locally on your machine; nothing is sent anywhere.
"""
import copy
import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import demo
import engine as E

APP_DIR = Path(__file__).parent
SETTINGS_FILE = APP_DIR / "settings.json"
FILE_KEYS = {"cases": "cases", "case": "cases", "case_comments": "comments", "casecomment": "comments",
             "casecomments": "comments", "comments": "comments", "case_history": "history",
             "casehistory": "history", "history": "history", "users": "users", "user": "users"}

st.set_page_config(page_title="Agent Gap Analysis", page_icon="📊", layout="wide")

# ---------------------------------------------------------------------------
# Theme (light / dark are separate, validated palettes)
# ---------------------------------------------------------------------------
try:
    DARK = st.context.theme.type == "dark"
except Exception:
    DARK = False
P = ({"series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
      "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
      "slow": "#e66767", "fast": "#3987e5",
      "div": [[0, "#6da7ec"], [.25, "#256abf"], [.5, "#383835"], [.75, "#a83a3a"], [1, "#e66767"]],
      "seq": [[0, "#1d2b3d"], [.5, "#256abf"], [1, "#9ec5f4"]], "seq_text_flip": .55}
     if DARK else
     {"series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
      "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
      "slow": "#d03b3b", "fast": "#2a78d6",
      "div": [[0, "#1c5cab"], [.25, "#86b6ef"], [.5, "#f0efec"], [.75, "#f0a3a2"], [1, "#c23b3a"]],
      "seq": [[0, "#e8f1fc"], [.5, "#5598e7"], [1, "#0d366b"]], "seq_text_flip": .5})
HALF_HOURS = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)] + ["24:00"]
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"
PLOT_CFG = {"displayModeBar": False}


def style(fig, height=360, legend=True):
    fig.update_layout(
        height=height, margin=dict(l=8, r=16, t=30 if legend else 8, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=13, color=P["ink2"]),
        hoverlabel=dict(font=dict(family=FONT, size=12)),
        showlegend=legend, legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title=None,
                                       font=dict(color=P["ink2"])),
    )
    fig.update_xaxes(automargin=True, gridcolor=P["grid"], linecolor=P["axis"], zerolinecolor=P["axis"],
                     tickfont=dict(color=P["muted"]), title_font=dict(color=P["ink2"], size=12))
    fig.update_yaxes(automargin=True, gridcolor=P["grid"], linecolor=P["axis"], zerolinecolor=P["axis"],
                     tickfont=dict(color=P["muted"]), title_font=dict(color=P["ink2"], size=12))
    return fig


def chart_title(text, note=None):
    st.markdown(f"**{text}**" + (f"  \n<span style='font-size:0.85rem;opacity:.7'>{note}</span>" if note else ""),
                unsafe_allow_html=True)


def show(fig, key=None):
    st.plotly_chart(fig, theme=None, config=PLOT_CFG, key=key)


def cell_ink(share):
    """Readable text on a sequential cell (the ramp runs light->dark in light mode, dark->light in dark)."""
    if DARK:
        return "#0b0b0b" if share > .6 else P["ink"]
    return "#ffffff" if share > .5 else P["ink"]


def wrap(text, width=14):
    words, lines, cur = str(text).split(), [], ""
    for w in words:
        if cur and len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    return "<br>".join(lines + [cur])


def pct(v, plus=True):
    if v is None or pd.isna(v):
        return "–"
    return f"{v:+.0%}" if plus else f"{v:.0%}"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Generating demo data…")
def load_demo():
    return demo.make_demo()


@st.cache_data(show_spinner=False, max_entries=20)
def read_csv_bytes(b: bytes):
    df = pd.read_csv(io.BytesIO(b), dtype=str, keep_default_na=False, na_values=[""],
                     encoding_errors="replace")
    return E.normalize_columns(df)


def table_key(filename: str):
    stem = Path(filename).stem.lower().replace(" ", "_").replace("-", "_")
    return FILE_KEYS.get(stem)


def from_uploads(files):
    tables, h, unknown = {}, hashlib.md5(), []
    for f in sorted(files, key=lambda f: f.name):
        k = table_key(f.name)
        if not k:
            unknown.append(f.name)
            continue
        b = f.getvalue()
        h.update(f.name.encode() + b)
        tables[k] = read_csv_bytes(b)
    return tables, h.hexdigest(), unknown


def from_folder(folder: str):
    p = Path(folder).expanduser()
    if not p.is_dir():
        st.error(f"Folder not found: {p}")
        return None, None, []
    tables, h = {}, hashlib.md5()
    for f in sorted(p.glob("*.csv")):
        k = table_key(f.name)
        if k:
            stat = f.stat()
            h.update(f"{f}{stat.st_mtime}{stat.st_size}".encode())
            tables[k] = read_csv_bytes(f.read_bytes())
    return tables, h.hexdigest(), []


def settings_key(S):
    return json.dumps({k: v for k, v in S.items() if k != "topic_names"}, sort_keys=True, default=str)


@st.cache_data(show_spinner="Reading SLAs and finding comment topics…", max_entries=6)
def run_prepare(fp, skey, _tables):
    return E.prepare(_tables, json.loads(skey))


@st.cache_data(show_spinner="Fitting models…", max_entries=12)
def run_analyze(fp, skey, sla, _d, _meta):
    return E.analyze(_d, _meta, json.loads(skey), sla)


@st.cache_data(show_spinner=False, max_entries=6)
def run_agent_sla(fp, skey, _d, _meta):
    return E.agent_sla_table(_d, _meta, json.loads(skey))


# ---------------------------------------------------------------------------
# Session settings
# ---------------------------------------------------------------------------
if "settings" not in st.session_state:
    st.session_state.settings = {"demo": demo.demo_settings(), "real": E.load_settings(SETTINGS_FILE)}
    st.session_state.ver = {"demo": 0, "real": 0}

with st.sidebar:
    st.header("Agent gap analysis")
    source = st.radio("Data", ["Demo data", "Upload CSV files", "Folder on this computer"], key="source",
                      help="Everything stays on this computer. Nothing is uploaded to the internet.")
    tables, fp, unknown = None, None, []
    if source == "Demo data":
        tables, fp = load_demo(), "demo-v3"
        st.caption("Synthetic data with planted weak spots (see the Overview tab).")
    elif source == "Upload CSV files":
        files = st.file_uploader("Drop your exports here", type="csv", accept_multiple_files=True,
                                 help="Name them cases.csv, case_comments.csv, case_history.csv, users.csv")
        if files:
            tables, fp, unknown = from_uploads(files)
    else:
        folder = st.text_input("Folder with the CSV exports", key="folder",
                               placeholder="/Users/you/Documents/case_exports")
        if folder:
            tables, fp, unknown = from_folder(folder)
    if unknown:
        st.warning("Not recognised (check the file names): " + ", ".join(unknown))

mode = "demo" if source == "Demo data" else "real"
S = st.session_state.settings[mode]
ver = st.session_state.ver[mode]


def tn(label):
    return S.get("topic_names", {}).get(str(label)) or str(label)


def short(label):
    t = tn(label)
    return t.split(": ", 1)[-1] if t == str(label) else t


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------
SOQL = pd.DataFrame([
    ("cases.csv", "Required",
     "SELECT Id, OwnerId, Owner.Name, Type, Origin, Priority, Status, CreatedDate, ClosedDate, "
     "<your SLA fields>, <your business-hours-to-close field>, <Department / LOB fields> "
     "FROM Case WHERE CreatedDate = LAST_N_MONTHS:6"),
    ("case_comments.csv", "Recommended",
     "SELECT ParentId, CommentBody FROM CaseComment WHERE ParentId IN (SELECT Id FROM Case WHERE CreatedDate = LAST_N_MONTHS:6)"),
    ("case_history.csv", "Optional",
     "SELECT CaseId, Field, OldValue, NewValue, CreatedDate FROM CaseHistory WHERE Field = 'Status' AND CaseId IN "
     "(SELECT Id FROM Case WHERE CreatedDate = LAST_N_MONTHS:6)  (only used when closure time comes from the dates)"),
    ("users.csv", "Optional", "SELECT Id, Name FROM User WHERE IsActive = true (add a hire-date field if you have one)"),
], columns=["File", "Need", "SOQL"])

if not tables or "cases" not in tables:
    st.title("Agent gap analysis")
    st.write("Find which agents need coaching on which topics, using case comments, closure time and your SLA "
             "fields. Start with **Demo data** in the sidebar, or load your exports.")
    st.subheader("Exports to pull (Data Loader or Workbench)")
    st.dataframe(SOQL, hide_index=True, column_config={"SOQL": st.column_config.TextColumn(width="large")})
    st.caption("Header names are matched without regard to case, so Data Loader's upper-case headers work. "
               "Dates are read as Mountain Time. Exports contain PHI: keep them on an approved, encrypted machine.")
    st.stop()

cases_raw = tables["cases"]
if mode == "real" and not SETTINGS_FILE.exists() and not st.session_state.get(f"autodetected_{fp}"):
    # first look at this data: suggest SLA fields and the closure-time field, then confirm in Setup
    st.session_state[f"autodetected_{fp}"] = True
    if not S.get("slas"):
        S["slas"] = E.auto_detect_slas(cases_raw)
        if S["slas"]:
            st.toast(f"Found {len(S['slas'])} SLA field(s). Check them in the Setup tab.")
    if not S.get("closure_hours_field"):
        S["closure_hours_field"] = E.suggest_hours_field(cases_raw)


# ---------------------------------------------------------------------------
# Setup tab (also shown on its own when the data can't be processed)
# ---------------------------------------------------------------------------
def clean_sla(r):
    """Plain JSON-safe strings for one SLA row from the editor."""
    def s(v):
        return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip()
    return {"name": s(r.get("name")), "method": s(r.get("method")) or E.SLA_METHODS[0],
            "field": s(r.get("field")), "breach_values": s(r.get("breach_values")), "target": s(r.get("target"))}


def render_setup(tables, S, mode, ver):
    cases = tables["cases"]
    cols = list(cases.columns)
    opt = ["—"] + cols
    st.subheader("Setup")
    st.caption("Changes apply when you press **Apply settings**." +
               (f" Your settings are saved to `{SETTINGS_FILE.name}` next to the app." if mode == "real"
                else " Demo settings are not saved.") + " Dates are read as Mountain Time.")

    found = E.sla_fields(cases)
    with st.expander(f"SLA-looking fields in cases.csv ({len(found)})", expanded=not S.get("slas")):
        if found.empty:
            st.write("No column names contain 'SLA'. Add your SLA fields to the cases export.")
        else:
            st.dataframe(found, hide_index=True)

    k = f"{mode}_{ver}_"
    with st.form(f"setup_{mode}_{ver}", border=False):
        st.markdown("##### SLAs (read from your Case fields)")
        st.caption("Your org already calculates these in business hours; the app only reads them. "
                   "**Met/Missed field**: missed when the field has one of the values you list (for example "
                   "`missed`, or `false` for an SLA Met checkbox). **Due date field**: missed when ClosedDate is "
                   "after the due date. **Hours field vs target**: missed when the hours in the field exceed the "
                   "target (a number, or a field that holds the target). Blank values mean the SLA doesn't apply "
                   "to that case.")
        sla_df = pd.DataFrame([dict(E.SLA_TEMPLATE, **s) for s in S.get("slas", [])] or [E.SLA_TEMPLATE])
        sla_df = sla_df[list(E.SLA_TEMPLATE)]
        sla_edit = st.data_editor(
            sla_df, key=k + "slas", num_rows="dynamic", hide_index=True,
            column_config={
                "name": st.column_config.TextColumn("SLA name", required=True),
                "method": st.column_config.SelectboxColumn("Method", options=E.SLA_METHODS, required=True),
                "field": st.column_config.SelectboxColumn("Case field", options=[""] + cols),
                "breach_values": st.column_config.TextColumn("Values that mean missed",
                                                             help="Met/Missed only. Comma separated, not case sensitive."),
                "target": st.column_config.TextColumn("Target hours (number or field)",
                                                      help="Hours field vs target only."),
            })

        st.markdown("##### Closure time")
        num_cols = [c for c in cols if E.detect_kind(cases[c]) == "number"]
        cur = S.get("closure_hours_field") or ""
        options = [""] + sorted(set(num_cols) | ({cur} if cur in cols else set()))
        hours_field = st.selectbox(
            "Business hours to close (your field)", options, key=k + "hours",
            index=options.index(cur) if cur in options else 0,
            format_func=lambda x: x or "None: use CreatedDate to ClosedDate",
            help="Pick the Case field where your org stores business hours from open to close.")
        hist = tables.get("history")
        statuses = set(hist["NewValue"].dropna().astype(str)) if hist is not None and "NewValue" in hist else set()
        waiting = st.multiselect("Only when using the dates: statuses to leave out (waiting on provider/member)",
                                 sorted(statuses | set(S["waiting_statuses"])),
                                 default=S["waiting_statuses"], key=k + "waiting",
                                 help="Needs case_history.csv. Ignored when you pick a business-hours field.")

        st.markdown("##### Case fields")
        c = st.columns(4)
        roles = [("case_id", "Case Id"), ("agent", "Agent (owner Id)"), ("agent_name", "Agent name (optional)"),
                 ("created", "Created date"), ("closed", "Closed date"), ("case_type", "Case type"),
                 ("channel", "Channel"), ("priority", "Priority")]
        picked = {}
        for i, (role, label) in enumerate(roles):
            cur = S["columns"].get(role)
            picked[role] = c[i % 4].selectbox(label, opt, index=opt.index(cur) if cur in opt else 0,
                                              key=k + "col_" + role)
        mix = st.multiselect("Extra case-mix fields (for example Department, Line of business)",
                             [x for x in cols if x not in picked.values()],
                             default=[x for x in S.get("extra_case_mix", []) if x in cols], key=k + "mix",
                             help="Fields that make a case harder or easier, outside the agent's control.")

        st.markdown("##### Comment topics and behavior keywords")
        n_topics = st.slider("Number of topics to find in comments", 3, 15, int(S["n_topics"]), key=k + "ntop")
        flags_df = pd.DataFrame({"name": list(S["flags"]), "pattern": list(S["flags"].values())})
        flags_edit = st.data_editor(flags_df, key=k + "flags", num_rows="dynamic", hide_index=True,
                                    column_config={"name": st.column_config.TextColumn("Behavior"),
                                                   "pattern": st.column_config.TextColumn(
                                                       "Keywords (regular expression)", width="large")})

        st.markdown("##### Thresholds")
        T = S["thresholds"]
        t = st.columns(5)
        gap = t[0].number_input("Gap: slower than expected by (%)", 5, 200, int(T["gap"] * 100), key=k + "gap")
        spec = t[1].number_input("Topic-specific: slower than own pace by (%)", 5, 200,
                                 int(T["specific"] * 100), key=k + "spec")
        min_cell = t[2].number_input("Min cases per agent × topic", 3, 100, int(T["min_cell"]), key=k + "mcell")
        min_agent = t[3].number_input("Min cases per agent", 5, 500, int(T["min_agent"]), key=k + "magent")
        rare = t[4].number_input("Group categories smaller than", 5, 500, int(T["rare"]), key=k + "rare")

        b1, b2, _ = st.columns([1, 1.4, 3])
        apply = b1.form_submit_button("Apply settings", type="primary")
        redetect = b2.form_submit_button("Re-detect SLA fields")

    if apply or redetect:
        new = copy.deepcopy(S)
        new["slas"] = [clean_sla(r) for r in sla_edit.to_dict("records") if str(r.get("name") or "").strip()
                       and str(r.get("name")) != "nan"]
        new["closure_hours_field"] = hours_field
        new["waiting_statuses"] = waiting
        new["columns"] = {r: (v if v != "—" else "") for r, v in picked.items()}
        new["extra_case_mix"] = mix
        new["n_topics"] = n_topics
        new["flags"] = {str(r["name"]).strip(): str(r["pattern"]) for _, r in flags_edit.iterrows()
                        if str(r.get("name") or "").strip() and str(r.get("pattern") or "").strip()}
        new["thresholds"] = dict(T, gap=gap / 100, specific=spec / 100, min_cell=int(min_cell),
                                 min_agent=int(min_agent), rare=int(rare))
        if redetect:
            new["slas"] = E.auto_detect_slas(cases)
            st.session_state.ver[mode] += 1
        st.session_state.settings[mode] = new
        if mode == "real":
            E.save_settings(new, SETTINGS_FILE)
        st.rerun()

    if st.button("Reset settings to defaults", key=f"reset_{mode}"):
        st.session_state.settings[mode] = demo.demo_settings() if mode == "demo" else E.default_settings()
        st.session_state.ver[mode] += 1
        if mode == "real" and SETTINGS_FILE.exists():
            SETTINGS_FILE.unlink()
        st.rerun()

    with st.expander("Which exports does the app read?"):
        st.dataframe(SOQL, hide_index=True, column_config={"SOQL": st.column_config.TextColumn(width="large")})


# ---------------------------------------------------------------------------
# Run the analysis
# ---------------------------------------------------------------------------
skey = settings_key(S)
try:
    d, topic_table, meta = run_prepare(fp, skey, tables)
except E.DataError as e:
    st.title("Agent gap analysis")
    st.error(f"Can't analyze yet: {e}")
    render_setup(tables, S, mode, ver)
    st.stop()

slas = E.sla_names(meta)
U = meta["time_unit"]          # "business hours" from your field, or "hours" from the dates
sla_color = {s["name"]: P["series"][i % 8] for i, s in enumerate(S.get("slas", []))}
sla_color[E.ANY_SLA] = P["muted"]

with st.sidebar:
    if slas:
        sla = st.selectbox("SLA to analyze", slas, key=f"sla_pick_{mode}",
                           help="Breach rates, drivers and coaching evidence use this SLA.")
    else:
        sla = None
        st.info("No SLA set up yet. Add one in the Setup tab.")

try:
    res = run_analyze(fp, skey, sla, d, meta)
except E.DataError as e:
    st.error(str(e))
    render_setup(tables, S, mode, ver)
    st.stop()
ast = run_agent_sla(fp, skey, d, meta)

flags = meta["flags"]
flag_label = {f: f[5:].replace("_", " ") for f in flags}


def topic_cols(df, col="topic"):
    df = df.copy()
    df[col] = df[col].map(lambda x: x if x in ("All topics",) else tn(x))
    return df


# ---------------------------------------------------------------------------
# Excel export
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=6)
def excel_bytes(fp, skey, sla, names_json, _res, _ast, _topic_table):
    names = json.loads(names_json)
    rn = lambda df, c="topic": df.assign(**{c: df[c].map(lambda x: names.get(str(x), x))}) if c in df else df
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        rn(_res["prio"]).to_excel(xw, sheet_name="Coaching priorities", index=False)
        rn(_res["cells"]).pivot(index="agent_name", columns="topic", values="pct_vs_expected") \
            .to_excel(xw, sheet_name="Skill matrix")
        _res["agents"].to_excel(xw, sheet_name="Agents")
        _ast.to_excel(xw, sheet_name="Agent x SLA", index=False)
        if len(_res["breach_drivers"]):
            _res["breach_drivers"].to_excel(xw, sheet_name="SLA breach drivers", index=False)
        _res["time_drivers"].to_excel(xw, sheet_name="Closure-time drivers", index=False)
        t = _res["topics"].reset_index()
        rn(t).to_excel(xw, sheet_name="Topics", index=False)
        if _topic_table is not None:
            rn(_topic_table).to_excel(xw, sheet_name="Topic terms", index=False)
        rn(_res["cohorts"], "primary_gap").to_excel(xw, sheet_name="Cohorts", index=False)
    return out.getvalue()


with st.sidebar:
    st.divider()
    st.download_button("Download results (Excel)",
                       excel_bytes(fp, skey, sla, json.dumps(S.get("topic_names", {})), res, ast, topic_table),
                       file_name=f"agent_gap_analysis_{(sla or 'no_sla').replace(' ', '_')}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       width="stretch")
    st.caption("Outputs name agents. Use them for coaching, not discipline.")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Agent gap analysis")
span = f"{meta['date_min']:%b %d, %Y} – {meta['date_max']:%b %d, %Y}"
st.caption(f"{meta['n_closed']:,} closed cases · {d['agent'].nunique()} agents · {span} (Mountain Time) · "
           f"Closure time: {meta['time_source']}")
issues = meta["notes"] + res["notes"]
if issues:
    with st.expander(f"Data checks ({len(issues)})"):
        for n in issues:
            st.write("• " + n)

tabs = st.tabs(["Overview", "Coaching priorities", "Skill matrix", "Agent", "Drivers", "Topics",
                "Cohorts", "Setup"])

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
with tabs[0]:
    d_week = d.assign(week=d["created"].dt.tz_localize(None).dt.to_period("W").dt.start_time)
    d_week = d_week[d_week["week"].isin(meta["stable_weeks"])]
    k = st.columns(4)
    k[0].metric("Closed cases", f"{meta['n_closed']:,}")
    k[1].metric("Agents", f"{d['agent'].nunique()}")
    k[2].metric(f"Median {U} to close", f"{d['handle_h'].median():.1f} h",
                help=f"From {meta['time_source']}" + (", minus time in waiting statuses." if meta["has_waiting"] else "."))
    k[3].metric("SLAs tracked", f"{len([s for s in meta['slas'] if s['applies'] > 0])}",
                help="SLA fields with values, from Setup.")

    st.markdown("##### SLA performance")
    sla_rows = [s for s in meta["slas"] if s["applies"] > 0]
    if not sla_rows:
        st.info("No SLA results yet. Set up your SLAs in the Setup tab.")
    else:
        cols = st.columns(min(len(sla_rows), 4))
        recent = d_week["week"].max() - pd.Timedelta(weeks=4)
        for i, s in enumerate(sla_rows):
            w = d_week.groupby("week")[s["key"]].agg(["mean", "count"])
            w = w[w["count"] >= 10]
            last = d_week.loc[d_week["week"] > recent, s["key"]].mean()
            prev = d_week.loc[(d_week["week"] <= recent) & (d_week["week"] > recent - pd.Timedelta(weeks=4)),
                              s["key"]].mean()
            cols[i % len(cols)].metric(
                f"{s['name']} missed", pct(s["rate"], False),
                delta=None if pd.isna(last) or pd.isna(prev) else f"{(last - prev) * 100:+.1f} pp vs prior 4 wks",
                delta_color="inverse", border=True,
                chart_data=(w["mean"] * 100).round(1).tolist() if len(w) > 2 else None, chart_type="line",
                help=f"{s['method']}{': ' + s['field'] if s.get('field') else ''}. "
                     f"Applies to {s['applies']:.0%} of closed cases.")

        c1, c2 = st.columns([1, 1.15])
        with c1:
            fig = go.Figure()
            for s in sla_rows:
                w = d_week.groupby("week")[s["key"]].agg(["mean", "count"]).reset_index()
                w = w[w["count"] >= 10]
                fig.add_scatter(x=w["week"], y=w["mean"], name=s["name"], mode="lines",
                                line=dict(color=sla_color.get(s["name"], P["muted"]), width=2),
                                hovertemplate="%{y:.1%} missed (%{customdata} cases)<extra>" + s["name"] + "</extra>",
                                customdata=w["count"])
            fig.update_layout(hovermode="x unified")
            chart_title("Weekly SLA miss rate", "By week the case was created. Recent weeks with many open "
                                                 "cases are left out, since only quick cases have closed.")
            fig.update_yaxes(tickformat=".0%", rangemode="tozero")
            show(style(fig, 340), "ov_trend")
        with c2:
            m = d.groupby("topic").agg(**{s["name"]: (s["key"], "mean") for s in sla_rows})
            n = d.groupby("topic").agg(**{s["name"]: (s["key"], "count") for s in sla_rows})
            order = d["topic"].value_counts().index
            m, n = m.loc[order], n.loc[order]
            zmax = max(float(np.nanmax(m.values)), 0.05)
            m.columns = n.columns = [wrap(c, 10) for c in m.columns]
            fig = go.Figure(go.Heatmap(
                z=m.values, x=m.columns, y=[short(t) for t in m.index], colorscale=P["seq"], zmin=0, zmax=zmax,
                xgap=2, ygap=2, showscale=False, customdata=n.values,
                hovertemplate="%{y}<br>%{x}: %{z:.1%} missed<br>%{customdata} cases<extra></extra>"))
            for yi, t in enumerate(m.index):
                for xi, c in enumerate(m.columns):
                    v = m.loc[t, c]
                    if pd.notna(v):
                        fig.add_annotation(x=c, y=short(t), text=f"{v:.0%}", showarrow=False,
                                           font=dict(size=12, color=cell_ink(v / zmax)))
            chart_title("Miss rate by topic and SLA", "Stronger color = more cases missed. Hover for case counts.")
            fig.update_yaxes(autorange="reversed", showgrid=False)
            fig.update_xaxes(side="top", showgrid=False, tickfont=dict(color=P["ink2"]), tickangle=0)
            show(style(fig, 340, legend=False), "ov_heat")

    top = res["prio"].head(3)
    if len(top):
        st.markdown("##### Biggest coaching opportunities")
        for _, r in top.iterrows():
            where = "across all topics" if r["gap_type"] == "General pace" else f"on **{short(r['topic'])}**"
            st.markdown(f"- **{r['agent_name']}** {where}: {r['evidence']} "
                        f"(about {r['excess_h']:,.0f} extra {U}).")

    with st.expander("How this works"):
        st.markdown(
            "1. **SLAs and closure time** come straight from your Case fields (business hours already applied by your org).\n"
            "2. **Topics**: case comments are grouped into topics with TF-IDF + NMF (no LLM). Behavior flags come from keywords.\n"
            "3. **Expected time and expected SLA miss** for every case, from case mix only (type, channel, priority, topic and any extra fields).\n"
            "4. **Mixed-effects model**: each agent's pace after adjusting for case mix, pulled toward average when they have few cases.\n"
            "5. **Gaps**: an agent × topic is flagged when it is slower than the team expects *and* slower than the agent's own pace elsewhere. "
            "An agent is flagged for **general pace** when they're slow across topics.\n"
            "6. **Logistic regression** shows what drives SLA misses; **linear regression on log closure time** shows what drives closure time.\n\n"
            "These are associations, not causes. Confirm findings with team leads before acting.")
        if mode == "demo":
            st.info("Demo data plants: Agents 07 and 15 weak on prior auth, Agent 22 weak on payment disputes "
                    "(escalates), Agent 27 weak on credentialing, Agent 31 slow on everything (new hire), Agent 30 "
                    "somewhat slow, Agent 19 fast, and Agents 03 and 11 average but routed the hardest cases.")

# ---------------------------------------------------------------------------
# Coaching priorities
# ---------------------------------------------------------------------------
with tabs[1]:
    prio = topic_cols(res["prio"])
    st.caption(f"Ranked by extra {U} versus what the case mix predicts. **Topic-specific** = slower "
               "on this topic than the team expects and than the agent's own pace elsewhere (a knowledge gap). "
               "**General pace** = slower across the board.")
    if prio.empty:
        st.success("No gaps above the current thresholds. Lower them in Setup to see smaller gaps.")
    else:
        types = st.pills("Gap type", ["Topic-specific", "General pace"], selection_mode="multi",
                         default=["Topic-specific", "General pace"], key="gap_types")
        view = prio[prio["gap_type"].isin(types or [])]
        top = view.head(15).iloc[::-1]
        fig = go.Figure()
        for i, gt in enumerate(["Topic-specific", "General pace"]):
            part = top[top["gap_type"] == gt]
            if part.empty:
                continue
            fig.add_bar(y=part["agent_name"] + " · " + part["topic"].map(lambda t: t.split(": ", 1)[-1]),
                        x=part["excess_h"], orientation="h", name=gt, marker=dict(color=P["series"][i], cornerradius=4),
                        customdata=np.stack([part["n"], part["evidence"]], axis=1),
                        hovertemplate="%{y}<br>%{x:,.0f} extra " + U + " over %{customdata[0]} cases"
                                      "<br>%{customdata[1]}<extra></extra>")
        fig.update_layout(barmode="overlay", bargap=.35)
        chart_title(f"Extra {U} by coaching opportunity (top 15)")
        fig.update_yaxes(categoryorder="array", categoryarray=list(top["agent_name"] + " · " + top["topic"].map(
            lambda t: t.split(": ", 1)[-1])), showgrid=False, tickfont=dict(color=P["ink2"]))
        fig.update_xaxes(title=f"Extra {U} in this period", rangemode="tozero")
        show(style(fig, 90 + 30 * len(top)), "co_bar")

        tbl = view[["agent_name", "topic", "gap_type", "n", "pct_vs_expected", "pct_vs_own_pace",
                    "breach_excess_pp", "excess_h", "evidence"]].copy()
        tbl["pct_vs_expected"] *= 100
        tbl["pct_vs_own_pace"] *= 100
        st.dataframe(tbl, hide_index=True, column_config={
            "agent_name": "Agent", "topic": "Topic", "gap_type": "Gap type",
            "n": st.column_config.NumberColumn("Cases"),
            "pct_vs_expected": st.column_config.NumberColumn("vs team expected", format="%+.0f%%"),
            "pct_vs_own_pace": st.column_config.NumberColumn("vs own pace", format="%+.0f%%"),
            "breach_excess_pp": st.column_config.NumberColumn(f"{sla or 'SLA'} miss vs expected", format="%+.0f pp"),
            "excess_h": st.column_config.NumberColumn(f"Extra {U}", format="%.0f"),
            "evidence": st.column_config.TextColumn("Evidence", width="large")})

# ---------------------------------------------------------------------------
# Skill matrix
# ---------------------------------------------------------------------------
with tabs[2]:
    cells = res["cells"]
    st.caption("Each cell: how much slower (red) or faster (blue) than expected an agent is on a topic, after "
               "case mix. Agents are sorted by overall adjusted pace, slowest at the top. Blank = fewer than "
               f"{S['thresholds']['min_cell']} cases. Hover for detail.")
    measure = st.segmented_control("Compare with", ["Team expected", "Agent's own pace"], default="Team expected",
                                   key="mx_measure")
    col = "pct_vs_own_pace" if measure == "Agent's own pace" else "pct_vs_expected"
    order = [a for a in res["agents"].index if a in set(cells["agent"])]
    topics_order = list(d["topic"].value_counts().index.astype(str))
    z = cells.pivot(index="agent", columns="topic", values=col).reindex(index=order, columns=topics_order)
    n = cells.pivot(index="agent", columns="topic", values="n").reindex(index=order, columns=topics_order)
    act = cells.pivot(index="agent", columns="topic", values="actual_median_h").reindex(index=order, columns=topics_order)
    exp_ = cells.pivot(index="agent", columns="topic", values="expected_h").reindex(index=order, columns=topics_order)
    names = res["agents"]["agent_name"].reindex(order)
    cd = np.dstack([z.values, n.values, act.values, exp_.values])
    fig = go.Figure(go.Heatmap(
        z=z.values.clip(-.6, .6), x=[wrap(short(t), 16) for t in topics_order], y=names.values, colorscale=P["div"],
        zmid=0, zmin=-.6, zmax=.6, xgap=2, ygap=2, customdata=cd, hoverongaps=False,
        hovertemplate="%{y} · %{x}<br>%{customdata[0]:+.0%} vs " + ("own pace" if col == "pct_vs_own_pace" else "team expected")
                      + "<br>%{customdata[1]} cases · median %{customdata[2]:.1f} h vs expected %{customdata[3]:.1f} h<extra></extra>",
        colorbar=dict(tickformat="+.0%", thickness=10, len=min(1, 400 / (80 + 22 * len(order))), y=1, yanchor="top",
                      tickvals=[-.6, -.3, 0, .3, .6], ticktext=["−60% faster", "−30%", "0", "+30%", "+60% slower"],
                      tickfont=dict(color=P["muted"]))))
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color=P["ink2"]))
    fig.update_xaxes(side="top", showgrid=False, tickfont=dict(color=P["ink2"]), tickangle=0)
    show(style(fig, 80 + 22 * len(order), legend=False), "mx_heat")
    with st.expander("Table view"):
        st.dataframe((z * 100).round(0).rename(index=names.to_dict(), columns=lambda t: short(t)),
                     column_config={c: st.column_config.NumberColumn(format="%+.0f%%") for c in
                                    [short(t) for t in topics_order]})

# ---------------------------------------------------------------------------
# Agent drill-down
# ---------------------------------------------------------------------------
with tabs[3]:
    ag = res["agents"]
    first = list(dict.fromkeys(list(res["prio"]["agent"]) + list(ag.index)))
    agent = st.selectbox("Agent", first, format_func=lambda a: ag.loc[a, "agent_name"], key="agent_pick",
                         help="Agents with coaching opportunities are listed first.")
    r = ag.loc[agent]
    m = st.columns(4)
    m[0].metric("Closed cases", f"{int(r['n_cases'])}")
    m[1].metric("Pace vs team (raw)", pct(r["raw_pct"]), help="Positive = slower than the team average, before "
                                                             "accounting for case mix.")
    m[2].metric("Pace after case mix", pct(r["adjusted_pct"]),
                help="Mixed-effects estimate: the agent's own effect once case difficulty is accounted for.")
    m[3].metric(f"{sla or 'SLA'} misses vs expected",
                "–" if pd.isna(r["breach_excess_pp"]) else f"{r['breach_excess_pp']:+.0f} pp",
                help="Agent's miss rate minus the rate their case mix predicts.")
    if abs(r["raw_pct"] - r["adjusted_pct"]) >= .2:
        st.info(f"Case mix matters here: {r['agent_name']} looks {pct(r['raw_pct'])} on raw numbers but "
                f"{pct(r['adjusted_pct'])} once case difficulty is accounted for.")

    ac = topic_cols(res["cells"][res["cells"]["agent"] == agent]).sort_values("pct_vs_expected")
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Bar(
            y=ac["topic"].map(lambda t: t.split(": ", 1)[-1]), x=ac["pct_vs_expected"], orientation="h",
            marker=dict(color=np.where(ac["pct_vs_expected"] > 0, P["slow"], P["fast"]), cornerradius=4),
            customdata=np.stack([ac["n"], ac["pct_vs_own_pace"], ac["actual_median_h"], ac["expected_h"]], axis=1),
            hovertemplate="%{y}<br>%{x:+.0%} vs team expected · %{customdata[1]:+.0%} vs own pace"
                          "<br>%{customdata[0]} cases · median %{customdata[2]:.1f} h vs expected %{customdata[3]:.1f} h<extra></extra>"))
        fig.update_layout(bargap=.35)
        chart_title("Closure time vs expected, by topic", "Red = slower than the team expects for these cases.")
        fig.update_xaxes(tickformat="+.0%", zeroline=True, zerolinewidth=1)
        fig.update_yaxes(showgrid=False, tickfont=dict(color=P["ink2"]))
        show(style(fig, 90 + 34 * max(len(ac), 3), legend=False), "ag_topics")
    with c2:
        if flags:
            team = d[flags].mean()
            mine = d.loc[d["agent"] == agent, flags].mean()
            lab = [flag_label[f] for f in flags]
            fig = go.Figure()
            fig.add_bar(y=lab, x=mine.values, orientation="h", name=r["agent_name"],
                        marker=dict(color=P["series"][0], cornerradius=4),
                        hovertemplate="%{y}: %{x:.0%} of cases<extra>" + r["agent_name"] + "</extra>")
            fig.add_bar(y=lab, x=team.values, orientation="h", name="Team", marker=dict(color=P["muted"], cornerradius=4),
                        hovertemplate="%{y}: %{x:.0%} of cases<extra>Team</extra>")
            fig.update_layout(barmode="group", bargap=.3, bargroupgap=.08)
            chart_title("Share of cases whose notes mention…", "This agent compared with the team.")
            fig.update_xaxes(tickformat=".0%", rangemode="tozero")
            fig.update_yaxes(showgrid=False, autorange="reversed", tickfont=dict(color=P["ink2"]))
            show(style(fig, 100 + 44 * len(flags)), "ag_flags")
        else:
            st.caption("Behavior flags need case_comments.csv.")

    a_sla = ast[ast["agent"] == agent]
    if len(a_sla):
        st.markdown("##### SLA results for this agent")
        t = a_sla[["sla", "cases", "breach_rate", "expected", "excess_pp"]].copy()
        t[["breach_rate", "expected"]] *= 100
        st.dataframe(t, hide_index=True, column_config={
            "sla": "SLA", "cases": "Cases",
            "breach_rate": st.column_config.NumberColumn("Missed", format="%.0f%%"),
            "expected": st.column_config.NumberColumn("Expected for this case mix", format="%.0f%%"),
            "excess_pp": st.column_config.NumberColumn("Difference", format="%+.0f pp")})

    if len(ac) and d["notes"].str.len().sum() > 0:
        worst = ac.sort_values("pct_vs_expected").iloc[-1]
        raw_topic = res["cells"][(res["cells"]["agent"] == agent)].sort_values("pct_vs_expected").iloc[-1]["topic"]
        ex = res["cases"][(res["cases"]["agent"] == agent) & (res["cases"]["topic"].astype(str) == raw_topic)]
        ex = ex.assign(ratio=ex["handle_h"] / ex["exp_h"]).sort_values("ratio", ascending=False).head(6)
        with st.expander(f"Example notes from the slowest cases on {short(raw_topic)} (PHI patterns removed)"):
            st.dataframe(ex[["case_id", "handle_h", "exp_h", "notes"]], hide_index=True, column_config={
                "case_id": "Case", "handle_h": st.column_config.NumberColumn("Hours", format="%.1f"),
                "exp_h": st.column_config.NumberColumn("Expected h", format="%.1f"),
                "notes": st.column_config.TextColumn("Notes", width="large")})
            st.caption("Names in free text are not removed. Use these for coaching conversations only.")

    with st.expander("All agents: pace before and after case mix"):
        st.caption("Raw = compared with the team average. After case mix = the mixed-effects estimate once "
                   "case difficulty is accounted for. Big differences mean an agent gets unusually easy or "
                   "hard cases.")
        allg = ag.reset_index()[["agent_name", "n_cases", "median_h", "raw_pct", "adjusted_pct",
                                 "breach_rate", "breach_excess_pp"]].copy()
        allg[["raw_pct", "adjusted_pct", "breach_rate"]] *= 100
        st.dataframe(allg, hide_index=True, column_config={
            "agent_name": "Agent", "n_cases": "Cases",
            "median_h": st.column_config.NumberColumn(f"Median {U}", format="%.1f"),
            "raw_pct": st.column_config.NumberColumn("Raw vs team", format="%+.0f%%"),
            "adjusted_pct": st.column_config.NumberColumn("After case mix", format="%+.0f%%"),
            "breach_rate": st.column_config.NumberColumn(f"{sla or 'SLA'} missed", format="%.0f%%"),
            "breach_excess_pp": st.column_config.NumberColumn("Missed vs expected", format="%+.0f pp")})

# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------
KIND_COLOR = {"Case mix": P["muted"], "Agent behavior": P["series"][0], "Agent profile": P["series"][2]}


def term_label(t):
    if t.startswith("Topic = "):
        return "Topic = " + short(t.split(" = ", 1)[1])
    return t


def forest(df, val, lo, hi, log, title, spec, suffix, ref):
    f = lambda v: "%{" + v + spec + "}" + suffix
    df = df.iloc[::-1].assign(term=lambda x: x["term"].map(term_label))
    fig = go.Figure()
    for kind, color in KIND_COLOR.items():
        p = df[df["kind"] == kind]
        if p.empty:
            continue
        fig.add_scatter(
            x=p[val], y=p["term"],
            mode="markers", name=kind, marker=dict(color=color, size=9, line=dict(width=2, color="rgba(0,0,0,0)")),
            error_x=dict(type="data", symmetric=False, array=p[hi] - p[val], arrayminus=p[val] - p[lo],
                         color=color, thickness=1.5, width=0),
            customdata=np.stack([p[lo], p[hi], p["p_value"]], axis=1),
            hovertemplate="%{y}<br>" + f("x") + " (95% CI " + f("customdata[0]") + " to "
                          + f("customdata[1]") + ")<br>p = %{customdata[2]:.3f}<extra></extra>")
    fig.add_vline(x=ref, line=dict(color=P["axis"], width=1))
    chart_title(title)
    fig.update_xaxes(type="log" if log else "linear", tickformat=".2~f" if log else "+.0%")
    fig.update_yaxes(showgrid=False, categoryorder="array", tickfont=dict(color=P["ink2"]),
                     categoryarray=list(df["term"]))
    return style(fig, 110 + 26 * len(df))


with tabs[4]:
    sig = st.toggle("Only show significant drivers (p < 0.05)", value=True, key="sig_only")
    st.caption("**Case mix** (gray) is outside the agent's control. **Agent behavior** comes from keywords in "
               "notes. These are associations, not causes: for example, research notes go with longer cases, "
               "but the research may be a symptom of a knowledge gap rather than its cause.")
    c1, c2 = st.columns(2)
    with c1:
        bd = res["breach_drivers"]
        if bd.empty:
            st.info(f"No breach model for {sla or 'this SLA'} (needs both met and missed cases).")
        else:
            bd = bd[bd["p_value"] < .05] if sig else bd
            show(forest(bd, "odds_ratio", "odds_ratio_low", "odds_ratio_high", True,
                        f"What raises the odds of missing {sla}", ":.2f", "×", 1), "dr_breach")
            if res["auc"]:
                st.caption(f"Odds ratio 2× = twice the odds of missing, other things equal. Checked on the most "
                           f"recent 20% of cases: AUC {res['auc']:.2f} (0.5 = guessing, 1 = perfect).")
    with c2:
        td = res["time_drivers"]
        td = td[td["p_value"] < .05] if sig else td
        show(forest(td, "pct_change", "pct_change_low", "pct_change_high", False,
                    "What changes closure time", ":+.0%", "", 0), "dr_time")
        if res["r2"] is not None:
            st.caption(f"+50% = cases take half again as long, other things equal. Checked on the most recent "
                       f"20% of cases: R² {res['r2']:.2f} on log closure time.")

# ---------------------------------------------------------------------------
# Topics
# ---------------------------------------------------------------------------
with tabs[5]:
    tp = res["topics"].reset_index()
    if res["has_sla"] and "share_agents_above_avg" in tp:
        key_ = E.sla_key(meta, sla)
        overall = d.loc[d[key_].notna(), key_].mean()
        CLASSES = {"Widespread: check process, routing or KB": P["series"][1],
                   "Concentrated: coach the agents involved": P["series"][0],
                   "Low miss rate": P["muted"]}

        def classify(r):
            if r["breach_rate"] < overall * .5:
                return "Low miss rate"
            return list(CLASSES)[0] if r["share_agents_above_avg"] >= .5 else list(CLASSES)[1]

        tp["class"] = tp.apply(classify, axis=1)
        tpo = tp.sort_values("breach_rate")
        chart_title(f"{sla} miss rate by topic: systemic or agent-level?",
                    f"The line marks the overall miss rate ({overall:.0%}). Widespread = at least half of agents "
                    "miss more than average on this topic, so training alone won't fix it. Concentrated = a few "
                    "agents drive the misses.")
        fig = go.Figure()
        for cls, color in CLASSES.items():
            p = tpo[tpo["class"] == cls]
            if p.empty:
                continue
            fig.add_bar(y=p["topic"].map(short), x=p["breach_rate"], orientation="h", name=cls,
                        marker=dict(color=color, cornerradius=4),
                        text=p["share_agents_above_avg"].map(lambda v: f"{v:.0%} of agents above avg"),
                        texttemplate="%{text}",
                        textposition="outside", textfont=dict(color=P["ink2"], size=12), cliponaxis=False,
                        customdata=np.stack([p["cases"], p["median_h"]], axis=1),
                        hovertemplate="%{y}<br>%{x:.1%} missed · %{text}<br>%{customdata[0]} cases · "
                                      "median %{customdata[1]:.1f} h<extra></extra>")
        fig.add_vline(x=overall, line=dict(color=P["axis"], width=1))
        fig.update_layout(barmode="overlay", bargap=.35)
        fig.update_xaxes(tickformat=".0%", range=[0, float(tpo["breach_rate"].max()) * 1.5])
        fig.update_yaxes(showgrid=False, categoryorder="array", categoryarray=list(tpo["topic"].map(short)),
                         tickfont=dict(color=P["ink2"]))
        show(style(fig, 110 + 40 * len(tpo)), "tp_bar")

    st.markdown("##### Topics found in case comments")
    if topic_table is None:
        st.caption("No case_comments.csv, so topics are Case Types.")
    else:
        st.caption("Rename topics once you've read their key terms. Names are used everywhere in the app.")
        tt = topic_table.copy()
        tt["name"] = tt["topic"].map(lambda t: S.get("topic_names", {}).get(t, ""))
        tt = tt.merge(tp[["topic", "breach_rate", "median_h"]], on="topic", how="left")
        tt["breach_rate"] *= 100
        edited = st.data_editor(
            tt[["topic", "name", "cases", "breach_rate", "median_h", "top_terms"]], hide_index=True,
            key=f"topic_names_{fp}_{S['n_topics']}", disabled=["topic", "cases", "breach_rate", "median_h", "top_terms"],
            column_config={"topic": "Topic", "name": st.column_config.TextColumn("Your name for it"),
                           "cases": "Cases",
                           "breach_rate": st.column_config.NumberColumn(f"{sla or 'SLA'} missed", format="%.0f%%"),
                           "median_h": st.column_config.NumberColumn("Median hours", format="%.1f"),
                           "top_terms": st.column_config.TextColumn("Key terms", width="large")})
        new_names = {r["topic"]: str(r["name"]).strip() for _, r in edited.iterrows()
                     if r["name"] is not None and str(r["name"]).strip()}
        if new_names != S.get("topic_names", {}):
            S["topic_names"] = new_names
            if mode == "real":
                E.save_settings(S, SETTINGS_FILE)
            st.rerun()

# ---------------------------------------------------------------------------
# Cohorts
# ---------------------------------------------------------------------------
with tabs[6]:
    co = res["cohorts"]
    st.caption(f"Agents grouped by shared gaps, for group training. {res['cohort_method']}")
    groups = [g for g in co["cohort"].unique() if g != 0]
    cols = st.columns(min(max(len(groups), 1), 3))
    for i, g in enumerate(groups):
        part = co[co["cohort"] == g]
        focus = part["focus"].iloc[0]
        with cols[i % len(cols)].container(border=True):
            st.markdown(f"**Focus: {' + '.join(short(f) if f != 'Overall pace' else f for f in focus.split(' + '))}**")
            st.caption(f"{len(part)} agent{'s' if len(part) > 1 else ''}")
            kmeans = res["cohort_method"].startswith("k-means")
            show_t = part[["agent_name"] + (["primary_gap"] if kmeans else []) + ["adjusted_pct"]].copy()
            if kmeans:
                show_t["primary_gap"] = show_t["primary_gap"].map(lambda x: x if x in ("", "Overall pace") else short(x))
            show_t["adjusted_pct"] *= 100
            st.dataframe(show_t, hide_index=True, column_config={
                "agent_name": "Agent", "primary_gap": "Main gap",
                "adjusted_pct": st.column_config.NumberColumn("Overall pace after case mix", format="%+.0f%%")})
    rest = co[co["cohort"] == 0]
    with st.expander(f"No major gaps ({len(rest)} agents)"):
        st.write(", ".join(rest["agent_name"].astype(str)))

with tabs[7]:
    render_setup(tables, S, mode, ver)
