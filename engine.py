"""
Analysis engine for the Agent Gap Analysis app (no UI code in this file).

Same analysis as agent_gap_analysis.py:
- Comment topics (TF-IDF + NMF) and keyword behavior flags
- Logistic regression (SLA miss drivers), linear regression on log closure time
- Mixed-effects agent effects, agent x topic gaps, coaching priorities, cohorts
SLAs are read as-is from your Case fields (your org already applies business hours).
All dates are Mountain Time.
"""
from __future__ import annotations

import copy
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import r2_score, roc_auc_score, silhouette_score

warnings.filterwarnings("ignore")

TZ = "America/Denver"            # Mountain Time (MDT/MST); every date is read and shown in it
SLA_METHODS = ["Met/Missed field", "Due date field", "Hours field vs target"]
ANY_SLA = "Any SLA"
FROM_DATES = ""                  # closure time computed from CreatedDate -> ClosedDate

DEFAULT_FLAGS = {
    "escalated": r"escalat|supervisor|team lead",
    "transferred": r"transfer|warm hand ?off|routed to",
    "research": r"research|not sure|unsure|need to (check|verify|confirm)|look into",
    "callback": r"call ?back|follow[- ]?up|left (a )?(voicemail|vm|message)",
    "rework": r"reopen|second time|third time|already called|still not|again",
}

DEFAULT_SETTINGS = {
    "columns": {
        "case_id": "Id", "agent": "OwnerId", "agent_name": "Owner.Name", "case_type": "Type",
        "channel": "Origin", "priority": "Priority", "created": "CreatedDate", "closed": "ClosedDate",
    },
    "closure_hours_field": FROM_DATES,
    "extra_case_mix": [],
    "hire_field": "HireDate__c",
    "waiting_statuses": ["Waiting on Provider", "Waiting on Member", "On Hold",
                         "Pending External", "Waiting on Customer"],
    "slas": [],
    "n_topics": 8,
    "flags": DEFAULT_FLAGS,
    "thresholds": {"gap": 0.25, "specific": 0.15, "min_cell": 8, "min_agent": 20,
                   "shrink_k": 5, "rare": 30},
    "topic_names": {},
}

SLA_TEMPLATE = {"name": "", "method": "Met/Missed field", "field": "", "breach_values": "", "target": ""}

# Salesforce field names, matched case-insensitively (Data Loader exports upper-case headers)
KNOWN_FIELDS = ["Id", "OwnerId", "Owner.Name", "Type", "Origin", "Priority", "Status", "CreatedDate",
                "ClosedDate", "ParentId", "CommentBody", "CaseId", "Field", "OldValue", "NewValue",
                "Name", "HireDate__c"]

DOMAIN_STOP_WORDS = {"provider", "member", "called", "calling", "caller", "states", "stated",
                     "advised", "per", "re", "pt", "mbr", "spoke", "office", "call",
                     "requested", "requesting", "regarding"}

PHI_PATTERNS = [
    r"[\w.+-]+@[\w-]+\.[\w.]+",                       # email
    r"\b\d{3}-\d{2}-\d{4}\b",                         # SSN
    r"\(?\b\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",       # phone
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",             # dates
    r"\b[A-Za-z]{0,3}\d{5,}\b",                       # member / claim / NPI ids
]


class DataError(Exception):
    """A problem with the input files that the user needs to fix."""


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
def _merge(base, extra):
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        if k not in base:
            continue                  # ignore settings from older versions
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k not in ("flags", "topic_names"):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    if "slas" in out:
        out["slas"] = [dict(SLA_TEMPLATE, **{kk: vv for kk, vv in s.items() if kk in SLA_TEMPLATE})
                       for s in out["slas"] if s.get("method") in SLA_METHODS]
    return out


def default_settings():
    return copy.deepcopy(DEFAULT_SETTINGS)


def load_settings(path: Path):
    try:
        return _merge(DEFAULT_SETTINGS, json.loads(Path(path).read_text()))
    except Exception:
        return default_settings()


def save_settings(settings, path: Path):
    Path(path).write_text(json.dumps(settings, indent=2, default=str))


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known Salesforce fields to their usual spelling, ignoring case."""
    lookup = {f.lower(): f for f in KNOWN_FIELDS}
    ren = {c: lookup[c.strip().lower()] for c in df.columns
           if c.strip().lower() in lookup and c != lookup[c.strip().lower()]}
    return df.rename(columns=ren)


def parse_ts(s: pd.Series) -> pd.Series:
    """Parse timestamps to Mountain Time. Values with an offset ('...Z') are converted;
    values without one are taken as Mountain Time already."""
    s = s.astype("string").str.strip().replace("", pd.NA)
    try:
        dt = pd.to_datetime(s, errors="coerce", format="mixed")
        if getattr(dt.dt, "tz", None) is None:
            return dt.dt.tz_localize(TZ, ambiguous=np.zeros(len(dt), dtype=bool), nonexistent="shift_forward")
        return dt.dt.tz_convert(TZ)
    except Exception:
        return pd.to_datetime(s, errors="coerce", utc=True, format="mixed").dt.tz_convert(TZ)


def scrub_phi(text: str) -> str:
    for p in PHI_PATTERNS:
        text = re.sub(p, " ", text)
    return re.sub(r"\s{2,}", " ", text)


def _safe(name: str) -> str:
    return "x_" + re.sub(r"\W", "_", name)


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype("string").str.replace(",", "").str.strip(), errors="coerce")


# ---------------------------------------------------------------------------
# SLA fields
# ---------------------------------------------------------------------------
_NOISE = (r"due|date|datetime|time|target|deadline|met|missed|miss|status|breach(ed)?|"
          r"violat(ed|ion)|flag|hours?|hrs|elapsed|actual|result|compliance|compliant|c|indicator|"
          r"business|bus|age|duration")
BREACH_WORDS = {"true", "1", "yes", "y", "missed", "miss", "breached", "breach", "violated",
                "failed", "fail", "not met", "out of sla", "overdue", "late", "red"}
MET_FIELD_BREACH = {"false", "0", "no", "n", "not met", "missed", "miss", "breached", "failed",
                    "fail", "out of sla", "overdue", "late", "red"}


def sla_group_name(col: str) -> str:
    base = re.sub(r"__c$", "", col, flags=re.I)
    base = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", base)
    base = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", base)
    keep = [t for t in re.split(r"[_\s.]+", base) if t and not re.fullmatch(_NOISE, t, re.I)]
    keep = [t.capitalize() if t.isupper() and len(t) > 3 else t for t in keep]   # Data Loader caps
    name = " ".join(keep) or col
    return re.sub(r"\bsla\b", "SLA", name, flags=re.I)


def detect_kind(s: pd.Series) -> str:
    v = s.dropna().astype(str).str.strip()
    v = v[v != ""]
    if v.empty:
        return "empty"
    sample = v.head(500)
    looks_date = sample.str.contains(r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}").mean() > .9
    if looks_date and pd.to_datetime(sample, errors="coerce", utc=True, format="mixed").notna().mean() > .9:
        return "date/time"
    nums = _num(sample)
    if nums.notna().mean() > .9 and not set(nums.dropna().unique()) <= {0, 1}:
        return "number"
    if v.str.lower().nunique() <= 6:
        return "met/missed values"
    return "text"


def guess_breach_values(col: str, s: pd.Series) -> str:
    vals = set(s.dropna().astype(str).str.strip().str.lower()) - {""}
    met_style = re.search(r"met|within|compliant|achiev|on.?time", col, re.I) and \
        not re.search(r"miss|breach|violat|not|late|over", col, re.I)
    bad = MET_FIELD_BREACH if met_style else BREACH_WORDS
    return ", ".join(sorted(v for v in vals if v in bad))


def sla_fields(cases: pd.DataFrame) -> pd.DataFrame:
    """Case columns that look like SLA fields, with detected type and sample values."""
    rows = []
    for c in cases.columns:
        if not re.search(r"sla", c, re.I):
            continue
        s = cases[c]
        filled = s.notna() & (s.astype(str).str.strip() != "")
        rows.append({"field": c, "suggested SLA": sla_group_name(c), "type": detect_kind(s),
                     "filled": f"{filled.mean():.0%}",
                     "sample values": ", ".join(s[filled].astype(str).str.strip().value_counts().index[:4])})
    return pd.DataFrame(rows, columns=["field", "suggested SLA", "type", "filled", "sample values"])


def auto_detect_slas(cases: pd.DataFrame) -> list:
    """One SLA per field group: a Met/Missed field wins, then a due date, then an hours field."""
    found = sla_fields(cases)
    rank = {"met/missed values": 0, "date/time": 1, "number": 2}
    out = {}
    for _, r in found.iterrows():
        if r["type"] not in rank:
            continue
        name = r["suggested SLA"]
        if name in out and rank[out[name]["_kind"]] <= rank[r["type"]]:
            continue
        sla = dict(SLA_TEMPLATE, name=name, field=r["field"], _kind=r["type"])
        if r["type"] == "met/missed values":
            sla["method"] = "Met/Missed field"
            sla["breach_values"] = guess_breach_values(r["field"], cases[r["field"]])
        elif r["type"] == "date/time":
            sla["method"] = "Due date field"
        else:
            sla["method"] = "Hours field vs target"
        out[name] = sla
    return [{k: v for k, v in s.items() if k != "_kind"} for s in out.values()]


def suggest_hours_field(cases: pd.DataFrame) -> str:
    """A numeric Case field that looks like business hours to close, if there is one."""
    for c in cases.columns:
        if re.search(r"(business|bus)[_\s]*(hours?|hrs)|hours?[_\s]*to[_\s]*(close|resol)|"
                     r"(resolution|handle|closure)[_\s]*(time|hours?|hrs)", c, re.I) \
                and not re.search(r"sla|target", c, re.I) and detect_kind(cases[c]) == "number":
            return c
    return FROM_DATES


def compute_sla(sla: dict, raw: pd.DataFrame, closed: pd.Series):
    """Read one SLA from the Case fields. Returns (miss 1/0/NaN array, problem or None)."""
    n = len(raw)
    method, field = sla.get("method"), (sla.get("field") or "").strip()
    if field not in raw.columns:
        return np.full(n, np.nan), (f"field '{field}' not found" if field else "pick a Case field")

    if method == "Met/Missed field":
        v = raw[field].astype("string").str.strip().str.lower()
        bad = {b.strip().lower() for b in str(sla.get("breach_values") or "").split(",") if b.strip()}
        if not bad:
            return np.full(n, np.nan), "list the values that mean missed"
        return np.where(v.isna() | (v == ""), np.nan, v.isin(bad).astype(float)), None

    if method == "Due date field":
        due = parse_ts(raw[field]).reset_index(drop=True)
        c = closed.reset_index(drop=True)
        return np.where(due.isna() | c.isna(), np.nan, (c > due).astype(float)), None

    if method == "Hours field vs target":
        hours = _num(raw[field]).to_numpy()
        t = str(sla.get("target") or "").strip()
        try:
            target = np.full(n, float(t))
        except ValueError:
            if t not in raw.columns:
                return np.full(n, np.nan), "set the target: a number of hours or a target-hours field"
            target = _num(raw[t]).to_numpy()
        return np.where(np.isnan(hours) | np.isnan(target), np.nan, (hours > target).astype(float)), None

    return np.full(n, np.nan), f"unknown method '{method}'"


# ---------------------------------------------------------------------------
# Comments
# ---------------------------------------------------------------------------
def build_topics(text: pd.Series, n_topics: int):
    stop = list(ENGLISH_STOP_WORDS | DOMAIN_STOP_WORDS)
    vec = TfidfVectorizer(stop_words=stop, ngram_range=(1, 2), min_df=5 if len(text) >= 300 else 1,
                          max_df=0.4, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b", sublinear_tf=True)
    X = vec.fit_transform(text)
    k = max(2, min(n_topics, X.shape[1] - 1, X.shape[0] - 1))
    nmf = NMF(n_components=k, random_state=42, init="nndsvda", max_iter=500)
    W = nmf.fit_transform(X)
    terms = np.array(vec.get_feature_names_out())
    labels = []
    for i, row in enumerate(nmf.components_):
        picked = []
        for t in terms[row.argsort()[::-1]]:
            if not any(t in p or p in t for p in picked):
                picked.append(t)
            if len(picked) == 3:
                break
        labels.append(f"T{i + 1}: " + " / ".join(picked))
    topic = np.where(W.sum(axis=1) > 0, np.array(labels)[W.argmax(axis=1)], "Unclassified")
    table = pd.DataFrame({"topic": labels,
                          "top_terms": [", ".join(terms[r.argsort()[::-1][:10]]) for r in nmf.components_]})
    return topic, table


# ---------------------------------------------------------------------------
# Prepare: one row per closed case with outcomes and features
# ---------------------------------------------------------------------------
def check_columns(cases: pd.DataFrame, S: dict) -> list:
    col = S["columns"]
    labels = {"case_id": "Case Id", "agent": "Agent (owner)", "created": "Created date",
              "closed": "Closed date"}
    return [f"{lbl}: column '{col.get(k)}' not found in cases.csv" for k, lbl in labels.items()
            if not col.get(k) or col.get(k) not in cases.columns]


def prepare(tables: dict, S: dict):
    cases = tables.get("cases")
    if cases is None:
        raise DataError("cases.csv is required.")
    problems = check_columns(cases, S)
    if problems:
        raise DataError("; ".join(problems))
    col, notes = S["columns"], []

    def get(key, default="Unknown"):
        c = col.get(key)
        if c and c in cases.columns:
            return cases[c].fillna(default).astype(str).replace("", default)
        return pd.Series(default, index=cases.index)

    d = pd.DataFrame({
        "case_id": cases[col["case_id"]].astype(str),
        "agent": cases[col["agent"]].astype(str),
        "case_type": get("case_type"), "channel": get("channel"), "priority": get("priority"),
        "created": parse_ts(cases[col["created"]]), "closed": parse_ts(cases[col["closed"]]),
    })
    extra = {}
    for c in S.get("extra_case_mix", []):
        if c in cases.columns:
            d[_safe(c)] = cases[c].fillna("Unknown").astype(str).replace("", "Unknown")
            extra[_safe(c)] = c

    n_total = len(d)
    queue = d["agent"].str.startswith("00G")
    owned = d[~queue & d["created"].notna()]
    n_open = int(owned["closed"].isna().sum())
    # weeks (by created date) complete enough to trend: few cases still open
    wk = owned["created"].dt.tz_localize(None).dt.to_period("W").dt.start_time
    open_share = owned["closed"].isna().groupby(wk).mean()
    limit = max(0.10, 2 * float(open_share.median())) if len(open_share) else 1
    last_full = owned["created"].max().tz_localize(None).normalize() - pd.Timedelta(days=6) \
        if len(owned) else pd.Timestamp.max
    stable_weeks = [w for w, v in open_share.items() if v <= limit and w <= last_full]

    keep = ~queue & d["closed"].notna() & d["created"].notna() & (d["agent"] != "nan")
    d, raw = d[keep].reset_index(drop=True), cases[keep].reset_index(drop=True)
    if len(d) < 50:
        raise DataError(f"Only {len(d)} closed, agent-owned cases found. Need at least 50.")
    if queue.any():
        notes.append(f"{int(queue.sum()):,} queue-owned cases excluded.")

    # closure time: your business-hours field, or CreatedDate -> ClosedDate minus waiting time
    hours_field = S.get("closure_hours_field") or FROM_DATES
    d["waiting_h"] = 0.0
    history = tables.get("history")
    hist_ok = history is not None and {"CaseId", "Field", "NewValue", "CreatedDate"} <= set(history.columns)
    if hours_field and hours_field in raw.columns:
        d["handle_h"] = _num(raw[hours_field]).to_numpy()
        bad = d["handle_h"].isna() | (d["handle_h"] < 0)
        if bad.mean() > .5:
            raise DataError(f"Closure time field '{hours_field}' is blank or not a number for most cases.")
        if bad.any():
            notes.append(f"{int(bad.sum()):,} cases have no value in {hours_field} and were left out.")
            d, raw = d[~bad].reset_index(drop=True), raw[~bad].reset_index(drop=True)
        time_source, time_unit = hours_field, "business hours"
    else:
        if hours_field:
            notes.append(f"Closure time field '{hours_field}' not found; using CreatedDate to ClosedDate.")
        d["handle_h"] = (d["closed"] - d["created"]).dt.total_seconds() / 3600
        statuses = set(S.get("waiting_statuses") or [])
        if hist_ok and statuses:
            h = history[history["Field"] == "Status"][["CaseId", "NewValue", "CreatedDate"]].copy()
            h.columns = ["case_id", "status", "at"]
            h["at"] = parse_ts(h["at"])
            h = h.dropna(subset=["at"]).sort_values(["case_id", "at"])
            h["until"] = h.groupby("case_id")["at"].shift(-1)
            h = h.merge(d[["case_id", "closed"]], on="case_id", how="inner")
            h["until"] = h["until"].fillna(h["closed"])
            h["hrs"] = ((h["until"] - h["at"]).dt.total_seconds() / 3600).clip(lower=0)
            d["waiting_h"] = d["case_id"].map(h[h["status"].isin(statuses)].groupby("case_id")["hrs"].sum()).fillna(0)
            d["handle_h"] = d["handle_h"] - d["waiting_h"]
        elif history is not None and not hist_ok:
            notes.append("case_history.csv is missing CaseId, Field, NewValue or CreatedDate; ignored.")
        else:
            notes.append("No case_history.csv: waiting time is not removed from closure time.")
        time_source, time_unit = "CreatedDate to ClosedDate", "hours"
    d["handle_h"] = d["handle_h"].clip(lower=0.1)
    d["log_hours"] = np.log(d["handle_h"])

    # SLAs, read from your fields
    sla_meta = []
    for i, sla in enumerate(S.get("slas", [])):
        name = (sla.get("name") or "").strip()
        if not name:
            continue
        miss, problem = compute_sla(sla, raw, d["closed"])
        key = f"sla_{i}"
        d[key] = miss
        applies = float(np.mean(~np.isnan(miss)))
        sla_meta.append({"name": name, "key": key, "method": sla.get("method"), "field": sla.get("field"),
                         "applies": applies, "rate": float(np.nanmean(miss)) if applies else np.nan,
                         "problem": problem})
        if problem:
            notes.append(f"{name}: {problem}.")
        elif applies == 0:
            notes.append(f"{name}: no cases have a value; check the field and the missed values.")
    live = [m["key"] for m in sla_meta if m["applies"] > 0]
    if len(live) > 1:
        d["sla_any"] = d[live].max(axis=1, skipna=True).where(d[live].notna().any(axis=1))

    # comments
    comments = tables.get("comments")
    topic_table = None
    if comments is not None and {"ParentId", "CommentBody"} <= set(comments.columns):
        c = comments[comments["ParentId"].isin(set(d["case_id"]))]
        text = (c.assign(body=c["CommentBody"].fillna("").astype(str).map(scrub_phi))
                .groupby("ParentId")["body"].apply(" ".join))
        d["notes"] = d["case_id"].map(text).fillna("")
        d["log_comments"] = np.log1p(d["case_id"].map(c.groupby("ParentId").size()).fillna(0))
        try:
            d["topic"], topic_table = build_topics(d["notes"], int(S.get("n_topics", 8)))
            topic_table["cases"] = topic_table["topic"].map(d["topic"].value_counts()).fillna(0).astype(int)
        except ValueError:
            d["topic"] = d["case_type"]
            notes.append("Comments too sparse for topic modeling; using Case Type as topic.")
        for name, pattern in (S.get("flags") or {}).items():
            try:
                d["flag_" + re.sub(r"\W", "_", name)] = d["notes"].str.contains(pattern, case=False, regex=True).astype(int)
            except re.error:
                notes.append(f"Behavior keyword '{name}' has an invalid pattern; skipped.")
    else:
        d["topic"] = d["case_type"]
        d["notes"] = ""
        notes.append("No case_comments.csv: topics fall back to Case Type and behavior flags are off.")

    # agents
    users = tables.get("users")
    names = None
    if users is not None and {"Id", "Name"} <= set(users.columns):
        names = users.set_index(users["Id"].astype(str).str[:15])["Name"]
        hire = S.get("hire_field")
        if hire and hire in users.columns:
            h = parse_ts(users[hire])
            h.index = users["Id"].astype(str).str[:15]
            ten = ((d["created"] - d["agent"].str[:15].map(h)).dt.days / 365.25).clip(lower=0)
            if ten.notna().mean() > .5:
                d["tenure_years"] = ten.fillna(ten.median())
    d["agent_name"] = d["agent"].str[:15].map(names) if names is not None else np.nan
    an = col.get("agent_name")
    if an and an in raw.columns:
        d["agent_name"] = d["agent_name"].fillna(raw[an])
    d["agent_name"] = d["agent_name"].fillna(d["agent"]).astype(str)

    meta = {"n_total": n_total, "n_open": n_open, "n_closed": len(d), "slas": sla_meta, "extra": extra,
            "time_source": time_source, "time_unit": time_unit,
            "has_waiting": bool(d["waiting_h"].sum() > 0), "has_comments": topic_table is not None,
            "notes": notes, "flags": [c for c in d.columns if c.startswith("flag_")],
            "date_min": d["created"].min(), "date_max": d["closed"].max(), "stable_weeks": stable_weeks}
    return d, topic_table, meta


# ---------------------------------------------------------------------------
# Modeling helpers
# ---------------------------------------------------------------------------
BASE_MIX = ["case_type", "channel", "priority", "topic"]


def _mix_cols(d):
    return BASE_MIX + [c for c in d.columns if c.startswith("x_")]


def _prep_categoricals(d, rare):
    for c in _mix_cols(d):
        vc = d[c].value_counts()
        d[c] = d[c].where(d[c].map(vc) >= rare, "Other")
        d[c] = pd.Categorical(d[c], categories=d[c].value_counts().index.tolist())
    return d


def _formula(target, d, behavior=False):
    same_topic = (d["topic"].astype(str) == d["case_type"].astype(str)).all()
    terms = [f"C({c})" for c in _mix_cols(d)
             if d[c].nunique() > 1 and not (c == "topic" and same_topic)]
    if behavior:
        extra = ["log_comments", "tenure_years"] + [c for c in d.columns if c.startswith("flag_")]
        terms += [c for c in extra if c in d.columns and d[c].nunique() > 1]
    return f"{target} ~ " + (" + ".join(terms) if terms else "1")


@dataclass
class Fitted:
    model: object
    info: object
    keep: list
    dropped: list

    def predict(self, data):
        X = patsy.build_design_matrices([self.info], data, return_type="dataframe")[0]
        return np.asarray(self.model.predict(X[self.keep]))


def _independent(X: pd.DataFrame) -> list:
    """Columns of X that add information, kept in formula order, so case type, channel,
    priority and topic win over extra fields that duplicate them."""
    A = X.to_numpy(dtype=float)
    keep, rank = [], 0
    for i in range(A.shape[1]):
        r = np.linalg.matrix_rank(A[:, keep + [i]])
        if r > rank:
            keep.append(i)
            rank = r
    return [X.columns[i] for i in keep]


def _fit(kind, f, data, groups=None) -> Fitted:
    """Fit OLS / logit / mixed model after dropping redundant (aliased) columns, e.g. a
    Department field that is fully determined by topic."""
    y, X = patsy.dmatrices(f, data, return_type="dataframe")
    keep = _independent(X)
    dropped = [c for c in X.columns if c not in keep]
    y, Xk = y.iloc[:, 0], X[keep]
    if kind == "ols":
        m = sm.OLS(y, Xk).fit()
    elif kind == "logit":
        try:
            m = sm.Logit(y, Xk).fit(disp=0, maxiter=300)
            if not np.isfinite(m.params).all() or not np.isfinite(m.bse).all():
                raise ValueError
        except Exception:
            m = sm.Logit(y, Xk).fit(method="bfgs", disp=0, maxiter=1000)
    else:
        m = None
        for method in ("lbfgs", "powell"):
            try:
                m = sm.MixedLM(y, Xk, groups=groups.loc[y.index]).fit(reml=True, method=method)
                if float(np.ravel(m.cov_re)[0]) > 1e-6:
                    break
            except Exception:
                m = None
    return Fitted(m, X.design_info, keep, dropped)


def _pretty_var(v, extra):
    v = extra.get(v, v)
    return re.sub(r"__c$", "", v).replace("_", " ").strip()


def _tidy(term, extra):
    m = re.match(r"C\((\w+)\)\[T\.(.*)\]$", term)
    if m:
        return f"{_pretty_var(m.group(1), extra).capitalize()} = {m.group(2)}"
    return {"log_comments": "Number of comments (log)", "tenure_years": "Tenure (years)"}.get(
        term, "Notes mention: " + term[5:].replace("_", " ") if term.startswith("flag_") else term)


def _kind(term):
    if term.startswith(("flag_", "log_comments")):
        return "Agent behavior"
    if term.startswith("tenure"):
        return "Agent profile"
    return "Case mix"


def _coef_table(model, transform, label, extra):
    ci = model.conf_int()
    t = pd.DataFrame({"raw": model.params.index, "coef": model.params.values,
                      "p_value": model.pvalues.values, "lo": ci[0].values, "hi": ci[1].values})
    t = t[(t["raw"] != "Intercept") & np.isfinite(t["hi"]) & np.isfinite(t["lo"])].copy()
    t[label], t[label + "_low"], t[label + "_high"] = transform(t["coef"]), transform(t["lo"]), transform(t["hi"])
    t["kind"] = t["raw"].map(_kind)
    t["term"] = t["raw"].map(lambda x: _tidy(x, extra))
    return t.drop(columns=["lo", "hi", "raw", "coef"]).reset_index(drop=True)


def _time_split(d, frac=0.8):
    cut = d["created"].quantile(frac)
    return d[d["created"] <= cut], d[d["created"] > cut]


def _expected_breach(d, target):
    """Case-mix-only expected breach probability (NaN where the SLA does not apply)."""
    sub = d[d[target].notna()]
    if sub[target].nunique() < 2 or sub[target].sum() < 5:
        return pd.Series(np.nan, index=d.index)
    try:
        p = pd.Series(_fit("logit", _formula(target, sub), sub).predict(d), index=d.index)
    except Exception:
        p = d["topic"].map(sub.groupby("topic", observed=True)[target].mean()).astype(float)
    return p.where(d[target].notna())


def sla_key(meta, name):
    if name == ANY_SLA:
        return "sla_any"
    return next((s["key"] for s in meta["slas"] if s["name"] == name), None)


def sla_names(meta):
    names = [s["name"] for s in meta["slas"] if s["applies"] > 0]
    return names + ([ANY_SLA] if len(names) > 1 else [])


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def analyze(d0: pd.DataFrame, meta: dict, S: dict, sla_name: str | None):
    T = S["thresholds"]
    d = _prep_categoricals(d0.copy(), T["rare"])
    extra, flags = meta["extra"], meta["flags"]
    res = {"notes": []}
    key = sla_key(meta, sla_name) if sla_name else None
    has_sla = key is not None and key in d and d[key].notna().sum() >= 30 and d[key].nunique() == 2
    d["breach"] = d[key] if has_sla else np.nan
    train, test = _time_split(d)

    # 1. Logistic regression: SLA breach drivers
    res["auc"], res["breach_drivers"] = None, pd.DataFrame()
    if has_sla:
        bd = d[d["breach"].notna()]
        tr, te = _time_split(bd)
        f_b = _formula("breach", bd, behavior=True)
        try:
            if te["breach"].nunique() == 2:
                res["auc"] = roc_auc_score(te["breach"], _fit("logit", f_b, tr).predict(te))
        except Exception:
            pass
        try:
            fb = _fit("logit", f_b, bd)
            res["breach_drivers"] = _coef_table(fb.model, np.exp, "odds_ratio", extra) \
                .sort_values("odds_ratio", ascending=False)
            if fb.dropped:
                res["notes"].append("Left out of the breach model as redundant with other fields: "
                                    + ", ".join(_tidy(c, extra) for c in fb.dropped) + ".")
        except Exception:
            res["notes"].append(f"{sla_name}: breach model could not be fitted (too few breaches?).")
    elif sla_name:
        res["notes"].append(f"{sla_name}: not enough cases with both outcomes to model.")

    # 2. Linear regression on log closure time
    f_t = _formula("log_hours", d, behavior=True)
    try:
        res["r2"] = r2_score(test["log_hours"], _fit("ols", f_t, train).predict(test))
    except Exception:
        res["r2"] = None
    ft = _fit("ols", f_t, d)
    res["time_drivers"] = _coef_table(ft.model, lambda b: np.exp(b) - 1, "pct_change", extra) \
        .sort_values("pct_change", ascending=False)

    # 3. Expected outcomes from case mix only (behavior is what we measure, so it stays out)
    base = _fit("ols", _formula("log_hours", d), d)
    d["exp_log_hours"] = base.predict(d)
    d["exp_h"] = np.exp(d["exp_log_hours"])
    d["resid"] = d["log_hours"] - d["exp_log_hours"]
    d["exp_breach"] = _expected_breach(d, "breach") if has_sla else np.nan

    # 4. Mixed-effects model: agent effect after case mix, shrunk toward average
    mm = _fit("mixed", _formula("log_hours", d), d, groups=d["agent"]).model
    k = T["shrink_k"]
    g = d.groupby("agent")["resid"]
    if mm is not None and float(np.ravel(mm.cov_re)[0]) > 1e-6:
        re_agent = pd.Series({a: float(np.ravel(v)[0]) for a, v in mm.random_effects.items()})
        res["agent_sd"] = float(np.ravel(mm.cov_re)[0]) ** .5
    else:
        re_agent, res["agent_sd"] = g.sum() / (g.size() + k), None

    overall = d["log_hours"].mean()
    agents = d.groupby("agent").agg(
        agent_name=("agent_name", "first"), n_cases=("case_id", "size"),
        mean_log=("log_hours", "mean"), resid_sum=("resid", "sum"), median_h=("handle_h", "median"),
        breach_rate=("breach", "mean"), expected_breach=("exp_breach", "mean"),
        **{f + "_rate": (f, "mean") for f in flags})
    agents["raw_pct"] = np.exp(agents["mean_log"] - overall) - 1
    agents["adjusted_pct"] = np.exp(agents.index.map(re_agent).astype(float)) - 1
    agents["breach_excess_pp"] = 100 * (agents["breach_rate"] - agents["expected_breach"])
    agents = agents[agents["n_cases"] >= T["min_agent"]].sort_values("adjusted_pct", ascending=False)
    if agents.empty:
        raise DataError(f"No agent has at least {T['min_agent']} closed cases. Lower the minimum in Setup.")

    # 5. Agent x topic cells: vs team expected, and vs the agent's own pace on other topics
    cells = d.groupby(["agent", "topic"], observed=True).agg(
        agent_name=("agent_name", "first"), n=("case_id", "size"), resid_sum=("resid", "sum"),
        expected_h=("exp_h", "mean"), actual_median_h=("handle_h", "median"),
        breach_rate=("breach", "mean"), expected_breach=("exp_breach", "mean"),
        **{f + "_rate": (f, "mean") for f in flags}).reset_index()
    cells = cells[cells["agent"].isin(agents.index)].copy()
    cells["topic"] = cells["topic"].astype(str)
    a_sum, a_n = cells["agent"].map(agents["resid_sum"]), cells["agent"].map(agents["n_cases"])
    cell_eff = cells["resid_sum"] / (cells["n"] + k)
    other_eff = (a_sum - cells["resid_sum"]) / (a_n - cells["n"] + k)
    cells["pct_vs_expected"] = np.exp(cell_eff) - 1
    cells["pct_vs_own_pace"] = np.exp(cell_eff - other_eff) - 1
    cells["breach_excess_pp"] = 100 * (cells["breach_rate"] - cells["expected_breach"])
    rates = d.groupby("topic", observed=True)[flags].mean() if flags else pd.DataFrame()
    rates.index = rates.index.astype(str)
    for f in flags:
        cells[f + "_vs_peers"] = cells[f + "_rate"] / cells["topic"].map(rates[f]).replace(0, np.nan)
    cells = cells[cells["n"] >= T["min_cell"]].reset_index(drop=True)

    # 6. Coaching priorities
    #    Topic-specific: slower than the team expects AND slower than the agent's own pace
    #    elsewhere, or much slower than their own pace even if still near team average.
    spec = cells[((cells["pct_vs_expected"] >= T["gap"]) & (cells["pct_vs_own_pace"] >= T["specific"]))
                 | ((cells["pct_vs_own_pace"] >= 2 * T["gap"]) & (cells["pct_vs_expected"] >= 0))].copy()
    spec["gap_type"] = "Topic-specific"
    spec["excess_h"] = spec["n"] * spec["expected_h"] * spec["pct_vs_expected"].clip(lower=0) \
        + spec["n"] * spec["expected_h"] * (spec["pct_vs_own_pace"] - spec["pct_vs_expected"]).clip(lower=0) * .5
    #    General pace: slow across topics once their topic-specific gaps are taken out
    sp = spec.groupby("agent")[["resid_sum", "n"]].sum()
    ex_sum = agents["resid_sum"] - sp["resid_sum"].reindex(agents.index).fillna(0)
    ex_n = agents["n_cases"] - sp["n"].reindex(agents.index).fillna(0)
    agents["general_pct"] = np.exp(ex_sum / (ex_n + k)) - 1
    gen = agents[agents["general_pct"] >= T["gap"]].copy()
    gen = gen.assign(agent=gen.index, topic="All topics", gap_type="General pace", n=gen["n_cases"],
                     pct_vs_expected=gen["general_pct"], pct_vs_own_pace=np.nan)
    gen["excess_h"] = gen["agent"].map(d.groupby("agent")["exp_h"].sum()) * gen["pct_vs_expected"]

    def evidence(r):
        if r["gap_type"] == "General pace":
            parts = [f"{r['pct_vs_expected']:+.0%} time vs expected across topics"]
        else:
            parts = [f"{r['pct_vs_expected']:+.0%} vs team expected",
                     f"{r['pct_vs_own_pace']:+.0%} vs own pace on other topics"]
        if pd.notna(r.get("breach_excess_pp")) and abs(r["breach_excess_pp"]) >= 5:
            parts.append(f"{sla_name} breach {r['breach_excess_pp']:+.0f} pp vs expected")
        for f in flags:
            v = r.get(f + "_vs_peers")
            if pd.notna(v) and v >= 2 and r[f + "_rate"] >= 0.2:
                parts.append(f"'{f[5:].replace('_', ' ')}' notes {v:.1f}x peers")
        return "; ".join(parts)

    cols = ["agent", "agent_name", "topic", "gap_type", "n", "pct_vs_expected", "pct_vs_own_pace",
            "breach_excess_pp", "excess_h"] + [f + s for f in flags for s in ("_rate", "_vs_peers")]
    prio = pd.concat([spec.reindex(columns=cols), gen.reindex(columns=cols)], ignore_index=True)
    prio["evidence"] = prio.apply(evidence, axis=1) if len(prio) else pd.Series(dtype=str)
    prio = prio.sort_values("excess_h", ascending=False).reset_index(drop=True)

    # 7. Training cohorts
    prof = np.log1p(cells.pivot(index="agent", columns="topic", values="pct_vs_own_pace").clip(lower=0))
    prof = prof.where(cells.pivot(index="agent", columns="topic", values="pct_vs_expected") >= 0)
    prof["Overall pace"] = np.log1p(agents["general_pct"].clip(lower=0).reindex(prof.index))
    prof = prof.reindex(agents.index).fillna(0)
    flagged = set(prio["agent"])
    need = prof.index[prof.index.isin(flagged)]
    labels = pd.Series(0, index=prof.index)
    names = {0: "No major gaps"}
    primary = pd.Series("", index=prof.index)
    for a in need:
        rows = prio[prio["agent"] == a]
        primary[a] = "Overall pace" if (rows["gap_type"] == "General pace").any() else rows.iloc[0]["topic"]
    res["cohort_method"] = "No agents need coaching at the current thresholds."
    if 0 < len(need) < 12:
        res["cohort_method"] = "Grouped by each agent's main gap (fewer than 12 agents to cluster)."
        for i, gname in enumerate(primary[need].value_counts().index, start=1):
            labels.loc[need[primary[need] == gname]] = i
            names[i] = gname
    elif len(need) >= 12:
        res["cohort_method"] = "k-means on gap profiles; number of groups chosen by silhouette score."
        X = prof.loc[need].values
        best = (-1, np.zeros(len(need), dtype=int))
        for kk in range(2, min(7, len(need))):
            km = KMeans(n_clusters=kk, n_init=20, random_state=42).fit(X)
            s = silhouette_score(X, km.labels_)
            if s > best[0]:
                best = (s, km.labels_)
        labels.loc[need] = best[1] + 1
        for c in sorted(set(best[1] + 1)):
            center = prof.loc[labels[labels == c].index].mean().sort_values(ascending=False)
            focus = [t for t, v in center.items() if v >= np.log1p(T["specific"])]
            names[c] = " + ".join(focus[:3]) if focus else "Mixed gaps"
    cohorts = pd.DataFrame({"agent": prof.index, "cohort": labels.values,
                            "focus": labels.map(names).values, "primary_gap": primary.values})
    cohorts["agent_name"] = cohorts["agent"].map(agents["agent_name"])
    cohorts["adjusted_pct"] = cohorts["agent"].map(agents["adjusted_pct"])
    cohorts = cohorts.sort_values(["cohort", "adjusted_pct"], ascending=[True, False])

    # 8. Topics: systemic vs agent-level
    topics = d.groupby("topic", observed=True).agg(
        cases=("case_id", "size"), breach_rate=("breach", "mean"),
        median_h=("handle_h", "median"), **{f + "_rate": (f, "mean") for f in flags})
    if has_sla:
        atb = d.groupby(["topic", "agent"], observed=True)["breach"].agg(["mean", "size"]).reset_index()
        atb = atb[atb["size"] >= 5]
        atb["hi"] = atb["mean"] > d["breach"].mean()
        topics["share_agents_above_avg"] = atb.groupby("topic", observed=True)["hi"].mean()
    topics.index = topics.index.astype(str)
    topics = topics.sort_values("breach_rate" if has_sla else "median_h", ascending=False)

    res.update(has_sla=has_sla, sla_name=sla_name, agents=agents.drop(columns=["mean_log", "resid_sum"]),
               cells=cells.drop(columns=["resid_sum"]), prio=prio, cohorts=cohorts, topics=topics,
               cases=d[["case_id", "agent", "agent_name", "topic", "handle_h", "exp_h", "breach",
                        "exp_breach", "notes", "created", "closed"] + flags])
    return res


def agent_sla_table(d0: pd.DataFrame, meta: dict, S: dict) -> pd.DataFrame:
    """Each agent's breach rate vs case-mix expected, for every SLA."""
    d = _prep_categoricals(d0.copy(), S["thresholds"]["rare"])
    rows = []
    for s in meta["slas"]:
        k = s["key"]
        if k not in d or d[k].notna().sum() < 30:
            continue
        exp = _expected_breach(d, k)
        g = pd.DataFrame({"agent": d["agent"], "agent_name": d["agent_name"], "b": d[k], "e": exp})
        g = g[g["b"].notna()].groupby("agent").agg(agent_name=("agent_name", "first"),
                                                   cases=("b", "size"), breach_rate=("b", "mean"),
                                                   expected=("e", "mean")).reset_index()
        g["sla"] = s["name"]
        rows.append(g)
    if not rows:
        return pd.DataFrame(columns=["agent", "agent_name", "sla", "cases", "breach_rate", "expected", "excess_pp"])
    out = pd.concat(rows, ignore_index=True)
    out["excess_pp"] = 100 * (out["breach_rate"] - out["expected"])
    return out
