"""Synthetic Salesforce exports for trying the app without real data.

SLA fields and business hours are pre-calculated on the Case, the way your org does it.

Planted on purpose, so you can check the app finds them:
  Agents 07 and 15  weak on prior auth (research and rework notes)
  Agent 22          weak on payment disputes (escalates)
  Agent 27          weak on credentialing
  Agent 31          slow on everything (new hire); Agent 30 somewhat slow
  Agent 19          fast
  Agents 03 and 11  average skill, but routed the hardest cases
"""
import copy

import numpy as np
import pandas as pd

import engine as E

CAL_PER_BH = 168 / 45     # calendar hours per business hour (9 h x 5 days a week)

TEMPLATES = {
    "claims_status": [
        "Checked claim status for DOS {date}. Claim {cid} pended for adjudication, gave expected processing timeline.",
        "Claim {cid} processed and paid, provided EFT trace number and remittance date.",
        "Claim denied for missing modifier, explained corrected claim resubmission process.",
    ],
    "eligibility": [
        "Verified eligibility and benefits for ID {mid}. Coverage active, gave copay deductible and coinsurance.",
        "Eligibility check shows plan termed, gave termination date and coordination of benefits info.",
        "Benefits verification for outpatient therapy visits, gave remaining visit limit and copay.",
    ],
    "prior_auth": [
        "Prior authorization status for MRI, auth pending clinical review, confirmed fax clinicals received.",
        "Procedure requires prior auth, gave fax number for clinical documentation, auth reference created.",
        "Authorization denied for medical necessity, explained peer to peer review and appeal options.",
    ],
    "provider_update": [
        "Demographic update request, new practice address and phone {phone}. Submitted roster update form.",
        "Updated billing address and TIN, directory correction request submitted to provider data team.",
        "Practice location add for new clinic site, roster form received, directory update pending.",
    ],
    "payment_dispute": [
        "Disputing payment amount, states underpaid per contract fee schedule. Opened reconsideration request.",
        "Recoupment letter received, requests overpayment review and offset details on remittance.",
        "Payment dispute on bundled services, reviewed contract rates, reconsideration submitted.",
    ],
    "credentialing": [
        "Credentialing application status, application in committee review, gave estimated decision date.",
        "Recredentialing packet missing CAQH attestation and malpractice certificate, asked to update CAQH.",
        "New practitioner credentialing, application incomplete, missing board certification and license copy.",
    ],
}
FLAG_TEXT = {
    "research": "Not sure on this policy, need to research and verify with the policy manual.",
    "escalate": "Escalated to supervisor for review.",
    "transfer": "Warm transfer to specialty unit.",
    "callback": "Will call back with an update.",
    "rework": "This is the second time calling about the same issue.",
}
NAMES = ["Kim", "Lopez", "Patel", "Nguyen", "Okafor", "Smith", "Rossi", "Chen", "Haddad", "Brown"]

BASE_BH = {"claims_status": 3, "eligibility": 2, "prior_auth": 12, "provider_update": 6,
           "payment_dispute": 15, "credentialing": 20}
TYPE_OF = {"claims_status": "Claims", "eligibility": "Eligibility", "prior_auth": "Authorization",
           "provider_update": "Provider Data", "payment_dispute": "Claims", "credentialing": "Provider Data"}
DEPT_OF = {"claims_status": "Claims", "payment_dispute": "Claims", "eligibility": "Member Services",
           "prior_auth": "Clinical", "provider_update": "Network", "credentialing": "Network"}
DEPT_TARGET = {"Claims": 24, "Member Services": 8, "Clinical": 20, "Network": 40}
OPTUM_TARGET = {"High": 16, "Medium": 27, "Low": 45}
PRI_P = {"claims_status": [.2, .6, .2], "eligibility": [.3, .5, .2], "prior_auth": [.6, .3, .1],
         "provider_update": [.1, .4, .5], "payment_dispute": [.2, .5, .3], "credentialing": [.1, .4, .5]}

DEMO_SETTINGS = E.default_settings()
DEMO_SETTINGS.update({
    "closure_hours_field": "Business_Hours_To_Close__c",
    "extra_case_mix": ["Department__c", "Line_of_Business__c"],
    "n_topics": 6,
    "slas": [
        dict(E.SLA_TEMPLATE, name="Optum SLA", method="Due date field", field="Optum_SLA_Due_Date__c"),
        dict(E.SLA_TEMPLATE, name="GW SLA", method="Met/Missed field", field="GW_SLA_Status__c",
             breach_values="missed"),
        dict(E.SLA_TEMPLATE, name="Department SLA", method="Met/Missed field",
             field="Department_SLA_Met__c", breach_values="false"),
        dict(E.SLA_TEMPLATE, name="Internal SLA", method="Hours field vs target",
             field="Internal_SLA_Hours__c", target="27"),
    ],
})


def demo_settings():
    return copy.deepcopy(DEMO_SETTINGS)


def make_demo(n_cases: int = 6000, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    topics = list(TEMPLATES)
    agents = [f"005A{i:03d}" for i in range(1, 41)]
    now = pd.Timestamp("2026-10-01", tz="UTC")
    tenure_m = rng.integers(8, 72, len(agents))
    tenure_m[[30, 34]] = [2, 4]
    users = pd.DataFrame({"Id": agents, "Name": [f"Agent {i:02d}" for i in range(1, 41)],
                          "HireDate__c": [(now - pd.DateOffset(months=int(m))).date().isoformat()
                                          for m in tenure_m]})
    skill = {a: rng.normal(0, 0.12) for a in agents}
    weak = {("005A007", "prior_auth"): 0.75, ("005A015", "prior_auth"): 0.65,
            ("005A022", "payment_dispute"): 0.6, ("005A027", "credentialing"): 0.6}
    skill["005A031"] += 0.35
    skill["005A019"] -= 0.3
    hard_mix = {"005A003", "005A011"}

    ag = np.array(agents)[rng.integers(len(agents), size=n_cases)]
    tp = np.array([topics[rng.choice(6, p=[.05, .05, .15, .05, .35, .35] if a in hard_mix
                                      else [.25, .2, .2, .12, .13, .1])] for a in ag])
    pri = np.array([rng.choice(["High", "Medium", "Low"], p=PRI_P[t]) for t in tp])
    ch = rng.choice(["Phone", "Web", "Email"], p=[.6, .25, .15], size=n_cases)
    lob = rng.choice(["GW", "Commercial", "Medicaid"], p=[.35, .4, .25], size=n_cases)
    ten = tenure_m[[agents.index(a) for a in ag]]
    log_h = (np.log([BASE_BH[t] for t in tp]) + np.array([skill[a] for a in ag])
             + np.array([weak.get((a, t), 0) for a, t in zip(ag, tp)])
             + np.where(ch == "Email", 0.2, 0) + np.where(ten < 6, 0.2, 0)
             - np.where(pri == "High", 0.15, 0) + rng.normal(0, 0.45, n_cases))
    handle = np.exp(log_h)                                         # business hours of agent work
    wait = np.where(rng.random(n_cases) < .3, rng.exponential(10, n_cases), 0.0)

    t0 = pd.Timestamp("2026-04-01", tz=E.TZ)
    created = (t0 + pd.to_timedelta(rng.integers(0, 180 * 24 * 60, n_cases), unit="min")).to_series().reset_index(drop=True)
    closed = created + pd.to_timedelta((handle + wait) * CAL_PER_BH, unit="h")
    due = created + pd.to_timedelta(np.array([OPTUM_TARGET[p] for p in pri]) * CAL_PER_BH, unit="h")
    is_open = rng.random(n_cases) < .04
    dept = np.array([DEPT_OF[t] for t in tp])

    iso = lambda s: s.dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    ids = [f"500C{i:07d}" for i in range(n_cases)]
    cases = pd.DataFrame({
        "Id": ids, "OwnerId": ag,
        "Type": np.where(rng.random(n_cases) > .15, [TYPE_OF[t] for t in tp], "General Inquiry"),
        "Origin": ch, "Priority": pri, "Status": np.where(is_open, "Working", "Closed"),
        "CreatedDate": iso(created), "ClosedDate": np.where(is_open, "", iso(closed)),
        "Department__c": dept, "Line_of_Business__c": lob,
        "Business_Hours_To_Close__c": np.where(is_open, "", np.round(handle, 2).astype(str)),
        "Optum_SLA_Due_Date__c": iso(due),
        "GW_SLA_Status__c": np.where(lob == "GW", np.where(handle > 20, "Missed", "Met"), ""),
        "Department_SLA_Met__c": np.where(handle <= np.array([DEPT_TARGET[x] for x in dept]), "true", "false"),
        "Internal_SLA_Hours__c": np.round(handle, 2).astype(str),
    })
    cases.loc[is_open, ["GW_SLA_Status__c", "Department_SLA_Met__c", "Internal_SLA_Hours__c"]] = ""

    comments = []
    for i in range(n_cases):
        a, t = ag[i], tp[i]
        gap = weak.get((a, t), 0) > 0
        for _ in range(int(np.clip(rng.poisson(1 + np.log1p(handle[i]) / 2), 1, 8))):
            body = rng.choice(TEMPLATES[t]).format(
                date=f"{rng.integers(1, 13)}/{rng.integers(1, 28)}/2026", cid=f"C{rng.integers(10**9, 10**10)}",
                mid=f"M{rng.integers(10**8, 10**9)}",
                phone=f"(555) {rng.integers(200, 999)}-{rng.integers(1000, 9999)}")
            if rng.random() < .3:
                body = f"Spoke with {rng.choice(NAMES)} at the practice. " + body
            if rng.random() < (.55 if gap and a != "005A022" else .05):
                body += " " + FLAG_TEXT["research"]
            if rng.random() < (.45 if (a == "005A022" and t == "payment_dispute") else .04):
                body += " " + FLAG_TEXT["escalate"]
            if rng.random() < .06:
                body += " " + FLAG_TEXT["transfer"]
            if rng.random() < .15:
                body += " " + FLAG_TEXT["callback"]
            if rng.random() < (.2 if gap else .03):
                body += " " + FLAG_TEXT["rework"]
            comments.append((ids[i], body))
    comments = pd.DataFrame(comments, columns=["ParentId", "CommentBody"])

    w_start = created + pd.to_timedelta(handle * 0.4 * CAL_PER_BH, unit="h")
    w_end = w_start + pd.to_timedelta(wait * CAL_PER_BH, unit="h")
    m, c = wait > 0, ~is_open
    history = pd.concat([
        pd.DataFrame({"CaseId": ids, "Field": "Status", "OldValue": "New", "NewValue": "Working",
                      "CreatedDate": iso(created + pd.Timedelta(minutes=5))}),
        pd.DataFrame({"CaseId": np.array(ids)[m], "Field": "Status", "OldValue": "Working",
                      "NewValue": "Waiting on Provider", "CreatedDate": iso(w_start[m])}),
        pd.DataFrame({"CaseId": np.array(ids)[m], "Field": "Status", "OldValue": "Waiting on Provider",
                      "NewValue": "Working", "CreatedDate": iso(w_end[m])}),
        pd.DataFrame({"CaseId": np.array(ids)[c], "Field": "Status", "OldValue": "Working",
                      "NewValue": "Closed", "CreatedDate": iso(closed[c])}),
    ], ignore_index=True)

    tables = {"cases": cases, "comments": comments, "history": history, "users": users}
    return {k: v.astype(str).replace({"nan": np.nan, "": np.nan}) for k, v in tables.items()}


if __name__ == "__main__":
    from pathlib import Path
    out = Path("demo_data")
    out.mkdir(exist_ok=True)
    names = {"cases": "cases.csv", "comments": "case_comments.csv", "history": "case_history.csv",
             "users": "users.csv"}
    for k, df in make_demo().items():
        df.to_csv(out / names[k], index=False)
    print(f"Demo CSVs written to {out.resolve()}")
