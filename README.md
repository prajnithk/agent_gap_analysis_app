# Agent Gap Analysis (Streamlit)

The agent gap analysis in a browser UI. It shows which agents need coaching on which topics, using case comments, closure time and your SLA fields. It runs locally in Python, uses no LLM, and nothing leaves your machine.

## Run it

```bash
pip install -r requirements.txt      # Python 3.10 or newer
streamlit run app.py
```

Your browser opens at http://localhost:8501. Start with **Demo data** in the sidebar, then switch to **Folder on this computer** (or **Upload CSV files**) for your own exports.

## Exports

Use Data Loader or Workbench. Header names are matched without regard to case, and dates are read as Mountain Time.

| File | Need | SOQL |
|---|---|---|
| `cases.csv` | Required | `SELECT Id, OwnerId, Owner.Name, Type, Origin, Priority, Status, CreatedDate, ClosedDate, <your SLA fields>, <your business-hours-to-close field>, <Department / LOB fields> FROM Case WHERE CreatedDate = LAST_N_MONTHS:6` |
| `case_comments.csv` | Recommended | `SELECT ParentId, CommentBody FROM CaseComment WHERE ParentId IN (SELECT Id FROM Case WHERE CreatedDate = LAST_N_MONTHS:6)` |
| `case_history.csv` | Optional | `SELECT CaseId, Field, OldValue, NewValue, CreatedDate FROM CaseHistory WHERE Field = 'Status' AND CaseId IN (...)` (only used when closure time comes from the dates) |
| `users.csv` | Optional | `SELECT Id, Name FROM User WHERE IsActive = true` |

## Setup tab

**SLAs.** The app reads your SLA fields as they are; your org already applies business hours. On first load it suggests one SLA per field group (for example `Optum_SLA_Status__c` becomes "Optum SLA"). Check the suggestions and press **Apply settings**.

| Method | When missed |
|---|---|
| Met/Missed field | The field has one of the values you list (for example `missed`, or `false` for an SLA Met checkbox) |
| Due date field | ClosedDate is after the due date in the field |
| Hours field vs target | The hours in the field are above the target (a number, or a field that holds the target) |

A blank value means that SLA doesn't apply to the case, so an SLA that only covers some cases works without extra setup.

**Closure time.** Pick the Case field where your org stores business hours to close. If you leave it empty, the app uses CreatedDate to ClosedDate and, with `case_history.csv`, removes time in waiting statuses.

Settings are saved to `settings.json` next to the app. It holds field names only, no data.

## Tabs

| Tab | What it shows |
|---|---|
| Overview | Miss rate for each SLA, weekly trend, miss rate by topic × SLA, top coaching opportunities |
| Coaching priorities | Ranked agent × topic gaps with evidence |
| Skill matrix | Agent × topic heatmap, against the team or against the agent's own pace |
| Agent | One agent: pace before and after case mix, gaps by topic, notes vs team, every SLA, example notes. Also a table of all agents |
| Drivers | Logistic regression (what drives SLA misses) and linear regression (what drives closure time) |
| Topics | Systemic topics (fix process or KB) vs agent-level topics (coach); rename topics here |
| Cohorts | Agents grouped by shared gaps, for group training |
| Setup | SLAs, closure time, field mapping, keywords, thresholds |

**Download results (Excel)** in the sidebar exports every table.

## Handle with care

- Exports contain PHI. Run the app only on an approved, encrypted machine. It listens on localhost only and doesn't send usage statistics (see `.streamlit/config.toml`).
- Example notes have emails, phone numbers, dates and ID numbers removed. Names are **not** removed.
- Outputs name agents. Use them for coaching, not discipline. The models show associations, not causes.

## Files

- `app.py`: the Streamlit UI
- `engine.py`: the analysis (topics, logistic and linear regression, mixed-effects model, gaps, cohorts)
- `demo.py`: synthetic data with planted weak spots. Run `python demo.py` to write the demo CSVs to `demo_data/`
