# Sample data sources

## `docs/` — healthcare corpus (public domain)
Fetched and converted to Markdown by `backend/scripts/fetch_samples.py` (re-run to refresh).

| File | Source | Licence |
|---|---|---|
| `fda-label-{metformin,lisinopril,atorvastatin,amoxicillin,albuterol,warfarin}.md` | openFDA drug label API (FDA Structured Product Labeling), https://open.fda.gov/ | U.S. Government work, public domain |
| `medlineplus-{type-2-diabetes,high-blood-pressure,asthma,cholesterol,flu}.md` | MedlinePlus health topics web service, National Library of Medicine, https://medlineplus.gov/ | NLM-authored summaries, public domain |
| `clinical-trial-nct01422876.md` | ClinicalTrials.gov API v2, https://clinicaltrials.gov/study/NCT01422876 | NLM, public domain |

openFDA's disclaimer applies: the data is unvalidated and must not be used for medical decisions.
Atlas presents it as reference material only (see the medical-safety guardrail).

## `meetings/` — synthetic
Three transcripts written for this project (fictional people and studies): a clinical-trial
steering committee, a digital-health sprint review and a pharmacovigilance signal review. Format:
`[hh:mm:ss] Speaker: text`.

## `career/` — synthetic + public
- `resume-arjun-mehta.md` — fictional candidate.
- `jd-1-forward-deployed-engineer-newpage.md` — condensed from the Newpage FDE job description
  provided with the assignment.
- `jd-2-…`, `jd-3-…` — fictional job descriptions.
