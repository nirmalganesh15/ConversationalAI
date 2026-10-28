"""Download the public-domain healthcare corpus used by the Docs agent.

Sources (all US-government works, public domain):
  - openFDA drug labels (FDA Structured Product Labeling)       https://open.fda.gov/
  - MedlinePlus health-topic summaries (National Library of Med.) https://medlineplus.gov/
  - ClinicalTrials.gov study records (NLM)                          https://clinicaltrials.gov/

Usage:  uv run python scripts/fetch_samples.py [--out ../samples/docs]
Output is committed to the repo so builds are reproducible offline; re-run to refresh.
"""

import argparse
import html
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

DRUGS = {
    "metformin": "METFORMIN HYDROCHLORIDE",
    "lisinopril": "LISINOPRIL",
    "atorvastatin": "ATORVASTATIN CALCIUM",
    "amoxicillin": "AMOXICILLIN",
    "albuterol": "ALBUTEROL SULFATE",
    "warfarin": "WARFARIN SODIUM",
}
LABEL_SECTIONS = [
    ("indications_and_usage", "Indications and Usage"),
    ("dosage_and_administration", "Dosage and Administration"),
    ("dosage_forms_and_strengths", "Dosage Forms and Strengths"),
    ("contraindications", "Contraindications"),
    ("boxed_warning", "Boxed Warning"),
    ("warnings_and_cautions", "Warnings and Precautions"),
    ("warnings", "Warnings"),
    ("adverse_reactions", "Adverse Reactions"),
    ("drug_interactions", "Drug Interactions"),
    ("use_in_specific_populations", "Use in Specific Populations"),
    ("pregnancy", "Pregnancy"),
    ("overdosage", "Overdosage"),
]
TOPICS = {
    "type-2-diabetes": '"diabetes type 2"',
    "high-blood-pressure": '"high blood pressure"',
    "asthma": "asthma",
    "cholesterol": '"cholesterol"',
    "flu": '"flu"',
}
TRIAL_QUERY = "metformin type 2 diabetes"
MAX_SECTION_CHARS = 6000


def clean(text: str) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    text = re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n+", "\n\n", text))
    return re.sub(r"\n ", "\n", text).strip()


def fetch_label(client: httpx.Client, slug: str, generic: str) -> str | None:
    response = client.get(
        "https://api.fda.gov/drug/label.json",
        params={
            "search": f'openfda.generic_name.exact:"{generic}" AND _exists_:contraindications',
            "limit": 5,
        },
    )
    if response.status_code != 200:
        return None
    # Prefer a single-ingredient label (combination products confuse the demo).
    results = response.json()["results"]
    label = next((r for r in results if len(r["openfda"].get("generic_name", [])) == 1), results[0])
    brand = ", ".join(label["openfda"].get("brand_name", [])[:2])
    lines = [
        f"# {generic.title()} — FDA Prescribing Information",
        "",
        f"Source: openFDA drug label (set id {label.get('set_id', 'n/a')}), "
        f"brand(s): {brand or 'generic'}. Public domain (U.S. FDA).",
        "",
    ]
    for key, title in LABEL_SECTIONS:
        if key in label:
            body = clean(" ".join(label[key]))
            # Labels repeat the section number/title at the start of the body...
            body = re.sub(r"^\s*\d*(\.\d+)?\s*[A-Z][A-Z &]{3,}(?=\s)\s*", "", body)
            # ...and are littered with cross-reference markers like "( 2.1 )".
            body = re.sub(r"\(\s*[\d.,\s]+\)", "", body)
            # Start a new paragraph at each numbered subsection ("2.3 Renal Impairment").
            body = re.sub(r"\s(?=\d{1,2}\.\d{1,2}\s+[A-Z])", "\n\n", body)
            lines += [f"## {title}", "", body[:MAX_SECTION_CHARS], ""]
    return "\n".join(lines)


def fetch_topic(client: httpx.Client, term: str) -> tuple[str, str] | None:
    response = client.get(
        "https://wsearch.nlm.nih.gov/ws/query",
        params={"db": "healthTopics", "term": term, "retmax": 1, "rettype": "topic"},
    )
    if response.status_code != 200:
        return None
    root = ET.fromstring(response.text)  # noqa: S314
    topic = root.find(".//health-topic")
    summary_el = topic.find("full-summary") if topic is not None else None
    if topic is None or summary_el is None or not summary_el.text:
        return None
    title = topic.get("title", term)
    # The summary is escaped HTML: keep its <h3> headings and list items as markdown.
    summary = summary_el.text
    summary = re.sub(r"<h3>(.*?)</h3>", r"\n\n## \1\n\n", summary)
    summary = re.sub(r"<li>", "\n- ", summary)
    summary = re.sub(r"</p>|<br\s*/?>", "\n\n", summary)
    body = clean(summary)
    text = (
        f"# {title}\n\nSource: MedlinePlus health topic ({topic.get('url')}), "
        f"National Library of Medicine. Public domain.\n\n{body}\n"
    )
    return title, text


def fetch_trial(client: httpx.Client, query: str) -> tuple[str, str] | None:
    response = client.get(
        "https://clinicaltrials.gov/api/v2/studies",
        params={"query.term": query, "filter.overallStatus": "COMPLETED", "pageSize": 1},
    )
    if response.status_code != 200:
        return None
    study = response.json()["studies"][0]["protocolSection"]
    ident = study["identificationModule"]
    desc = study.get("descriptionModule", {})
    elig = study.get("eligibilityModule", {})
    arms = study.get("armsInterventionsModule", {})
    outcomes = study.get("outcomesModule", {})
    design = study.get("designModule", {})
    nct = ident["nctId"]
    enrollment = design.get("enrollmentInfo", {}).get("count", "n/a")
    interventions = "\n".join(
        f"- {i.get('type', '')}: {i.get('name', '')} — {i.get('description', '')}"
        for i in arms.get("interventions", [])
    )
    primary = "\n".join(
        f"- {o.get('measure', '')} ({o.get('timeFrame', '')})"
        for o in outcomes.get("primaryOutcomes", [])
    )
    text = f"""# Clinical Trial {nct}: {ident.get("briefTitle", "")}

Source: ClinicalTrials.gov record {nct} (https://clinicaltrials.gov/study/{nct}). Public domain.

## Official Title

{ident.get("officialTitle", "")}

## Brief Summary

{clean(desc.get("briefSummary", ""))}

## Detailed Description

{clean(desc.get("detailedDescription", ""))[:MAX_SECTION_CHARS]}

## Study Design

Phase: {", ".join(design.get("phases", []))}. Enrollment: {enrollment}.

## Interventions

{interventions}

## Eligibility Criteria

{clean(elig.get("eligibilityCriteria", ""))[:MAX_SECTION_CHARS]}

## Primary Outcomes

{primary}
"""
    return nct, text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parents[2] / "samples" / "docs"
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    with httpx.Client(timeout=30, headers={"User-Agent": "atlas-sample-fetcher/1.0"}) as client:
        for slug, generic in DRUGS.items():
            if text := fetch_label(client, slug, generic):
                (args.out / f"fda-label-{slug}.md").write_text(text, encoding="utf-8")
                written.append(f"fda-label-{slug}.md")
        for slug, term in TOPICS.items():
            if topic := fetch_topic(client, term):
                (args.out / f"medlineplus-{slug}.md").write_text(topic[1], encoding="utf-8")
                written.append(f"medlineplus-{slug}.md")
        if trial := fetch_trial(client, TRIAL_QUERY):
            nct, text = trial
            (args.out / f"clinical-trial-{nct.lower()}.md").write_text(text, encoding="utf-8")
            written.append(f"clinical-trial-{nct.lower()}.md")
    print("\n".join(written))
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
