"""
Dataset loading module — Marathi Legal Judgment Retrieval System.

Loads the reviewed judgment dataset and resolves field fallback chains so the
frontend never has to display a blank category or legal-principle field.

Fallback design rationale (fixes a real gap found in the raw dataset, where
many rows had empty verified-category / legal-principle fields):
    category         : पडताळलेली_श्रेणी (human-verified) > category_suggested
                        (LLM-assisted, has confidence score) > स्वयंचलित_श्रेणी
                        (keyword auto-classifier, least reliable)
    legal_principle   : legal_principle_draft (case-specific, drafted per
                        record) > कायदेशीर_तत्त्वे (category-level boilerplate,
                        used only as a last resort since it repeats verbatim
                        across many records of the same category)
    final_decision    : प्रमाणित_अंतिम_निर्णय (normalized/certified) >
                        अंतिम_निर्णय (raw, sometimes mixed-script)
"""
import pandas as pd
from pathlib import Path
from .preprocessing import clean_text

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "reviewed_final.csv"


def _first_nonempty(*values):
    for v in values:
        if isinstance(v, str) and v.strip() and v.strip().lower() != "nan":
            return v.strip()
    return ""


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    records = []
    for _, row in df.iterrows():
        category = _first_nonempty(
            row.get("पडताळलेली_श्रेणी"),
            row.get("category_suggested"),
            row.get("स्वयंचलित_श्रेणी"),
        ) or "इतर"

        legal_principle = _first_nonempty(
            row.get("legal_principle_draft"),
            row.get("कायदेशीर_तत्त्वे"),
        ) or "या निकालासाठी स्वतंत्र कायदेशीर तत्त्व नोंदवलेले नाही."

        final_decision = _first_nonempty(
            row.get("प्रमाणित_अंतिम_निर्णय"),
            row.get("अंतिम_निर्णय"),
        ) or "माहिती उपलब्ध नाही"

        case_text = clean_text(row.get("खटला_मजकूर", ""))

        records.append({
            "judgment_id": row.get("judgment_id", ""),
            "case_name": clean_text(row.get("खटल्याचे_नाव", "")),
            "court_name": clean_text(row.get("न्यायालयाचे_नाव", "")),
            "year": row.get("वर्ष", ""),
            "category": category,
            "category_confidence": row.get("category_confidence", ""),
            "case_text": case_text,
            "final_decision": final_decision,
            "legal_principle": legal_principle,
            "source_pdf_url": row.get("source_pdf_url", ""),
        })

    out = pd.DataFrame(records)
    out = out[out["case_text"].str.len() > 0].reset_index(drop=True)
    return out


if __name__ == "__main__":
    d = load_dataset()
    print(f"Loaded {len(d)} judgments")
    print(d["category"].value_counts())
    print(d.iloc[0])
