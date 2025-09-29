import re
import difflib
import pandas as pd
import streamlit as st
from streamlit_searchbox import st_searchbox

# =========================
# Config
# =========================
st.set_page_config(page_title="Ayurvedic Risk Predictor 🌿", page_icon="🌿", layout="centered")
st.title("🌿 Ayurvedic Risk Predictor")
st.markdown("🔍 *Type a symptom, pick from auto-suggestions, and see the predicted **Common disease group**, **Disease Group**, **Dosha (cleaned)**, and **risk level***")

# =========================
# Weights (0–10) — as requested
# =========================
disease_group_weight = {
    "Urinary tract infections":              4,
    "Muscular disorders":                    5,
    "Cardiomyopathies":                      7,
    "Cardiovascular diseases":               9,
    "Ear diseases":                          3,
    "Eye diseases":                          4,
    "Hematological diseases":                6,
    "Liver disease":                         7,
    "Mental health / Psychiatric disorders": 6,
    "Nutritional Deficiency Diseases":       4,
    "Reproductive system diseases":          5,
    "Tropical diseases":                     6,
    "Endocrine and Metabolic Diseases":      7,
    "Cancer and neoplasms":                  9,
    "Zoonotic diseases":                     6,
}

dosha_weight = {
    "vata":        7.5,
    "pitta":       8.0,
    "kapha":       6.5,
    "vata|pitta":  8.5,
    "vata|kapha":  7.0,
    "pitta|kapha": 8.0,
    "tridosha":    9.5,
}

W_GROUP, W_DOSHA = 0.6, 0.4

# =========================
# Dosha normalization
# =========================
ORDER = ["vata", "pitta", "kapha"]
ORDER_SET = set(ORDER)

def normalize_dosha(x):
    if pd.isna(x):
        return pd.NA
    s = str(x).strip().lower()

    # Normalize tridosha variants
    if "tridosha" in s or "trisosha" in s or re.search(r"\btri\s*dosha\b", s):
        return "tridosha"

    # Unify separators, strip spaces & junk
    s = re.sub(r"[;,+/]+", "|", s)
    s = s.replace(" ", "")
    s = re.sub(r"[^a-z|]", "", s)

    parts = [p for p in s.split("|") if p]
    parts = [p for p in parts if p in ORDER_SET]
    if not parts:
        return pd.NA

    present = [d for d in ORDER if d in parts]
    if len(present) >= 3:
        return "tridosha"
    return "|".join(present)

# =========================
# Load data
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("data/symptoms.csv")
    df.rename(columns=lambda c: str(c).strip(), inplace=True)

    # Symptom column (support both)
    symptom_col = "Symptom" if "Symptom" in df.columns else ("Symptoms" if "Symptoms" in df.columns else None)
    if symptom_col is None:
        raise KeyError("CSV must include 'Symptom' or 'Symptoms' column.")
    df[symptom_col] = df[symptom_col].astype(str)

    # Mandatory columns
    if "Common disease group" not in df.columns:
        raise KeyError("CSV must include 'Common disease group' column.")
    if "Disease Group" not in df.columns:
        raise KeyError("CSV must include 'Disease Group' column.")

    # Ensure Dosha_Clean exists
    if "Dosha_Clean" not in df.columns:
        dosha_src = "Dosha Types" if "Dosha Types" in df.columns else ("Dosha types" if "Dosha types" in df.columns else None)
        if dosha_src is None:
            raise KeyError("CSV must include 'Dosha_Clean' or a raw dosha column ('Dosha Types' or 'Dosha types').")
        df["Dosha_Clean"] = df[dosha_src].apply(normalize_dosha)
    else:
        # Normalize again to be safe (handles casing/spacing)
        df["Dosha_Clean"] = df["Dosha_Clean"].apply(normalize_dosha)

    # Build suggestions
    symptoms = sorted(set(df[symptom_col].dropna().astype(str).tolist()))
    lookup = {s.strip().lower(): s for s in symptoms}
    return df, symptom_col, symptoms, lookup

df, SYMPTOM_COL, SYMPTOMS, SYMPTOM_LOOKUP = load_data()

# =========================
# Helpers
# =========================
def resolve_exact(text: str):
    return SYMPTOM_LOOKUP.get((text or "").strip().lower())

def risk_level_from_score(score: float):
    if score < 4:
        return "🟢 Low"
    elif score < 7:
        return "🟠 Medium"
    else:
        return "🔴 High"

def compute_risk(symptom_text: str, w_group=W_GROUP, w_dosha=W_DOSHA):
    canon = resolve_exact(symptom_text)
    if not canon:
        return {"found": False, "message": "⚠️ Symptom not found. Pick from suggestions or type an exact value from the dataset."}

    row = df.loc[df[SYMPTOM_COL] == canon].iloc[0]

    common_group = str(row["Common disease group"]).strip()
    disease_group = str(row["Disease Group"]).strip()
    dosha_clean  = str(row["Dosha_Clean"]) if pd.notna(row["Dosha_Clean"]) else ""

    g_w = float(disease_group_weight.get(common_group, 0.0))
    d_w = float(dosha_weight.get(dosha_clean, 0.0))

    score = round(w_group * g_w + w_dosha * d_w, 2)
    level = risk_level_from_score(score)

    return {
        "found": True,
        "symptom": canon,
        "common_group": common_group,
        "disease_group": disease_group,
        "dosha_clean": dosha_clean if dosha_clean else "-",
        "group_weight": g_w,
        "dosha_weight": d_w,
        "formula": f"Risk = {w_group}×{g_w} + {w_dosha}×{d_w}",
        "risk_score_0_10": score,
        "risk_level": level
    }

def get_suggestions(query, k=12):
    q = (query or "").strip().lower()
    if not q:
        return []
    prefix = [s for s in SYMPTOMS if s.lower().startswith(q)]
    substr = [s for s in SYMPTOMS if q in s.lower() and s not in prefix]
    fuzzy  = difflib.get_close_matches(q, SYMPTOMS, n=k*2, cutoff=0.6)
    fuzzy  = [s for s in fuzzy if s not in prefix and s not in substr]
    out, seen = [], set()
    for s in prefix + substr + fuzzy:
        if s not in seen:
            out.append(s); seen.add(s)
        if len(out) >= k:
            break
    return out

# =========================
# UI — Search + Predict
# =========================
st.markdown("### 🔎 Search a Symptom")

def search_symptoms(q):
    return get_suggestions(q, k=12)

picked = st_searchbox(
    search_symptoms,
    key="symptom_search",
    placeholder="✏️ Type or search a symptom…",
)

if st.button("🚀 Predict Risk", type="primary"):
    res = compute_risk(picked)
    if not res["found"]:
        st.error(res["message"])
    else:
        st.markdown("## 📊 Risk Assessment")
        st.write(f"**📝 Symptom:** `{res['symptom']}`")
        st.write(f"**📚 Common Disease Group:** `{res['common_group']}`")
        st.write(f"**🧩 Disease Group:** `{res['disease_group']}`")
        st.write(f"**🔥 Dosha (cleaned):** `{res['dosha_clean']}`")

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Group Weight", res["group_weight"])
            st.write(f"**W_group:** {W_GROUP}")
        with col2:
            st.metric("Dosha Weight", res["dosha_weight"])
            st.write(f"**W_dosha:** {W_DOSHA}")

        st.markdown("### ⚖️ Calculation")
        st.code(res["formula"])
        st.metric("Risk Score (0–10)", res["risk_score_0_10"])
        st.subheader(f"🏥 Final Risk Level: {res['risk_level']}")
