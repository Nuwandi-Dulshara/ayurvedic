import streamlit as st
import pandas as pd
import difflib
from streamlit_searchbox import st_searchbox   # 👈 new import

# ---------------------------
# Load data
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/symptoms.csv")
    symptoms = sorted(set(df["Symptom"].dropna().astype(str).tolist()))
    lookup = {s.strip().lower(): s for s in symptoms}
    return df, symptoms, lookup

df, SYMPTOMS, SYMPTOM_LOOKUP = load_data()

# ---------------------------
# Weights (0–10)
# ---------------------------
disease_group_weight = {
    "Incurable": 10, "Cardio-blood": 9, "Respiratory": 9,
    "Mental": 8, "Nervous": 8,
    "Digestive": 7, "Musculoskeletal": 7, "Urinary": 7,
    "Reproductive – female": 7, "Reproductive – male": 7,
    "Congenital": 6, "Obesity": 6, "Over-nutrition": 6, "Under-nutrition": 6,
    "Internal": 5, "Endogenous": 5, "Exogenous": 5, "Somatic": 5, "Psychosomatic": 5,
    "Seasonal": 4, "Sweat": 4, "Natural": 4, "External": 4, "Middle": 4,
    "Hereditary": 3, "thermoregulation": 3,
    "Curable": 2, "Metabolic": 2,
}

dosha_weight = {
    "Tridosha": 9, "Pitta and Kapha": 8,
    "Vata and Pitta": 7, "Vata and Kapha": 7,
    "Pitta": 6, "Vata": 5, "Kapha": 4,
}

W_GROUP, W_DOSHA = 0.6, 0.4

# ---------------------------
# Helpers
# ---------------------------
def resolve_exact(text: str):
    return SYMPTOM_LOOKUP.get((text or "").strip().lower())

def risk_level_from_score(score: float):
    if score < 4:
        return "🟢 Low Risk"
    elif score < 7:
        return "🟠 Medium Risk"
    else:
        return "🔴 High Risk"

def compute_risk(symptom_text: str, w_group=W_GROUP, w_dosha=W_DOSHA):
    canon = resolve_exact(symptom_text)
    if not canon:
        return {"found": False, "message": "⚠️ Symptom not found. Please select from suggestions or type exact."}

    row = df.loc[df["Symptom"] == canon].iloc[0]
    disease_name = str(row["Disease"])
    group_en = str(row["Disease group (English name)"])
    group_si = str(row["Disease group (Sinhala name)"])
    dosha = str(row["Dosha types"])

    g_w = float(disease_group_weight.get(group_en, 0.0))
    d_w = float(dosha_weight.get(dosha, 0.0))

    score = round(w_group * g_w + w_dosha * d_w, 2)
    level = risk_level_from_score(score)

    return {
        "found": True,
        "symptom": canon,
        "disease_name": disease_name,
        "disease_group_en": group_en,
        "disease_group_si": group_si,
        "dosha": dosha,
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

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Ayurvedic Risk Predictor 🌿", page_icon="🌿", layout="centered")

st.title("🌿 Ayurvedic Disease Dosha and Risk Predictor")
st.markdown("🔍 *Type a symptom, pick from auto-suggestions, and see the predicted disease group, dosha, and risk level.*")

# --- Autocomplete searchbox ---
def search_symptoms(q):
    return get_suggestions(q, k=12)

picked = st_searchbox(
    search_symptoms,
    key="symptom_search",
    placeholder="✏️ Type or search a symptom…"
)

# Predict button
if st.button("🚀 Predict Risk", type="primary"):
    res = compute_risk(picked)
    if not res["found"]:
        st.error(res["message"])
    else:
        st.markdown("## 📊 Risk Assessment")
        st.write(f"**📝 Symptom:** `{res['symptom']}`")
        st.write(f"**🌱 Disease (Ayurveda):** `{res['disease_name']}`")
        st.write(f"**📖 Disease Group (English):** `{res['disease_group_en']}`")
        st.write(f"**📖 Disease Group (Sinhala):** `{res['disease_group_si']}`")
        st.write(f"**🔥 Dosha Type:** `{res['dosha']}`")

        st.divider()
        st.markdown("### ⚖️ Calculation Details")
        st.write(f"**Disease Group Weight:** `{res['group_weight']}`")
        st.write(f"**Dosha Weight:** `{res['dosha_weight']}`")
        st.write(f"**Formula:** `{res['formula']}`")

        st.metric("Risk Score (0–10)", res["risk_score_0_10"])
        st.subheader(f"🏥 Final Risk Level: {res['risk_level']}")
