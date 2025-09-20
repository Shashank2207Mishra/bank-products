# app.py — Creative UI version (no debug or file status)
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pandas.api.types import is_numeric_dtype, is_string_dtype, is_object_dtype
from typing import Dict, Optional
import traceback

# ---------------- Config ----------------
CWD = Path.cwd()
MODEL_DIRS = [CWD, CWD / "model_outputs", CWD / "xgb_tuning_outputs", CWD / "xgb_models_optuna"]
TARGETS = [
    'interested_credit_card',
    'interested_personal_loan',
    'interested_home_loan',
    'interested_mutual_fund',
    'interested_insurance',
    'interested_fd'
]
MODEL_FILE_TEMPLATE = "{}_xgb_best_model.joblib"
DEFAULT_SAMPLE_CSV = "personal_banking_synthetic.csv"

# quietly ensure folders exist (no UI noise)
for d in MODEL_DIRS:
    try:
        Path(d).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

# ---------------- Helpers ----------------
def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required = ['avg_balance_k','annual_income_k','avg_monthly_spend_k','age','occupation',
                'credit_card_owned','personal_loan_owned','home_loan_owned','fd_owned',
                'mutual_fund_owned','insurance_owned']
    for c in required:
        if c not in df.columns:
            df[c] = 0 if c != 'occupation' else "Unknown"
    for c in ['avg_balance_k','annual_income_k','avg_monthly_spend_k','age']:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(float)
    prod_cols = ['credit_card_owned','personal_loan_owned','home_loan_owned','fd_owned','mutual_fund_owned','insurance_owned']
    for c in prod_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)
    df['balance_to_income_ratio'] = df['avg_balance_k'] / (df['annual_income_k'] + 1.0)
    df['spending_to_income_ratio'] = (df['avg_monthly_spend_k'] * 12.0) / (df['annual_income_k'] + 1.0)
    df['total_products_owned'] = df[prod_cols].sum(axis=1).astype(int)
    df['is_young_professional'] = ((df['age'] < 30) & df['occupation'].isin(['Salaried','Self-employed'])).astype(int)
    df['is_high_value_customer'] = ((df['annual_income_k'] > 1000) & (df['avg_balance_k'] > 2000)).astype(int)
    return df

def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        try:
            if is_numeric_dtype(df[col]):
                continue
            if is_object_dtype(df[col]) or is_string_dtype(df[col]):
                coerced = pd.to_numeric(df[col], errors='coerce')
                if coerced.notna().mean() >= 0.9:
                    df[col] = coerced
                    continue
                df[col] = df[col].astype("string")
            else:
                df[col] = df[col].astype("string")
        except Exception:
            df[col] = df[col].apply(lambda x: "" if pd.isna(x) else str(x)).astype("string")
    return df.reset_index(drop=True)

def find_model_files() -> Dict[str, Optional[str]]:
    found = {t: None for t in TARGETS}
    for d in MODEL_DIRS:
        dd = Path(d)
        if not dd.exists():
            continue
        for t in TARGETS:
            candidate = dd / MODEL_FILE_TEMPLATE.format(t)
            if candidate.exists():
                # prefer project root copy if present
                if found[t] is None or dd == CWD:
                    found[t] = str(candidate)
    return found

def safe_load_pipelines(paths_map):
    pipelines = {}
    for t, p in paths_map.items():
        if p is None:
            pipelines[t] = None
            continue
        try:
            pipelines[t] = joblib.load(p)
        except Exception:
            pipelines[t] = None
    return pipelines

def try_predict_proba(pipeline, X):
    # best-effort: try pipeline.predict_proba; silently fail (return None) if anything breaks
    try:
        probs = pipeline.predict_proba(X)[:,1]
        return np.asarray(probs)
    except Exception:
        # silent fallback attempts (no tracebacks shown)
        try:
            steps = getattr(pipeline, "named_steps", None)
            if steps and 'preprocessor' in steps:
                pre = steps['preprocessor']
                Xt = pre.transform(X)
                # find classifier after preprocessor
                clf = None
                for name, step in steps.items():
                    if name == 'preprocessor':
                        continue
                    clf = step
                    break
                if clf is not None:
                    try:
                        probs = clf.predict_proba(Xt)[:,1]
                        return np.asarray(probs)
                    except Exception:
                        pass
        except Exception:
            pass
        try:
            preds = pipeline.predict(X)
            return np.array([float(x) for x in preds])
        except Exception:
            return None

def predict_single(pipelines, input_df):
    results = {}
    for t, pipe in pipelines.items():
        if pipe is None:
            results[t] = None
            continue
        res = try_predict_proba(pipe, input_df)
        results[t] = float(res[0]) if (res is not None and hasattr(res, "__len__")) else (float(res) if isinstance(res, (int,float)) else None)
    return results

def predict_batch(pipelines, df):
    out = df.copy()
    for t, pipe in pipelines.items():
        col = f"{t}_prob"
        if pipe is None:
            out[col] = np.nan
            continue
        res = try_predict_proba(pipe, df)
        out[col] = res if (res is not None and hasattr(res, "__len__")) else np.nan
    prob_cols = [f"{t}_prob" for t in TARGETS]
    def top3(row):
        vals = [(-9999 if pd.isna(row[c]) else row[c]) for c in prob_cols]
        idx = np.argsort(vals)[-3:][::-1]
        names = [TARGETS[i].replace("interested_","").replace("_"," ").title() for i in idx]
        return ", ".join(names)
    out['top_3_products'] = out.apply(top3, axis=1)
    return out

# ---------------- Load models quietly ----------------
model_paths = find_model_files()
pipelines = safe_load_pipelines(model_paths)

# ---------------- UI styling ----------------
st.set_page_config(page_title="Banking Recommender", layout="wide")
st.markdown("<style> .stApp { background-color: #0f1720; color: #e6eef8 } .card { background:#0b1220; padding:14px; border-radius:8px; box-shadow: 0 2px 10px rgba(0,0,0,0.6);} .product-name{font-weight:700; font-size:18px;} </style>", unsafe_allow_html=True)

# Header
col1, col2 = st.columns([3,1])
with col1:
    st.title("✨ Personalized Banking Product Recommender")
    st.markdown("Recommend the top banking products (cards, loans, investments) for customers based on profile & behavior.")
with col2:
    st.image("https://img.icons8.com/ios-filled/100/ffffff/bank-cards.png", width=80)

st.markdown("---")

# Layout: sidebar for inputs, main for results
st.sidebar.header("Customer input")
mode = st.sidebar.radio("Mode", ["Single customer", "Batch CSV"])

# Sidebar form fields (same as synthetic dataset)
def build_sidebar_form():
    age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.sidebar.selectbox("Gender", ["Male","Female","Other"])
    marital_status = st.sidebar.selectbox("Marital status", ["Single","Married","Divorced","Widowed"])
    education = st.sidebar.selectbox("Education", ["High School","Bachelors","Masters","PhD","Other"])
    occupation = st.sidebar.selectbox("Occupation", ["Salaried","Self-employed","Student","Retired","Homemaker","Business Owner"])
    city_tier = st.sidebar.selectbox("City tier", [1,2,3], index=0)
    annual_income_k = st.sidebar.number_input("Annual income (k INR)", min_value=0, value=300)
    account_years = st.sidebar.number_input("Years with bank", min_value=0.0, value=2.0, step=0.1)
    avg_balance_k = st.sidebar.number_input("Average balance (k INR)", min_value=0.0, value=50.0, step=0.5)
    num_transactions_month = st.sidebar.number_input("Monthly transactions", min_value=0, value=20)
    mobile_banking = st.sidebar.selectbox("Uses mobile banking?", [0,1], index=1)
    online_txn_pct = st.sidebar.slider("Online txn %", 0.0, 1.0, 0.4, step=0.01)
    avg_monthly_spend_k = st.sidebar.number_input("Avg monthly spend (k INR)", min_value=0.0, value=25.0)
    travel_spend_pct = st.sidebar.slider("Travel spend %", 0.0, 1.0, 0.05)
    groceries_spend_pct = st.sidebar.slider("Groceries spend %", 0.0, 1.0, 0.25)
    entertainment_spend_pct = st.sidebar.slider("Entertainment spend %", 0.0, 1.0, 0.08)
    other_spend_pct = max(0.0, 1.0 - (travel_spend_pct + groceries_spend_pct + entertainment_spend_pct))
    credit_card_owned = st.sidebar.selectbox("Credit card owned?", [0,1], index=0)
    personal_loan_owned = st.sidebar.selectbox("Personal loan owned?", [0,1], index=0)
    home_loan_owned = st.sidebar.selectbox("Home loan owned?", [0,1], index=0)
    fd_owned = st.sidebar.selectbox("Fixed deposit owned?", [0,1], index=0)
    mutual_fund_owned = st.sidebar.selectbox("Mutual fund owned?", [0,1], index=0)
    insurance_owned = st.sidebar.selectbox("Insurance owned?", [0,1], index=0)
    responded_to_campaign = st.sidebar.selectbox("Responded to past campaign?", [0,1], index=0)
    last_offer_days = st.sidebar.number_input("Days since last offer", min_value=0, value=365)
    row = {
        'age': age, 'gender': gender, 'marital_status': marital_status, 'education': education,
        'occupation': occupation, 'city_tier': city_tier, 'annual_income_k': annual_income_k,
        'account_years': account_years, 'avg_balance_k': avg_balance_k, 'num_transactions_month': num_transactions_month,
        'mobile_banking': mobile_banking, 'online_txn_pct': online_txn_pct, 'avg_monthly_spend_k': avg_monthly_spend_k,
        'travel_spend_pct': travel_spend_pct, 'groceries_spend_pct': groceries_spend_pct,
        'entertainment_spend_pct': entertainment_spend_pct, 'other_spend_pct': other_spend_pct,
        'credit_card_owned': credit_card_owned, 'personal_loan_owned': personal_loan_owned,
        'home_loan_owned': home_loan_owned, 'fd_owned': fd_owned, 'mutual_fund_owned': mutual_fund_owned,
        'insurance_owned': insurance_owned, 'responded_to_campaign': responded_to_campaign,
        'last_offer_days': last_offer_days
    }
    return pd.DataFrame([row])

# Main area: show sample or upload area
if mode == "Single customer":
    st.subheader("Predict for a single customer")
    input_df = build_sidebar_form()
    input_df = create_engineered_features(input_df)
    safe_input = sanitize_dataframe(input_df)
    st.markdown("#### Input preview")
    st.dataframe(safe_input, use_container_width=True)

    if st.button("Get recommendations"):
        # compute predictions
        preds = predict_single(pipelines, safe_input)
        # prepare display: list of (product, prob)
        rows = []
        for t, p in preds.items():
            product_name = t.replace("interested_","").replace("_"," ").title()
            pct = None if p is None else int(round(p * 100))
            rows.append((product_name, pct))
        # sort by probability desc (None last)
        rows_sorted = sorted(rows, key=lambda x: (-9999 if x[1] is None else -x[1], x[0]))
        # show results in cards with progress bars
        st.markdown("### Recommendations")
        topk = rows_sorted[:3]
        cols = st.columns(3)
        for i, (prod, prob) in enumerate(topk):
            with cols[i]:
                st.markdown(f"<div class='card'><div class='product-name'>{prod}</div>", unsafe_allow_html=True)
                if prob is None:
                    st.markdown("<div style='color:#f1c40f'>Model unavailable</div></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='font-size:24px; font-weight:700'>{prob}%</div>", unsafe_allow_html=True)
                    st.progress(int(prob))
                    # subtle CTA
                    if prob >= 70:
                        st.success("High likelihood — prioritize contact")
                    elif prob >= 40:
                        st.info("Moderate likelihood — nurture with offer")
                    else:
                        st.write("Low likelihood")
                    st.markdown("</div>", unsafe_allow_html=True)

        # full table on right
        st.markdown("### All product probabilities")
        table_rows = []
        for prod, prob in rows_sorted:
            table_rows.append({'product': prod, 'probability_percent': ("N/A" if prob is None else f"{prob}%")})
        st.table(pd.DataFrame(table_rows))

else:
    st.subheader("Batch predictions (CSV)")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    batch_df = None
    if uploaded:
        try:
            batch_df = pd.read_csv(uploaded)
            st.success(f"Loaded {len(batch_df)} rows")
        except Exception:
            st.error("Could not read uploaded CSV. Please ensure it's a valid CSV with correct column names.")
            batch_df = None
    else:
        if Path(DEFAULT_SAMPLE_CSV).exists():
            if st.button("Use sample dataset"):
                batch_df = pd.read_csv(DEFAULT_SAMPLE_CSV)
                for t in TARGETS:
                    if t in batch_df.columns:
                        batch_df = batch_df.drop(columns=[t])

    if batch_df is not None:
        batch_df = create_engineered_features(batch_df)
        safe_batch = sanitize_dataframe(batch_df)
        st.markdown("#### Preview (first 10 rows)")
        st.dataframe(safe_batch.head(10), use_container_width=True)

        if st.button("Run batch predictions"):
            st.info("Running predictions...")
            result_df = predict_batch(pipelines, safe_batch)
            # show summary top-3 counts
            st.markdown("### Top-3 product distribution")
            top3_counts = result_df['top_3_products'].value_counts().head(10)
            st.bar_chart(top3_counts)

            st.markdown("#### Sample results")
            st.dataframe(result_df.head(10), use_container_width=True)

            csv_bytes = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download results CSV", data=csv_bytes, file_name="batch_predictions.csv", mime="text/csv")
            st.success("Batch predictions finished")

# Footer notes (no debug info)
st.markdown("---")
st.markdown(
    """
    **Notes**
    - Models should be saved as joblib pipelines with preprocessing included and placed in the project root
      (or `model_outputs/`, `xgb_tuning_outputs/`, `xgb_models_optuna/`).
    - Filenames expected: `<target>_xgb_best_model.joblib` where `<target>` is e.g. `interested_credit_card`.
    - This UI hides internal debug messages to keep the experience clean; check logs or run offline tests if models fail.
    """
)
