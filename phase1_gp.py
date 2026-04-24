import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold
import lightgbm as lgb

# ================================
# 📊 METRIC
# ================================
def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

# ================================
# 📦 LOAD DATA
# ================================
def load_data(path):
    xl = pd.ExcelFile(path)

    raw = xl.parse("Data Pack - Actual Bookings", header=None)

    prod = raw.iloc[2:31].copy()
    prod = prod[prod[1].notna()]

    product = prod[1].values
    lifecycle = prod[2].values

    actuals = prod.iloc[:,3:15].values

    f_dp = prod.iloc[:,16].values
    f_mkt = prod.iloc[:,17].values
    f_ds = prod.iloc[:,18].values

    return product, lifecycle, actuals, f_dp, f_mkt, f_ds

# ================================
# 🧠 BUILD DATASET
# ================================
def build_dataset(product, lifecycle, actuals):

    rows = []

    for i, p in enumerate(product):
        series = actuals[i]

        for t in range(4, len(series)):
            y = series[t]

            lag1 = series[t-1]
            lag2 = series[t-2]
            lag4 = series[t-4]

            if pd.isna(y) or pd.isna(lag1):
                continue

            # simulate "human forecasts"
            f_dp = lag1
            f_mkt = (lag1 + lag2)/2
            f_ds = lag4

            baseline = (f_dp + f_mkt + f_ds) / 3

            target = y / (baseline + 1e-6)
            target = np.clip(target, 0.3, 3.0)

            rows.append({
                "product": p,
                "lifecycle": lifecycle[i],
                "lag1": lag1,
                "lag2": lag2,
                "lag4": lag4,
                "f_dp": f_dp,
                "f_mkt": f_mkt,
                "f_ds": f_ds,
                "baseline": baseline,
                "target": target,
                "y": y
            })

    df = pd.DataFrame(rows)

    df["product_id"] = df["product"].astype("category").cat.codes
    df = pd.get_dummies(df, columns=["lifecycle"], prefix="lc")

    return df

# ================================
# 🧠 MODEL
# ================================
def train_model(df):

    features = [c for c in df.columns if c not in ["y","target","product"]]

    X = df[features]
    y = df["target"]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    oof = np.zeros(len(df))

    for tr, val in kf.split(X):

        model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31
        )

        model.fit(X.iloc[tr], y.iloc[tr])

        pred = model.predict(X.iloc[val])
        pred = np.clip(pred, 0.3, 3.0)

        oof[val] = pred * df.iloc[val]["baseline"].values

    return oof

# ================================
# 🚀 PIPELINE
# ================================
def run_pipeline(path):

    product, lifecycle, actuals, f_dp, f_mkt, f_ds = load_data(path)

    df = build_dataset(product, lifecycle, actuals)

    preds = train_model(df)

    print("\n🔥 FINAL MODEL (FORECAST CORRECTION + HUMAN BASELINE)")
    print("WMAPE:", wmape(df["y"].values, preds))

    return df, preds

# ================================
# ▶️ RUN
# ================================
run_pipeline("/CFL_External Data Pack_Phase1.xlsx")
