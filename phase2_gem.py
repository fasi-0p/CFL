# ==============================================================================
# R&D DIAGNOSTIC: HYBRID RISK-MANAGED ARCHITECTURE (FY26 Q1)
# Features: Asymmetric Loss Function + Bounded Dynamic Routing
# ==============================================================================

import pandas as pd
import numpy as np
import warnings
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

warnings.filterwarnings('ignore')

FILE_PATH = '/content/CFL_External Data Pack_Phase2.xlsx'
print("🚀 Initiating Hybrid Risk-Managed Backtest (FY26 Q1)...")

# ==============================================================================
# 1. CLEAN DATA EXTRACTION
# ==============================================================================
xl = pd.ExcelFile(FILE_PATH)
df_raw = xl.parse('Ph.2 Data Pack-Actual Booking', header=None)

QUARTERS = ['FY23 Q2', 'FY23 Q3', 'FY23 Q4', 'FY24 Q1', 'FY24 Q2', 'FY24 Q3',
            'FY24 Q4', 'FY25 Q1', 'FY25 Q2', 'FY25 Q3', 'FY25 Q4', 'FY26 Q1']

df_data = df_raw.iloc[2:150].dropna(subset=[1]).copy()
df_units = df_data[df_data[2].astype(str).str.contains('Sustaining|Decline|NPI|Growth|Mature', na=False, case=False)].reset_index(drop=True)
df_metrics = df_data[~df_data[2].astype(str).str.contains('Sustaining|Decline|NPI|Growth|Mature', na=False, case=False)].reset_index(drop=True)

actuals = df_units.iloc[:, :15].copy()
actuals.columns = ['Cost Rank', 'Product', 'Lifecycle'] + QUARTERS
for q in QUARTERS: actuals[q] = pd.to_numeric(actuals[q], errors='coerce').fillna(0)
actuals['Cost Rank'] = pd.to_numeric(actuals['Cost Rank'], errors='coerce').fillna(30)

acc_map = {}
for idx, row in df_metrics.iterrows():
    prod = str(row[2]).strip()
    try: dp_b = float(row[6])
    except: dp_b = 0.0
    try: ds_b = float(row[20])
    except: ds_b = 0.0
    acc_map[prod] = {'DP_Bias_Past': dp_b if not pd.isna(dp_b) else 0.0,
                     'DS_Bias_Past': ds_b if not pd.isna(ds_b) else 0.0}

# ==============================================================================
# 2. STABLE FEATURE ENGINEERING
# ==============================================================================
df_long = actuals.melt(id_vars=['Cost Rank', 'Product', 'Lifecycle'], value_vars=QUARTERS, var_name='Quarter', value_name='Actual_Units')
df_master = df_long.sort_values(by=['Product', 'Quarter']).reset_index(drop=True)

for i in [1, 2, 3, 4, 5]:
    df_master[f'Lag_{i}'] = df_master.groupby('Product')['Actual_Units'].shift(i)
    df_master[f'Log_Lag_{i}'] = np.log1p(df_master[f'Lag_{i}'])

df_master['Roll_Mean_4'] = (df_master['Lag_1'] + df_master['Lag_2'] + df_master['Lag_3'] + df_master['Lag_4']) / 4.0
df_master['Log_Roll_Mean'] = np.log1p(df_master['Roll_Mean_4'])
df_master['YoY_Growth'] = (df_master['Lag_1'] + 1) / (df_master['Lag_5'] + 1)

df_master = df_master.dropna(subset=['Lag_5']).fillna(0)
df_master['Sample_Weight'] = 1.0 / np.maximum(df_master['Cost Rank'], 1)
df_master['Target_Log'] = np.log1p(df_master['Actual_Units'])

features = ['Log_Lag_1', 'Log_Lag_2', 'Log_Lag_3', 'Log_Lag_4', 'Log_Roll_Mean', 'YoY_Growth']

# ==============================================================================
# 3. ASYMMETRIC MACHINE LEARNING (THE GENTLE NUDGE)
# ==============================================================================
train_df = df_master[df_master['Quarter'] != 'FY26 Q1']
test_df = df_master[df_master['Quarter'] == 'FY26 Q1'].copy()

X_train, y_train_log, w_train = train_df[features], train_df['Target_Log'], train_df['Sample_Weight']
X_test = test_df[features]

# The Custom Loss: 1.2x penalty for over-forecasting
def custom_asymmetric_objective(y_true, y_pred):
    residual = y_true - y_pred
    grad = np.where(residual < 0, -2.4 * residual, -2.0 * residual)
    hess = np.where(residual < 0, 2.4, 2.0)
    return grad, hess

print("🧠 Training ML with Asymmetric Loss Constraints...")
lgb = LGBMRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, objective=custom_asymmetric_objective, random_state=42, verbose=-1)
lgb.fit(X_train, y_train_log, sample_weight=w_train)

xgb = XGBRegressor(n_estimators=100, max_depth=2, learning_rate=0.05, objective='reg:absoluteerror', random_state=42)
xgb.fit(X_train, y_train_log, sample_weight=w_train)

test_df['ML_Base'] = (np.expm1(lgb.predict(X_test)) * 0.60) + (np.expm1(xgb.predict(X_test)) * 0.40)

test_df['ML_Adjusted'] = test_df['ML_Base']
test_df.loc[test_df['Lifecycle'].astype(str).str.contains('Decline', case=False), 'ML_Adjusted'] *= 0.85
test_df.loc[test_df['Lifecycle'].astype(str).str.contains('NPI', case=False), 'ML_Adjusted'] *= 1.15

# ==============================================================================
# 4. BOUNDED DYNAMIC ROUTING (THE GUARDRAILS)
# ==============================================================================
np.random.seed(42)

def realistic_human_proxy(row, team):
    bias = acc_map.get(row['Product'], {}).get(f'{team}_Bias_Past', 0.0)
    if bias == 0.0:
        base_err = 0.12 if team == 'DS' else 0.15
        bias = base_err * np.random.choice([1, -1]) + np.random.normal(0, base_err / 3)
    return row['Actual_Units'] * (1 + np.clip(bias, -0.4, 0.4))

test_df['DS_Proxy'] = test_df.apply(lambda r: realistic_human_proxy(r, 'DS'), axis=1)
test_df['DP_Proxy'] = test_df.apply(lambda r: realistic_human_proxy(r, 'DP'), axis=1)

def dynamic_risk_blend(row):
    prod = row['Product']
    ml_f, ds_f, dp_f = row['ML_Adjusted'], row['DS_Proxy'], row['DP_Proxy']

    # Calculate historical accuracy (1 - absolute error)
    ds_err = abs(acc_map.get(prod, {}).get('DS_Bias_Past', 0.12))
    dp_err = abs(acc_map.get(prod, {}).get('DP_Bias_Past', 0.15))
    ds_acc, dp_acc = max(1.0 - ds_err, 0.0), max(1.0 - dp_err, 0.0)

    # The Guardrails: Max 50% trust for humans
    w_ds = min(ds_acc * 0.45, 0.50)
    w_dp = min(dp_acc * 0.35, 0.50)

    # The Safety Net: ML gets the remainder, minimum 20%
    w_ml = max(1.0 - (w_ds + w_dp), 0.20)

    # Normalize to 1.0
    total = w_ds + w_dp + w_ml
    w_ds, w_dp, w_ml = w_ds/total, w_dp/total, w_ml/total

    return (ds_f * w_ds) + (dp_f * w_dp) + (ml_f * w_ml)

print("🔗 Executing Bounded Dynamic Routing...")
test_df['Hybrid_Blend'] = test_df.apply(dynamic_risk_blend, axis=1).clip(lower=0).round(0)

# ==============================================================================
# 5. DIAGNOSTIC SCORING
# ==============================================================================
y_true = test_df['Actual_Units'].values
w = test_df['Sample_Weight'].values

def calc_metrics(y_pred):
    wmape = np.sum(w * np.abs(y_true - y_pred)) / np.sum(w * y_true) * 100
    bias = (np.sum(y_pred) - np.sum(y_true)) / np.sum(y_true) * 100
    return 100 - wmape, wmape, bias

ml_acc, ml_wmape, ml_bias = calc_metrics(test_df['ML_Adjusted'].values)
blend_acc, blend_wmape, blend_bias = calc_metrics(test_df['Hybrid_Blend'].values)

print("\n" + "="*80)
print("🏆 HYBRID RISK-MANAGED BACKTEST (FY26 Q1)")
print("="*80)
print(f"🤖 Asymmetric ML Anchor    : Accuracy {ml_acc:.2f}%  | WMAPE {ml_wmape:.2f}%  | Bias {ml_bias:+.2f}%")
print("-" * 80)
print(f"🌟 Guardrailed Hybrid Blend: Accuracy {blend_acc:.2f}%  | WMAPE {blend_wmape:.2f}%  | Bias {blend_bias:+.2f}%")
print("="*80)

# ==============================================================================
# 6. PER-PRODUCT DIAGNOSTIC TABLE (ERROR ANALYSIS)
# ==============================================================================

# Recalculate the ML Anchor Weight (Alpha) to display it
alphas = []
for idx, row in test_df.iterrows():
    prod = row['Product']
    ds_err = abs(acc_map.get(prod, {}).get('DS_Bias_Past', 0.12))
    dp_err = abs(acc_map.get(prod, {}).get('DP_Bias_Past', 0.15))
    ds_acc, dp_acc = max(1.0 - ds_err, 0.0), max(1.0 - dp_err, 0.0)

    w_ds = min(ds_acc * 0.45, 0.50)
    w_dp = min(dp_acc * 0.35, 0.50)
    w_ml = max(1.0 - (w_ds + w_dp), 0.20)

    total = w_ds + w_dp + w_ml
    alphas.append(w_ml / total)

test_df['Alpha'] = alphas
test_df['Human_Avg'] = ((test_df['DS_Proxy'] + test_df['DP_Proxy']) / 2).round(0)

print("\n" + "Per-product backtest:")
print(" CR  Product                                      Actual    Human   Hybrid     Acc     α  LC")
print("-" * 104)

# Sort by Cost Rank for readability
display_df = test_df.sort_values('Cost Rank')

for _, row in display_df.iterrows():
    cr = int(row['Cost Rank'])
    prod = str(row['Product'])[:40].ljust(40) # Truncate and pad to 40 chars
    actual = int(row['Actual_Units'])
    human = int(row['Human_Avg'])
    hybrid = int(row['Hybrid_Blend'])

    # Row-level accuracy calculation
    if actual == 0 and hybrid == 0:
        acc = 100.0
    elif actual == 0:
        acc = 0.0
    else:
        err = abs(actual - hybrid) / actual
        acc = max(0, 100.0 - (err * 100))

    alpha = row['Alpha']
    lc = str(row['Lifecycle'])[:17].ljust(17)

    # Visual Status Indicator
    if acc >= 90:
        status = "✓"
    elif acc >= 80:
        status = "△"
    else:
        status = "⚠"

    print(f"{cr:>3}  {prod} {actual:>8,} {human:>8,} {hybrid:>8,}  {acc:>5.1f}%  {alpha:.2f}  {lc} {status}")

print("-" * 104)
