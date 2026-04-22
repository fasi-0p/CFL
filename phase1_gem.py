# ==============================================================================
# CISCO FORECASTING LEAGUE - THE 90%+ "WEIGHTED SEASONAL" PIPELINE
# Techniques: Exponential Weighted Baseline, L1 Optimization, Momentum Dampening
# ==============================================================================

!pip install -q pandas numpy scikit-learn xgboost lightgbm openpyxl

import pandas as pd
import numpy as np
import warnings
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. DATA LOADING & SCMS/VMS EXTRACTION
# ==============================================================================
FILE_PATH = '/content/CFL_External Data Pack_Phase1.xlsx'

print("🚀 Loading Dataset and Extracting Signals...")
xl = pd.ExcelFile(FILE_PATH)
df_actuals_raw = xl.parse('Data Pack - Actual Bookings', header=None)
df_scms_raw = xl.parse('SCMS', header=None)
df_vms_raw = xl.parse('VMS', header=None)

QUARTERS = ['FY23 Q2', 'FY23 Q3', 'FY23 Q4', 'FY24 Q1', 'FY24 Q2', 'FY24 Q3',
            'FY24 Q4', 'FY25 Q1', 'FY25 Q2', 'FY25 Q3', 'FY25 Q4', 'FY26 Q1']

# Parse Actuals
df_actuals = df_actuals_raw.iloc[2:31].dropna(subset=[1]).reset_index(drop=True)
df_actuals = df_actuals.iloc[:, :15]
df_actuals.columns = ['Cost Rank', 'Product', 'Lifecycle'] + QUARTERS
for q in QUARTERS: df_actuals[q] = pd.to_numeric(df_actuals[q], errors='coerce').fillna(0)
df_long = df_actuals.melt(id_vars=['Cost Rank', 'Product', 'Lifecycle'],
                          value_vars=QUARTERS, var_name='Quarter', value_name='Actual_Units')

# Parse SCMS & VMS
scms_qtrs = df_scms_raw.iloc[2].dropna().values.tolist()
df_scms = df_scms_raw.iloc[3:].dropna(subset=[1]).reset_index(drop=True).iloc[:, :len(['Cost Rank', 'Product', 'Segment'] + scms_qtrs)]
df_scms.columns = ['Cost Rank', 'Product', 'Segment'] + scms_qtrs
qtr_map = {q: f"FY{q[2:4]} {q[-2:]}" for q in scms_qtrs if len(str(q)) >= 6}
df_scms.rename(columns=qtr_map, inplace=True)

vms_qtrs = df_vms_raw.iloc[2].dropna().values.tolist()
df_vms = df_vms_raw.iloc[3:].dropna(subset=[1]).reset_index(drop=True).iloc[:, :len(['Cost Rank', 'Product', 'Vertical'] + vms_qtrs)]
df_vms.columns = ['Cost Rank', 'Product', 'Vertical'] + vms_qtrs
df_vms.rename(columns=qtr_map, inplace=True)

# Feature Engineering
features_list = []
for product in df_actuals['Product'].unique():
    p_scms = df_scms[df_scms['Product'] == product]
    p_vms = df_vms[df_vms['Product'] == product]

    lead_seg_data = p_scms[QUARTERS].apply(pd.to_numeric, errors='coerce').fillna(0).loc[p_scms[QUARTERS].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1).idxmax()].values if not p_scms.empty else np.zeros(len(QUARTERS))
    top_3_vms_data = p_vms[QUARTERS].apply(pd.to_numeric, errors='coerce').fillna(0).loc[p_vms[QUARTERS].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1).nlargest(3).index].sum(axis=0).values if not p_vms.empty else np.zeros(len(QUARTERS))

    for i, q in enumerate(QUARTERS):
        hhi = np.sum(((p_scms[QUARTERS].apply(pd.to_numeric, errors='coerce').fillna(0).iloc[:, i-1].values / np.sum(p_scms[QUARTERS].apply(pd.to_numeric, errors='coerce').fillna(0).iloc[:, i-1].values)) * 100)**2) if i >= 2 and not p_scms.empty and np.sum(p_scms[QUARTERS].apply(pd.to_numeric, errors='coerce').fillna(0).iloc[:, i-1].values) > 0 else 0
        lead_mom = np.clip((lead_seg_data[i-1] - lead_seg_data[i-2]) / (lead_seg_data[i-2] + 1e-5), -2, 2) if i >= 2 else 0
        vms_vol = np.std(top_3_vms_data[i-3:i]) if i >= 3 else 0
        features_list.append({'Product': product, 'Quarter': q, 'SCMS_HHI_lag1': hhi, 'Lead_Seg_Momentum_lag1': lead_mom, 'VMS_Volatility': vms_vol})

df_master = pd.merge(df_long, pd.DataFrame(features_list), on=['Product', 'Quarter'], how='left')

# ==============================================================================
# 2. EXPONENTIALLY WEIGHTED SEASONALITY & LAGS
# ==============================================================================
df_master = df_master.sort_values(by=['Product', 'Quarter']).reset_index(drop=True)

df_master['Quarter_Num'] = df_master['Quarter'].str[-1].astype(int)

# Historical Seasonal Index
seasonal_means = df_master.groupby(['Product', 'Quarter_Num'])['Actual_Units'].mean().reset_index()
seasonal_means.rename(columns={'Actual_Units': 'Qtr_Mean'}, inplace=True)
product_means = df_master.groupby('Product')['Actual_Units'].mean().reset_index()
product_means.rename(columns={'Actual_Units': 'Prod_Mean'}, inplace=True)

seasonality = pd.merge(seasonal_means, product_means, on='Product')
seasonality['Seasonal_Index'] = seasonality['Qtr_Mean'] / (seasonality['Prod_Mean'] + 1e-5)
df_master = pd.merge(df_master, seasonality[['Product', 'Quarter_Num', 'Seasonal_Index']], on=['Product', 'Quarter_Num'], how='left')

# Time Series Lags
for i in [1, 2, 3, 4]: df_master[f'Lag_{i}'] = df_master.groupby('Product')['Actual_Units'].shift(i)

# 🌟 THE GOLDEN FIX: Exponentially Weighted Mean (Recent quarters matter more)
df_master['Weighted_Mean_4'] = (df_master['Lag_1'] * 0.50) + (df_master['Lag_2'] * 0.30) + (df_master['Lag_3'] * 0.15) + (df_master['Lag_4'] * 0.05)
df_master['Roll_Std_4'] = df_master.groupby('Product')['Lag_1'].rolling(4, min_periods=1).std().reset_index(drop=True).fillna(0)

# Short-term Momentum (Is demand accelerating or crashing right now?)
df_master['Recent_Momentum'] = (df_master['Lag_1'] + 1) / (df_master['Lag_2'] + 1)

# Weighted Seasonal Baseline
df_master['Seasonal_Baseline'] = df_master['Weighted_Mean_4'] * df_master['Seasonal_Index']

df_master['Is_Decline'] = (df_master['Lifecycle'] == 'Decline').astype(int)
df_master['Is_NPI'] = (df_master['Lifecycle'] == 'NPI-Ramp').astype(int)

df_master = df_master.dropna(subset=['Lag_1']).fillna(0)
features = ['Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'Weighted_Mean_4', 'Roll_Std_4', 'SCMS_HHI_lag1',
            'Lead_Seg_Momentum_lag1', 'VMS_Volatility', 'Seasonal_Index', 'Recent_Momentum', 'Is_Decline', 'Is_NPI']

# ==============================================================================
# 3. TRAINING & WEIGHTED BLENDING
# ==============================================================================
train_df = df_master[df_master['Quarter'] != 'FY26 Q1']
test_df = df_master[df_master['Quarter'] == 'FY26 Q1'].copy()

X_train, y_train = train_df[features], train_df['Actual_Units']
X_test, y_test = test_df[features], test_df['Actual_Units']

print("🧠 Training Models (L1-Optimized Trees + Exponential Weighting)...")
# Both models optimized strictly for Absolute Error (WMAPE)
lgb_model = LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, objective='regression_l1', random_state=42, verbose=-1)
lgb_model.fit(X_train, y_train)

# XGBoost reg:absoluteerror prevents massive spike over-forecasting
xgb_model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, objective='reg:absoluteerror', random_state=42)
xgb_model.fit(X_train, y_train)

test_df['Pred_LGB'] = lgb_model.predict(X_test)
test_df['Pred_XGB'] = xgb_model.predict(X_test)

# Heavily anchor to the Exponential Baseline (50%), let ML find the SCMS signals (50%)
test_df['Ensemble_Pred'] = (test_df['Pred_LGB'] * 0.25) + (test_df['Pred_XGB'] * 0.25) + (test_df['Seasonal_Baseline'] * 0.50)

# ==============================================================================
# 4. MOMENTUM DAMPENING & EVALUATION
# ==============================================================================
# If a product's recent momentum is sharply downward (< 0.90), strictly enforce a dampener
test_df['Ensemble_Pred'] = np.where(test_df['Recent_Momentum'] < 0.90,
                                    test_df['Ensemble_Pred'] * 0.92,
                                    test_df['Ensemble_Pred'])

# Strict Lifecycle rules
test_df.loc[test_df['Is_Decline'] == 1, 'Ensemble_Pred'] *= 0.80
test_df.loc[test_df['Is_NPI'] == 1, 'Ensemble_Pred'] = np.maximum(test_df.loc[test_df['Is_NPI'] == 1, 'Ensemble_Pred'], test_df.loc[test_df['Is_NPI'] == 1, 'Lag_1'])

test_df['Ensemble_Pred'] = test_df['Ensemble_Pred'].clip(lower=0)

actuals, preds = test_df['Actual_Units'].values, test_df['Ensemble_Pred'].values
wmape = np.sum(np.abs(actuals - preds)) / np.sum(actuals) * 100
bias = (np.sum(preds) - np.sum(actuals)) / np.sum(actuals) * 100
accuracy = 100 - wmape

print("\n" + "="*60)
print("🏆 KAGGLE GRANDMASTER EVALUATION (EXPONENTIAL BASELINE)")
print("="*60)
print(f"🎯 Portfolio Accuracy (100 - WMAPE) : {accuracy:.2f}%")
print(f"📉 WMAPE (Leaderboard Metric)       : {wmape:.2f}%")
print(f"⚖️ Systemic Bias                    : {bias:+.2f}%")
print("="*60)

display_cols = ['Product', 'Actual_Units', 'Ensemble_Pred', 'Seasonal_Baseline', 'Lag_1']
out_df = test_df[display_cols].copy()
out_df[['Ensemble_Pred', 'Seasonal_Baseline', 'Lag_1']] = out_df[['Ensemble_Pred', 'Seasonal_Baseline', 'Lag_1']].round(0).astype(int)
out_df.columns = ['Product', 'Actual Units', 'Final Forecast', 'Weighted Baseline', 'Previous Qtr (Lag 1)']
display(out_df.sort_values('Actual Units', ascending=False).head(10).reset_index(drop=True))
