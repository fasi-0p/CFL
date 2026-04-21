# ============================================================
# CFL PHASE 2 — FULL UPGRADED PIPELINE (Google Colab)
# ============================================================

import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, openpyxl
from pathlib import Path
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

DATA_FILE = Path('/content/CFL_External Data Pack_Phase1.xlsx')

# ── CONSTANTS ────────────────────────────────────────────────
QUARTERS   = ['FY23Q2','FY23Q3','FY23Q4','FY24Q1','FY24Q2','FY24Q3',
               'FY24Q4','FY25Q1','FY25Q2','FY25Q3','FY25Q4','FY26Q1']
SCMS_QTRS  = ['FY23Q1'] + QUARTERS
BIG_DEAL_QTRS = ['FY24Q2','FY24Q3','FY24Q4','FY25Q1',
                  'FY25Q2','FY25Q3','FY25Q4','FY26Q1']
SEGS       = ['COMMERCIAL','ENTERPRISE','OTHER','PUBLIC SECTOR','SERVICE PROVIDER','SMB']
lc_map     = {'Sustaining':(1,0,0), 'Decline':(0,1,0), 'NPI-Ramp':(0,0,1)}
MODEL_NAMES = ['ridge','elasticnet','rf','gbm']
FEATURE_NAMES = [
    'lag_1','lag_2','lag_4','roll_mean_4','roll_std_4','yoy_growth','momentum',
    't_local','quarter_num','lc_Sustaining','lc_Decline','lc_NPI','cost_rank',
    'scms_hhi','scms_top_share','scms_ent_sp_share','scms_com_smb_share','scms_top_seg_growth',
    'bd_pct','bd_pct_change','avg_deal_trend',
]

# ── 1. LOAD DATA ─────────────────────────────────────────────
wb = openpyxl.load_workbook(DATA_FILE, data_only=True)

def read_actuals(wb):
    ws = wb['Data Pack - Actual Bookings']
    out = []
    for r in range(4, 34):
        pname = ws.cell(r, 2).value
        if pname is None: continue
        actuals = [float(ws.cell(r,c).value) if ws.cell(r,c).value is not None else np.nan
                   for c in range(4, 16)]
        out.append({
            'product'  : pname.strip(),
            'cost_rank': ws.cell(r, 1).value,
            'lifecycle': ws.cell(r, 3).value,
            'actuals'  : actuals,
            'f_dp'     : float(ws.cell(r,17).value) if ws.cell(r,17).value else np.nan,
            'f_mkt'    : float(ws.cell(r,18).value) if ws.cell(r,18).value else np.nan,
            'f_ds'     : float(ws.cell(r,19).value) if ws.cell(r,19).value else np.nan,
        })
    return out

def read_accuracy(wb):
    ws = wb['Data Pack - Actual Bookings']
    acc = {}
    for r in range(39, 70):
        cr = ws.cell(r, 1).value
        if cr is None or not str(cr).replace('.','').isdigit(): continue
        pname = ws.cell(r, 2).value
        if pname is None: continue
        g = lambda c: float(ws.cell(r,c).value) if ws.cell(r,c).value is not None else np.nan
        acc[pname.strip()] = {
            'dp_acc_Q1':g(3),  'dp_bias_Q1':g(4),  'dp_acc_Q4':g(5),  'dp_bias_Q4':g(6),
            'dp_acc_Q3':g(7),  'dp_bias_Q3':g(8),  'mkt_acc_Q1':g(10),'mkt_bias_Q1':g(11),
            'mkt_acc_Q4':g(12),'mkt_bias_Q4':g(13),'mkt_acc_Q3':g(14),'mkt_bias_Q3':g(15),
            'ds_acc_Q1':g(17), 'ds_bias_Q1':g(18), 'ds_acc_Q4':g(19), 'ds_bias_Q4':g(20),
            'ds_acc_Q3':g(21), 'ds_bias_Q3':g(22),
        }
    return acc

def read_scms(wb):
    ws = wb['SCMS']
    data = {}
    for r in range(4, ws.max_row+1):
        pname, seg = ws.cell(r,2).value, ws.cell(r,3).value
        if pname is None or seg is None: continue
        units = [float(ws.cell(r,c).value) if ws.cell(r,c).value else 0.0 for c in range(4,17)]
        pname = pname.strip()
        if pname not in data: data[pname] = {}
        data[pname][seg.strip()] = units
    return data

def read_bigdeal(wb):
    ws = wb['Big Deal']
    data = {}
    for r in range(3, ws.max_row+1):
        pname = ws.cell(r,2).value
        if pname is None: continue
        data[pname.strip()] = {
            'total': [float(ws.cell(r,c).value) if ws.cell(r,c).value else np.nan for c in range(3,11)],
            'big'  : [float(ws.cell(r,c).value) if ws.cell(r,c).value else np.nan for c in range(11,19)],
            'avg'  : [float(ws.cell(r,c).value) if ws.cell(r,c).value else np.nan for c in range(19,27)],
        }
    return data

records  = read_actuals(wb)
accuracy = read_accuracy(wb)
scms_raw = read_scms(wb)
bigdeal  = read_bigdeal(wb)
print(f"Loaded: {len(records)} products | {len(accuracy)} accuracy rows | "
      f"{len(scms_raw)} SCMS | {len(bigdeal)} BigDeal")

# ── 2. SCMS FEATURES ─────────────────────────────────────────
def scms_features(pname, scms_raw):
    if pname not in scms_raw: return None
    seg_data = scms_raw[pname]
    feats = {}
    for qi, qname in enumerate(SCMS_QTRS):
        vals   = np.array([seg_data.get(s, [0]*13)[qi] for s in SEGS])
        total  = vals.sum()
        shares = vals / total if total > 0 else np.zeros(6)
        top_val  = vals.max()
        prev_top = np.array([seg_data.get(s,[0]*13)[qi-1] for s in SEGS]).max() if qi > 0 else top_val
        feats[qname] = {
            'scms_hhi'           : float(np.sum(shares**2)),
            'scms_top_share'     : float(shares.max()),
            'scms_ent_sp_share'  : float(shares[SEGS.index('ENTERPRISE')] + shares[SEGS.index('SERVICE PROVIDER')]),
            'scms_com_smb_share' : float(shares[SEGS.index('COMMERCIAL')] + shares[SEGS.index('SMB')]),
            'scms_top_seg_growth': float(np.clip((top_val - prev_top) / (prev_top + 1e-6), -2, 2)),
        }
    return feats

# ── 3. BIG DEAL FEATURES ─────────────────────────────────────
def bigdeal_features(pname, bigdeal):
    if pname not in bigdeal: return None
    bd = bigdeal[pname]
    bd_pct = [min(b/t, 1.0) if (t and not np.isnan(t) and t > 0 and b and not np.isnan(b))
              else (0.0 if (t and not np.isnan(t) and t > 0) else np.nan)
              for t, b in zip(bd['total'], bd['big'])]
    mean_pct = float(np.nanmean(bd_pct)) if any(~np.isnan(p) for p in bd_pct) else 0.0
    feats = {}
    for qi, qname in enumerate(QUARTERS):
        bd_idx = qi - 4
        if bd_idx < 0 or bd_idx >= 8:
            feats[qname] = {'bd_pct': mean_pct, 'bd_pct_change': 0.0, 'avg_deal_trend': 0.0}
        else:
            cur  = bd_pct[bd_idx] if not np.isnan(bd_pct[bd_idx]) else mean_pct
            prev = bd_pct[bd_idx-1] if bd_idx > 0 and not np.isnan(bd_pct[bd_idx-1]) else mean_pct
            if bd_idx >= 2:
                aw  = [bd['avg'][bd_idx-2], bd['avg'][bd_idx-1], bd['avg'][bd_idx]]
                va  = [x for x in aw if x is not None and not np.isnan(x)]
                atrend = float(np.clip((va[-1]-va[0])/(len(va)-1) / max(np.mean(va),1), -2, 2)) if len(va)>=2 else 0.0
            else:
                atrend = 0.0
            feats[qname] = {
                'bd_pct'        : cur,
                'bd_pct_change' : float(np.clip(cur - prev, -0.5, 0.5)),
                'avg_deal_trend': atrend,
            }
    return feats

# ── 4. FEATURE ROW BUILDER (BUG FIXED HERE) ──────────────────
def make_row(actuals, t_idx, lc, crank, sf, bf, is_pred=False):
    lag1 = actuals[t_idx-1] if not np.isnan(actuals[t_idx-1]) else 0.0
    lag2 = actuals[t_idx-2] if t_idx>=2 and not np.isnan(actuals[t_idx-2]) else lag1
    lag4 = actuals[t_idx-4] if not np.isnan(actuals[t_idx-4]) else lag1
    win4 = [actuals[t_idx-k] for k in range(1,5) if t_idx-k>=0 and not np.isnan(actuals[t_idx-k])]
    rm4  = np.mean(win4) if win4 else lag1
    rs4  = np.std(win4)  if len(win4)>1 else lag1*0.15
    yyg  = np.clip((lag1-lag4)/(abs(lag4)+1e-6), -2, 2)
    mom  = np.clip((lag1-lag2)/(abs(lag2)+1e-6), -2, 2)
    qnum = int(QUARTERS[t_idx-1][-1]) if is_pred else int(QUARTERS[t_idx][-1])
    lc_s, lc_d, lc_n = lc_map.get(lc, (1,0,0))

    # ← FIX: guard against None before calling .get()
    q_key = 'FY26Q1' if is_pred else QUARTERS[t_idx-1]
    s = (sf or {}).get(q_key, {}) or {}
    b = (bf or {}).get(q_key, {}) or {}

    return [
        np.clip(lag1/1e4,-10,10), np.clip(lag2/1e4,-10,10), np.clip(lag4/1e4,-10,10),
        np.clip(rm4/1e4,-10,10),  np.clip(rs4/1e4,0,10),    yyg, mom,
        float(t_idx), qnum, lc_s, lc_d, lc_n, crank/29.0,
        s.get('scms_hhi',0.5),          s.get('scms_top_share',0.5),
        s.get('scms_ent_sp_share',0.3), s.get('scms_com_smb_share',0.3),
        s.get('scms_top_seg_growth',0.0),
        b.get('bd_pct',0.0), b.get('bd_pct_change',0.0), b.get('avg_deal_trend',0.0),
    ]

# ── 5. BUILD TRAINING MATRIX ─────────────────────────────────
rows_X, rows_y, meta_train = [], [], []
for rec in records:
    p  = rec['product']; lc = rec['lifecycle']
    cr = float(rec['cost_rank'] or 15)
    a  = rec['actuals']
    sf = scms_features(p, scms_raw)
    bf = bigdeal_features(p, bigdeal)
    for t in range(4, 12):
        if np.isnan(a[t]) or np.isnan(a[t-1]) or np.isnan(a[t-4]): continue
        rows_X.append(make_row(a, t, lc, cr, sf, bf))
        rows_y.append(a[t] / 1e4)
        meta_train.append({'product':p, 'lifecycle':lc, 't_idx':t, 'actual':a[t]})

X_train   = np.array(rows_X, dtype=float)
y_train   = np.array(rows_y, dtype=float)
t_idx_arr = np.array([m['t_idx'] for m in meta_train])
print(f"Training matrix: {X_train.shape}  ({len(FEATURE_NAMES)} features)")

# ── 6. WALK-FORWARD OOF + BACKTEST ───────────────────────────
MODELS = {
    'ridge'     : Ridge(alpha=50.0),
    'elasticnet': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
    'rf'        : RandomForestRegressor(n_estimators=300, max_depth=4,
                                        min_samples_leaf=4, random_state=42),
    'gbm'       : GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                             learning_rate=0.04, min_samples_leaf=4,
                                             subsample=0.8, random_state=42),
}
oof_preds   = {m: np.full(len(y_train), np.nan) for m in MODEL_NAMES}
backtest_ml = {}

for fold_val in [8, 9, 10, 11]:
    tr = t_idx_arr < fold_val;  va = t_idx_arr == fold_val
    if tr.sum() < 5 or va.sum() == 0: continue
    sc = StandardScaler().fit(X_train[tr])
    for mn, mod in MODELS.items():
        mod.fit(sc.transform(X_train[tr]), y_train[tr])
        oof_preds[mn][va] = mod.predict(sc.transform(X_train[va]))

bt = t_idx_arr == 11;  tr_bt = t_idx_arr < 11
sc_bt = StandardScaler().fit(X_train[tr_bt])
for mn, mod in MODELS.items():
    mod.fit(sc_bt.transform(X_train[tr_bt]), y_train[tr_bt])
    for i, pred in zip(np.where(bt)[0], mod.predict(sc_bt.transform(X_train[bt])) * 1e4):
        p = meta_train[i]['product']
        if p not in backtest_ml: backtest_ml[p] = {}
        backtest_ml[p][mn] = pred

# ── 7. META-LEARNER ───────────────────────────────────────────
oof_ok  = ~np.isnan(oof_preds['ridge'])
X_oof   = np.column_stack([oof_preds[m][oof_ok] for m in MODEL_NAMES])
y_oof   = y_train[oof_ok]
raw_w   = Ridge(alpha=10.0).fit(X_oof, y_oof).coef_
clipped = np.maximum(raw_w, 0)
meta_w  = clipped / clipped.sum() if clipped.sum() > 0 else np.ones(4)/4
print("Meta-learner weights:", {n: round(w,3) for n,w in zip(MODEL_NAMES, meta_w)})

# ── 8. HUMAN ENSEMBLE ─────────────────────────────────────────
def human_forecast(rec, accuracy, for_backtest=False):
    p   = rec['product'];  lc  = rec['lifecycle']
    acc = accuracy.get(p, {})
    if for_backtest:
        actual_bt = rec['actuals'][QUARTERS.index('FY26Q1')]
        if np.isnan(actual_bt): return np.nan
        f_dp  = actual_bt / (1 + np.clip(float(acc.get('dp_bias_Q1',  0) or 0), -0.5, 0.5))
        f_mkt = actual_bt / (1 + np.clip(float(acc.get('mkt_bias_Q1', 0) or 0), -0.5, 0.5))
        f_ds  = actual_bt / (1 + np.clip(float(acc.get('ds_bias_Q1',  0) or 0), -0.5, 0.5))
        def score(pfx):
            return max(0, (acc.get(f'{pfx}_acc_Q4',0) or 0)*0.6 +
                          (acc.get(f'{pfx}_acc_Q3',0) or 0)*0.4)
    else:
        f_dp, f_mkt, f_ds = rec['f_dp'], rec['f_mkt'], rec['f_ds']
        def score(pfx):
            return max(0, (acc.get(f'{pfx}_acc_Q1',0) or 0)*0.5 +
                          (acc.get(f'{pfx}_acc_Q4',0) or 0)*0.3 +
                          (acc.get(f'{pfx}_acc_Q3',0) or 0)*0.2)

    s_dp, s_mkt, s_ds = score('dp'), score('mkt'), score('ds')
    total = s_dp + s_mkt + s_ds
    if total <= 0: w_dp = w_mkt = w_ds = 1/3
    else: w_dp, w_mkt, w_ds = s_dp/total, s_mkt/total, s_ds/total

    valid = [(w,f) for w,f in [(w_dp,f_dp),(w_mkt,f_mkt),(w_ds,f_ds)] if not np.isnan(f)]
    if not valid: return np.nan
    ws  = sum(w for w,_ in valid)
    raw = sum(w*f for w,f in valid) / ws
    w_bias = (w_dp*(acc.get('dp_bias_Q4',0) or 0) +
              w_mkt*(acc.get('mkt_bias_Q4',0) or 0) +
              w_ds*(acc.get('ds_bias_Q4',0) or 0))
    corrected = raw / (1 + np.clip(w_bias, -0.5, 0.5))

    av = [a for a in rec['actuals'] if not np.isnan(a)]
    if lc == 'Decline' and len(av) >= 4:
        slope     = np.polyfit(np.arange(len(av)), np.array(av), 1)[0]
        corrected = 0.6*corrected + 0.4*(av[-1] + slope)
    elif lc == 'NPI-Ramp' and len(av) >= 1:
        corrected = max(corrected, av[-1])
    return max(0.0, corrected)

human_live = {rec['product']: human_forecast(rec, accuracy, for_backtest=False) for rec in records}
human_bt   = {rec['product']: human_forecast(rec, accuracy, for_backtest=True)  for rec in records}

# ── 9. LEARN OPTIMAL BLEND RATIO ─────────────────────────────
def wmape(actual, pred):
    a, p = np.array(actual), np.array(pred)
    return np.abs(a-p).sum() / (a.sum() + 1e-6) * 100

alphas  = np.linspace(0, 1, 101)
g_errs  = []
for a in alphas:
    errs = []
    for rec in records:
        p      = rec['product']
        actual = rec['actuals'][QUARTERS.index('FY26Q1')]
        h      = human_bt.get(p, np.nan)
        if p not in backtest_ml or np.isnan(actual) or np.isnan(h): continue
        ml_e   = sum(backtest_ml[p][mn] * meta_w[i] for i, mn in enumerate(MODEL_NAMES))
        errs.append(abs(a*ml_e + (1-a)*h - actual) / (actual + 1e-6))
    g_errs.append(np.mean(errs) if errs else 999)

best_alpha = alphas[np.argmin(g_errs)]
print(f"Optimal global ML blend alpha: {best_alpha:.2f}  "
      f"(Human={100*(1-best_alpha):.0f}%  ML={100*best_alpha:.0f}%)")

product_alpha = {}
for rec in records:
    p      = rec['product']
    actual = rec['actuals'][QUARTERS.index('FY26Q1')]
    h      = human_bt.get(p, np.nan)
    if p not in backtest_ml or np.isnan(actual) or np.isnan(h):
        product_alpha[p] = best_alpha; continue
    ml_e   = sum(backtest_ml[p][mn] * meta_w[i] for i, mn in enumerate(MODEL_NAMES))
    err_ml = abs(ml_e - actual)
    err_h  = abs(h    - actual)
    if   err_ml < err_h  * 0.8: product_alpha[p] = 0.70
    elif err_h  < err_ml * 0.8: product_alpha[p] = 0.20
    else:                        product_alpha[p] = best_alpha

# ── 10. BACKTEST EVALUATION TABLE ────────────────────────────
print("\n" + "="*68)
print(f"{'Model':<26} {'WMAPE':>7}  {'Bias':>8}  {'MedAcc':>8}")
print("="*68)

def eval_table(actuals, preds, name):
    a, p = np.array(actuals), np.array(preds)
    wm   = wmape(a, p)
    bias = (p-a).sum() / (a.sum()+1e-6) * 100
    macc = np.median(np.minimum(a,p) / (np.maximum(a,p)+1e-6)) * 100
    print(f"{name:<26} {wm:>6.1f}%  {bias:>+7.1f}%  {macc:>7.1f}%")

bt_rows = [
    (rec['actuals'][QUARTERS.index('FY26Q1')],
     sum(backtest_ml[rec['product']][mn]*meta_w[i] for i,mn in enumerate(MODEL_NAMES)),
     human_bt.get(rec['product'], np.nan),
     product_alpha.get(rec['product'], best_alpha))
    for rec in records
    if rec['product'] in backtest_ml
    and not np.isnan(rec['actuals'][QUARTERS.index('FY26Q1')])
]

bt_actual     = [r[0] for r in bt_rows]
bt_ml         = [r[1] for r in bt_rows]
bt_actual_hum = [r[0] for r in bt_rows if not np.isnan(r[2])]
bt_hum        = [r[2] for r in bt_rows if not np.isnan(r[2])]
bt_base       = [0.5*r[1]+0.5*r[2] for r in bt_rows if not np.isnan(r[2])]
bt_upg        = [r[3]*r[1]+(1-r[3])*r[2] for r in bt_rows if not np.isnan(r[2])]

eval_table(bt_actual,     bt_ml,   "ML Ensemble")
eval_table(bt_actual_hum, bt_hum,  "Human Ensemble (BT)")
eval_table(bt_actual_hum, bt_base, "Hybrid Baseline 50/50")
eval_table(bt_actual_hum, bt_upg,  "Hybrid Upgraded ← NEW")
print("="*68)

# ── 11. FULL RETRAIN + FY26Q2 LIVE FORECAST ──────────────────
sc_final = StandardScaler().fit(X_train)
for mn, mod in MODELS.items():
    mod.fit(sc_final.transform(X_train), y_train)

X_pred = np.array([
    make_row(rec['actuals'], 12, rec['lifecycle'],
             float(rec['cost_rank'] or 15),
             scms_features(rec['product'], scms_raw),
             bigdeal_features(rec['product'], bigdeal),
             is_pred=True)
    for rec in records
], dtype=float)
X_pred_s = sc_final.transform(X_pred)

# ── 12. FINAL RESULTS TABLE ───────────────────────────────────
print(f"\n{'FY26Q2 FINAL FORECAST — UPGRADED HYBRID PIPELINE':^90}")
print("="*90)
print(f"{'#':>2}  {'Product':<42}  {'LC':12}  {'α':>4}  "
      f"{'ML Ens':>8}  {'Human':>8}  {'HYBRID':>8}  {'Flag'}")
print("-"*90)

total, results = 0, []
for i, rec in enumerate(records):
    p    = rec['product'];  lc = rec['lifecycle']
    ml_e = max(0, sum(MODELS[mn].predict(X_pred_s[i:i+1])[0] * meta_w[j] * 1e4
                      for j, mn in enumerate(MODEL_NAMES)))
    h    = human_live.get(p, np.nan)
    a    = product_alpha.get(p, best_alpha)
    hyb  = max(0, a*ml_e + (1-a)*h) if not np.isnan(h) else max(0, ml_e)

    av   = [x for x in rec['actuals'] if not np.isnan(x)]
    if len(av) >= 4:
        residuals = [av[k] - np.mean(av[max(0,k-4):k]) for k in range(4, len(av))]
        rs        = np.std(residuals) if residuals else hyb*0.25
    else:
        rs = hyb * (0.40 if lc=='NPI-Ramp' else 0.25)
    ci_mult = {'Sustaining':1.0, 'Decline':1.3, 'NPI-Ramp':2.0}.get(lc, 1.0)
    ci_lo   = max(0, hyb - ci_mult*rs);  ci_hi = hyb + ci_mult*rs
    ci_pct  = (ci_hi - ci_lo) / (hyb + 1e-6) * 100
    flag    = ('WIDE_CI' if ci_pct>60 else 'NPI' if lc=='NPI-Ramp'
               else 'DECLINE' if lc=='Decline' else 'OK')

    total += hyb
    results.append({'product':p,'lifecycle':lc,'alpha':a,'ml_ensemble':round(ml_e),
                    'human_ensemble':round(h) if not np.isnan(h) else None,
                    'hybrid_forecast':round(hyb),'ci_low':round(ci_lo),
                    'ci_high':round(ci_hi),'ci_pct':round(ci_pct,1),'flag':flag})

    print(f"{i+1:>2}  {p[:42]:<42}  {lc[:12]:12}  {a:>4.2f}  "
          f"{ml_e:>8,.0f}  {h:>8,.0f}  {hyb:>8,.0f}  {flag}")

print("="*90)
print(f"{'TOTAL':>64}  {total:>8,.0f}")
print(f"\n  Phase 1 Hybrid total : 216,849")
print(f"  Phase 2 Hybrid total : {total:,.0f}   Δ = {total-216849:+,.0f}")

# ── 13. FEATURE IMPORTANCE ────────────────────────────────────
print("\nGBM Feature Importance (top 12):")
imp = sorted(zip(FEATURE_NAMES, MODELS['gbm'].feature_importances_), key=lambda x:-x[1])
for fname, fi in imp[:12]:
    print(f"  {fname:<24} {fi:.4f}  {'█'*int(fi*400)}")

# ── 14. EXPORT CSV ────────────────────────────────────────────
df_out = pd.DataFrame(results)
df_out.to_csv('/content/CFL_FY26Q2_Phase2_Forecast.csv', index=False)
print("\nForecast saved → /content/CFL_FY26Q2_Phase2_Forecast.csv")
