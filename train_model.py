import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
import math


# 1. AMS CALCULATION FUNCTION
def calculate_ams(true_labels, predictions, weights, threshold):
    
    y_pred = (predictions > threshold).astype(int)

    s = np.sum(weights[(true_labels == 1) & (y_pred == 1)])
    b = np.sum(weights[(true_labels == 0) & (y_pred == 1)])

    # AMS formula
    br = 10.0
    radicand = 2 * ((s + b + br) * math.log(1.0 + s / (b + br)) - s)
    if radicand < 0:
        return 0.0
    return math.sqrt(radicand)

# 2. DATA LOADING AND PREPARATION
print("Loading training data...")
df = pd.read_csv("data/training.csv")

# Map 's' (signal) to 1 and 'b' (background) to 0
df["Label"] = df["Label"].map({'s': 1, 'b': 0})

# Extract targets, weights, and features
y = df["Label"].values
weights = df["Weight"].values
X = df.drop(columns=['EventId', 'Weight', 'Label'])

# 3. XGBOOST CONFIGURATION & K-FOLD CV
print("Initializing Stratified 5-Fold Cross Validation...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
fold_ams_scores = []
fold_thresholds = []
oof_preds = np.zeros(len(y))   # out-of-fold predictions, for a robust global threshold
seeds = [23, 5, 2024, 777, 8888]

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'eta': 0.01,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n--- Training Fold {fold + 1}/5 ---")

    # Split data for the current fold
    X_tr, y_tr, w_tr = X.iloc[train_idx], y[train_idx], weights[train_idx]
    X_va, y_va, w_va = X.iloc[val_idx], y[val_idx], weights[val_idx]

    # Recalculate class balancing weights specific to this fold's training set
    sum_wpos = np.sum(w_tr[y_tr == 1])
    sum_wneg = np.sum(w_tr[y_tr == 0])
    params['scale_pos_weight'] = sum_wneg / sum_wpos
    params['seed'] = seeds[fold]
    
    # Rescale validation weights to match the official Test set scale
    w_va_scaled = w_va * (550000 / len(w_va))
    
    # Convert data into optimized XGBoost DMatrix structures
    dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr, missing=-999.0)
    dval = xgb.DMatrix(X_va, label=y_va, weight=w_va, missing=-999.0)

    # Train model
    model = xgb.train(
        params, 
        dtrain, 
        num_boost_round=5000,             
        evals=[(dval, 'validation')],     
        early_stopping_rounds=50,
        verbose_eval=False
    )

    # Store this fold's predictions for the pooled out-of-fold threshold search
    val_preds = model.predict(dval)
    oof_preds[val_idx] = val_preds

    # Per-fold threshold tuning (kept only as a diagnostic / reference)
    best_fold_ams = 0.0
    best_fold_threshold = 0.0
    
    for t in np.arange(0.80, 0.99, 0.01):
        ams = calculate_ams(y_va, val_preds, w_va_scaled, threshold=t)
        if ams > best_fold_ams:
            best_fold_ams = ams
            best_fold_threshold = t
            
    print(f"Fold {fold + 1} -> AMS: {best_fold_ams:.4f} | Threshold: {best_fold_threshold:.2f}")
    
    fold_ams_scores.append(best_fold_ams)
    fold_thresholds.append(best_fold_threshold)
    
    model_name = f'higgs_model_fold_{fold + 1}.json'
    model.save_model(model_name)
    print(f"[INFO] Saved: {model_name}")

# 4. FINAL RESULTS
final_ams = np.mean(fold_ams_scores)
w_all_scaled = weights * (550000 / len(weights))
best_oof_ams, oof_threshold = 0.0, 0.0
for t in np.arange(0.80, 0.99, 0.005):
    ams = calculate_ams(y, oof_preds, w_all_scaled, threshold=t)
    if ams > best_oof_ams:
        best_oof_ams = ams
        oof_threshold = t

print(f"   Mean per-fold AMS:  {final_ams:.4f}  (reference only)")
print(f"   OOF global AMS:     {best_oof_ams:.4f}")
print(f"   >> Threshold to use in generate_submission.py: {oof_threshold:.3f}")