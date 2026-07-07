import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import math

# 1. AMS CALCULATION FUNCTION (LOCAL VALIDATION)

def calculate_ams(true_labels, predicitons, weights, threshold):
    
    y_pred = (predicitons > threshold).astype(int)

    s = np.sum(weights[(true_labels == 1) & (y_pred == 1)])
    b = np.sum(weights[(true_labels == 0) & (y_pred == 1)])

    #AMS formula

    br = 10
    radicand = 2 * ((s + b + br) * math.log(1.0 + s / (b + br)) - s)
    if radicand < 0:
        return 0.0
    return math.sqrt(radicand)

# 2. DATA LOADING AND PREPARATION

print("Loading training data...")

df = pd.read_csv("data/training.csv")

# Map 's' (signal) to 1 and 'b' (background) to 0
df["Label"] = df["Label"].map({'s':1, 'b':0})

# Extract targets, weights, and features
y = df["Label"].values
weights = df["Weight"].values
X = df.drop(columns=['EventId', 'Weight', 'Label'])

# Split into train and validation
X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(X, y, weights, test_size=0.2, random_state=23)

# Rescale validation weights to siulate the original test size
w_val_scaled = w_val * (550000 / len(w_val))

# 3. XGBOOST CONFIGURATION & TRAINING
print("Preparing DMatrix objects...")
dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train, missing=-999.0)
dval = xgb.DMatrix(X_val, missing=-999.0)

# Class balancing
sum_wpos = np.sum(w_train[y_train == 1])
sum_wneg = np.sum(w_train[y_train == 0])

params = {
    'objective': 'binary:logistic',
    'scale_pos_weight': sum_wneg/sum_wpos,
    'eta': 0.15,
    'max_depth': 6,
    'eval_metric': 'auc',
    'seed': 23
}

print("Training... (This might take a few minutes)")
num_rounds = 300
model = xgb.train(params, dtrain, num_rounds)

# Local validation
print("Predicting on validation set")
val_preds = model.predict(dval)

# Calculate AMS assuming an 85% probability threshold to mark as Signal
ams_score = calculate_ams(y_val, val_preds, w_val_scaled, threshold=0.85)
print(f"\n>>> RESULT: Estimated Local AMS Score: {ams_score:.4f} <<<")

# Save the model for inference
model.save_model('higgs_model.json')
print("Model successfully saved as 'higgs_model.json'.")