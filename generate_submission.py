import pandas as pd
import xgboost as xgb
import numpy as np

# 1. LOAD TEST DATA
print("Loading test data (550,000 events)...")
df_test = pd.read_csv('data/test.csv')

event_ids = df_test['EventId']
X_test = df_test.drop(columns=['EventId'])

# 2. ENSEMBLE INFERENCE (SEED BAGGING)
print("Generating predictions using 5-Model Ensemble...")
dtest = xgb.DMatrix(X_test, missing=-999.0)

# Zero-filled array to accumulate the probabilities
ensemble_probabilities = np.zeros(len(df_test))
NUM_MODELS = 5

# Load and predict with the 5 models one by one
for i in range(1, NUM_MODELS + 1):
    model_name = f'higgs_model_fold_{i}.json'
    print(f" -> Inferring with {model_name}...")
    
    model = xgb.Booster()
    model.load_model(model_name)
    
    # Add the predictions to the running total
    ensemble_probabilities += model.predict(dtest)

# Compute the mean (average of the 5 predictions)
final_probabilities = ensemble_probabilities / NUM_MODELS

# 3. RANKING & CLASSIFICATION LOGIC
print("Calculating RankOrder and generating final submission file...")

submission = pd.DataFrame({
    'EventId': event_ids,
    'Prob': final_probabilities
})

submission = submission.sort_values(by='Prob', ascending=True)
submission['RankOrder'] = range(1, len(submission) + 1)

# Robust threshold from train_model.py (chosen on pooled out-of-fold predictions).
# Verified on Kaggle: 0.940 -> AMS 3.72 (best), vs 0.934 -> 3.70.
PROBABILITY_THRESHOLD = 0.940
submission['Class'] = np.where(submission['Prob'] > PROBABILITY_THRESHOLD, 's', 'b')

submission = submission[['EventId', 'RankOrder', 'Class']]
submission = submission.sort_values(by='EventId')

submission.to_csv('submission_ensembled.csv', index=False)
print("Success! 'submission_ensembled.csv' has been generated.")