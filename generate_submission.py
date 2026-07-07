import pandas as pd
import xgboost as xgb
import numpy as np
from features import extract_physics_features

# 1. LOAD MODEL AND TEST DATA
print("Loading pre-trained model...")
model = xgb.Booster()
model.load_model('higgs_model.json')

print("Loading test data (550,000 events)...")
df_test = pd.read_csv('data/test.csv')
df_test = extract_physics_features(df_test)

event_ids = df_test['EventId']
X_test = df_test.drop(columns=['EventId'])


# 2. INFERENCE
print("Generating predictions...")
dtest = xgb.DMatrix(X_test, missing=-999.0)
probabilities = model.predict(dtest)

# 3. RANKING & CLASSIFICATION LOGIC
print("Calculating RankOrder and generating final submission file...")

submission = pd.DataFrame({
    'EventId': event_ids,
    'Prob': probabilities
})

submission = submission.sort_values(by='Prob', ascending=True)
submission['RankOrder'] = range(1, len(submission) + 1)

PROBABILITY_THRESHOLD = 0.85
submission['Class'] = np.where(submission['Prob'] > PROBABILITY_THRESHOLD, 's', 'b')

submission = submission[['EventId', 'RankOrder', 'Class']]
submission = submission.sort_values(by='EventId')

submission.to_csv('submission_final.csv', index=False)
print("Success! 'submission_final.csv' has been generated.")