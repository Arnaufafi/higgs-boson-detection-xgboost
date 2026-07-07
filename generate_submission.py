import pandas as pd
import xgboost as xgb
import numpy as np
from features import extract_physics_features

# 1. LOAD TEST DATA
print("Loading test data (550,000 events)...")
df_test = pd.read_csv('data/test.csv')

# Apply feature engineering
df_test = extract_physics_features(df_test)

event_ids = df_test['EventId']
X_test = df_test.drop(columns=['EventId'])

# 2. ENSEMBLE INFERENCE (SEED BAGGING)
print("Generating predictions using 5-Model Ensemble...")
dtest = xgb.DMatrix(X_test, missing=-999.0)

# Array vacío de ceros para ir sumando las probabilidades
ensemble_probabilities = np.zeros(len(df_test))
NUM_MODELS = 5

# Cargamos y predecimos con los 5 modelos uno por uno
for i in range(1, NUM_MODELS + 1):
    model_name = f'higgs_model_fold_{i}.json'
    print(f" -> Inferring with {model_name}...")
    
    model = xgb.Booster()
    model.load_model(model_name)
    
    # Sumamos las predicciones al total
    ensemble_probabilities += model.predict(dtest)

# Calculamos la media (promedio de las 5 predicciones)
final_probabilities = ensemble_probabilities / NUM_MODELS

# 3. RANKING & CLASSIFICATION LOGIC
print("Calculating RankOrder and generating final submission file...")

submission = pd.DataFrame({
    'EventId': event_ids,
    'Prob': final_probabilities
})

submission = submission.sort_values(by='Prob', ascending=True)
submission['RankOrder'] = range(1, len(submission) + 1)

# Pon aquí el "Robust Mean Threshold" que te dio el script de entrenamiento
PROBABILITY_THRESHOLD = 0.942
submission['Class'] = np.where(submission['Prob'] > PROBABILITY_THRESHOLD, 's', 'b')

submission = submission[['EventId', 'RankOrder', 'Class']]
submission = submission.sort_values(by='EventId')

submission.to_csv('submission_ensembled.csv', index=False)
print("Success! 'submission_ensembled.csv' has been generated.")