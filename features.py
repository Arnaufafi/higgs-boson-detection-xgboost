import numpy as np

def extract_physics_features(df):
    
    df_feat = df.copy()
    
    # 1. Isolate only the events where BOTH the Tau and the Lepton exist
    valid_idx = (df_feat['PRI_tau_eta'] != -999.0) & (df_feat['PRI_lep_eta'] != -999.0)
    
    # 2. Calculate Delta Eta
    d_eta = df_feat.loc[valid_idx, 'PRI_tau_eta'] - df_feat.loc[valid_idx, 'PRI_lep_eta']
    
    # 3. Calculate Delta Phi
    d_phi = np.abs(df_feat.loc[valid_idx, 'PRI_tau_phi'] - df_feat.loc[valid_idx, 'PRI_lep_phi'])
    # If the distance is greater than PI, take the shorter path across the other side of the circle
    d_phi = np.where(d_phi > np.pi, 2 * np.pi - d_phi, d_phi)
    
    # 4. Calculate Delta R and inject it into the DataFrame
    df_feat['Delta_R_tau_lep'] = -999.0
    df_feat.loc[valid_idx, 'Delta_R_tau_lep'] = np.sqrt(d_eta**2 + d_phi**2)
    
    return df_feat 
