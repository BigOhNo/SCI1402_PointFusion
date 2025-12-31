# 4_optimiser_xgboost.py
import time
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem # Re-added AllChem for Morgan Fingerprints compatibility
from rdkit.Chem import Descriptors
# Removed: from rdkit.Chem.rdMolDescriptors import MorganGenerator # Not supported in current env
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from rdkit.Chem import Lipinski, Crippen, MolSurf, Descriptors3D
from rdkit.Chem import rdMolDescriptors
from rdkit import Chem
import numpy as np

# --- Désactivation des warnings ---
import warnings
from rdkit import RDLogger

# Suppress RDKit logger warnings
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

# Ignorer spécifiquement les avertissements de dépréciation pour MorganGenerator (Python warnings module)
warnings.filterwarnings("ignore", message="DEPRECATION WARNING: please use MorganGenerator")
# Ignorer les avertissements de dépréciation généraux (Python warnings module)
warnings.filterwarnings("ignore", category=DeprecationWarning)

#df_train = pd.read_csv('train_cleaned.csv')

# Constantes et paramètres
FINGERPRINT_SIZE = 1024
RANDOM_STATE = 42

# --- Fonctions utilitaires ---
def calculate_rdkit_features_dict(smiles):
    """Calcule tous les descripteurs RDKit pour un seul SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {name: np.nan for name, function in Descriptors.descList}
    try:
        return Descriptors.CalcMolDescriptors(mol)
    except Exception:
        return {name: np.nan for name, function in Descriptors.descList}

def calculate_morgan_fingerprint(smiles, size=FINGERPRINT_SIZE, radius=2):
    """Calcule le Morgan Fingerprint pour un seul SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0] * size
    try:
        # Reverted to using AllChem.GetMorganFingerprintAsBitVect for compatibility
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=size)
        return list(fingerprint) # Convert bit vector to list of ints
    except Exception:
        return [0] * size




if __name__ == '__main__':
    df_train = pd.read_csv("train_cleaned.csv")
    
    print("Calcul et concaténation des nouvelles caractéristiques...")

    # Calcul et Concaténation des Descripteurs RDKIT
    rdkit_features = df_train['SMILES'].apply(calculate_rdkit_features_dict).apply(pd.Series)
    
    # Calcul et Concaténation des Morgan FingerPrints
    morgan_features_list = df_train['SMILES'].apply(calculate_morgan_fingerprint, size=FINGERPRINT_SIZE).tolist()
    morgan_features = pd.DataFrame(morgan_features_list, index=df_train.index)
    morgan_features.columns = [f'MorganFP_{i}' for i in range(FINGERPRINT_SIZE)]
    
    # Définition de X et y
    feature_cols = [col for col in df_train.columns if col.startswith('Group')] # Inclut Group1...Group424
    X_group = df_train[feature_cols]
    
    # Concaténation finale de toutes les caractéristiques
    X = pd.concat([X_group, rdkit_features, morgan_features], axis=1)
    y = df_train['Tm']
    
    # Nettoyage des données (Imputation des NaN et suppression des colonnes constantes)
    X = X.fillna(X.mean())
    X = X.loc[:, X.nunique() > 1]
    print(f"-> Taille finale de X (incluant les nouvelles features) : {X.shape}")
    
    # Discrétisation de Tm pour stratification
    df_train['Tm_bins'] = pd.qcut(df_train['Tm'], q=10, labels=False, duplicates='drop')
    
    # Train/validation split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=df_train['Tm_bins']
    )
    
    print(f"\n✅ Split des données:")
    print(f"   - X_train shape: {X_train.shape}")
    print(f"   - X_test shape: {X_test.shape}")
    
    # Mise à l'échelle des données
    scaler = QuantileTransformer(output_distribution='normal', random_state=RANDOM_STATE)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("✅ Préparation des données terminée.")

    # Utilisation d'un jeu de paramètres performants pour un test rapide
    test_params = {
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }
    
    print("\n--- Entraînement et Test du Modèle XGBoost avec les Nouvelles Features ---")
    print(f"Paramètres utilisés: {test_params}")
    
    # Initialisation du modèle
    final_xgb_model = XGBRegressor(**test_params)
    
    # Entraînement sur le jeu d'entraînement
    start_time = time.time()
    final_xgb_model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    
    # Prédictions sur le jeu de test
    y_pred_test = final_xgb_model.predict(X_test_scaled)
    
    # Calcul des métriques finales
    r2_final = r2_score(y_test, y_pred_test)
    mae_final = mean_absolute_error(y_test, y_pred_test)
    mse_final = mean_squared_error(y_test, y_pred_test) # Calculate MSE first
    rmse_final = np.sqrt(mse_final) # Then calculate RMSE
    
    # ==============================================================================
    # 4. AFFICHAGE DES RÉSULTATS
    # ==============================================================================
    
    print("-" * 65)
    print(f"{'Temps d\'Entraînement (s)':<30} | {training_time:.2f}")
    print("-" * 65)
    print("📈 Métriques sur le Jeu de Test (X_test_scaled) :")
    print(f"R-squared (R²)                 | {r2_final:.4f}")
    print(f"Erreur Absolue Moyenne (MAE)   | {mae_final:.4f}")
    print(f"Racine de l'Erreur Quadratique (RMSE) | {rmse_final:.4f}")
    print("-" * 65)
    print("✅ Test avec les features chimiques terminé.")
    