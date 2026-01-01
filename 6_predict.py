import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import RDLogger
from tqdm import tqdm

# Silence RDKit
RDLogger.DisableLog('rdApp.*')
tqdm.pandas()

def calculer_features_dict(smiles, fp_size=1024):
    """Calcule les descripteurs RDKit et Morgan pour un SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Descripteurs physiques
    desc_dict = Descriptors.CalcMolDescriptors(mol)
    
    # Morgan Fingerprints
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_size)
    for i, bit in enumerate(list(fp)):
        desc_dict[f'MorganFP_{i}'] = bit
        
    return desc_dict

def executer_predictions(input_file, output_file):
    # 1. Chargement des outils
    print("Chargement du modèle et du scaler...")
    model = joblib.load("xgboost_tm_model.pkl")
    scaler = joblib.load("scaler_tm.pkl")
    
    # Récupérer les noms de colonnes exacts utilisés lors du fit
    # On les récupère depuis le scaler ou le modèle
    try:
        model_features = scaler.feature_names_in_
    except AttributeError:
        # Si le scaler n'a pas gardé les noms, on tente via le booster
        model_features = model.get_booster().feature_names

    # 2. Chargement du fichier test
    df_test = pd.read_csv(input_file)
    
    print(f"Traitement de {len(df_test)} molécules...")
    
    # 3. Calcul des caractéristiques (RDKit + Morgan)
    features_list = df_test['SMILES'].progress_apply(calculer_features_dict).tolist()
    
    # Filtrer les molécules invalides
    X_list = [f for f in features_list if f is not None]
    valid_indices = [i for i, f in enumerate(features_list) if f is not None]
    
    # Filtrer les molécules invalides pour le calcul
    X_data = []
    valid_indices = []
    for i, f in enumerate(features_list):
        if f is not None:
            X_data.append(f)
            valid_indices.append(i)
    
    df_features = pd.DataFrame(X_data)

    # --- ÉTAPE CRUCIALE : ALIGNEMENT ---
    # On crée un DataFrame vide avec les colonnes de l'entraînement
    X = pd.DataFrame(0, index=range(len(df_features)), columns=model_features)
    
    # On remplit avec les valeurs calculées si la colonne existe dans les deux
    for col in model_features:
        if col in df_features.columns:
            X[col] = df_features[col].values
    
    # On s'assure que l'ordre est strictement identique
    X = X[model_features]
    # -----------------------------------

    print("Calcul des prédictions...")
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    
    # Assemblage
    df_test['Tm_predit_K'] = np.nan
    df_test.loc[valid_indices, 'Tm_predit_K'] = preds
    
    df_test.to_csv(output_file, index=False)
    print(f"Terminé ! Résultats sauvegardés dans : {output_file}")
    
    # Affichage des 20 premiers
    print("\n--- Affichage des 20 premières prédictions ---")
    print(df_test[['id', 'SMILES', 'Tm_predit_K']].head(20))

if __name__ == '__main__':
    # Remplacez par vos noms de fichiers réels
    FILE_A_PREDIRE = "test.csv" 
    FILE_RESULTATS = "test_predictions.csv"
    
    try:
        executer_predictions(FILE_A_PREDIRE, FILE_RESULTATS)
    except FileNotFoundError:
        print(f"Erreur : Le fichier {FILE_A_PREDIRE} n'a pas été trouvé.")
