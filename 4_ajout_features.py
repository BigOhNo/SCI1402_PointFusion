# 4_ajout_features.py
import warnings
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import RDLogger

def ajouter_features_chimiques(df, smiles_column='SMILES', drop_smiles=False):
    """
    Calcule les descripteurs RDKit et supprime optionnellement la colonne SMILES.
    """
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    def _calculate_rdkit_features_dict(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {name: np.nan for name, _ in Descriptors.descList}
        try:
            return Descriptors.CalcMolDescriptors(mol)
        except Exception:
            return {name: np.nan for name, _ in Descriptors.descList}

    print("Calcul des descripteurs RDKit...")
    rdkit_df = df[smiles_column].apply(_calculate_rdkit_features_dict).apply(pd.Series)
    df_result = pd.concat([df, rdkit_df], axis=1)
    
    if drop_smiles:
        df_result = df_result.drop(columns=[smiles_column])
        
    return df_result


def ajouter_morgan_fingerprints(df, smiles_column='SMILES', size=1024, radius=2, drop_smiles=True):
    """
    Calcule les Morgan Fingerprints et supprime optionnellement la colonne SMILES.
    """
    def _calculate_morgan_fingerprint(smiles, size, radius):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [0] * size
        try:
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=size)
            return list(fingerprint)
        except Exception:
            return [0] * size

    print(f"Calcul des Morgan Fingerprints (taille={size})...")
    fps_list = df[smiles_column].apply(_calculate_morgan_fingerprint, size=size, radius=radius).tolist()
    fps_df = pd.DataFrame(fps_list, index=df.index)
    fps_df.columns = [f'MorganFP_{i}' for i in range(size)]
    
    df_result = pd.concat([df, fps_df], axis=1)
    
    if drop_smiles:
        # On vérifie si la colonne existe encore (au cas où ajouter_features_chimiques l'aurait déjà supprimée)
        if smiles_column in df_result.columns:
            df_result = df_result.drop(columns=[smiles_column])
            
    return df_result

# **************
# Point d'entrée
# **************
if __name__ == '__main__':
    
    # Chargement et enrichissement
    df_train_cleaned = pd.read_csv("train_cleaned.csv")
    
    df_train_cleaned = ajouter_features_chimiques(df_train_cleaned)
    df_train_cleaned = ajouter_morgan_fingerprints(df_train_cleaned)
    
    df_train_cleaned.to_csv("train_features.csv", index=False)
    print(f"train_features.csv a été créé avec succès! (Colonnes : {df_train_cleaned.shape[1]})")
