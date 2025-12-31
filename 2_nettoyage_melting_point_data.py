# 2_nettoyage_melting_point_data.py
import pandas as pd

# *******************************************
# 3.1 Normaliser les colonnes Group 1...Group 424
# *******************************************
def normalize_group_columns(df):
    """Supprime les espaces dans les noms des colonnes commençant par 'Group '."""
    # Identification des colonnes à renommer (ex: 'Group 1' -> 'Group1') 
    group_cols = {col: col.replace(' ', '') for col in df.columns if col.startswith('Group ')}
    df.rename(columns=group_cols, inplace=True)
    return df

# *************************************
# 3.2 Traitement des valeurs manquantes
# *************************************
def nettoyer_valeurs_manquantes(df, est_test=False):
    """Supprime les lignes sans ID ou SMILES (et Tm pour le train). Remplit tous les autres NaN
    (GroupX) par 0."""
    # Colonnes indispensables (Tm seulement si ce n'est pas le test)
    cols_critiques = ['id', 'SMILES']
    if not est_test:
        cols_critiques.append('Tm')
    
    # 1. Suppression des lignes avec des valeurs manquantes critiques
    # Note: On ne supprime normalement pas de lignes dans le test.csv pour garder les IDs,
    # mais si l'ID ou le SMILES manque, la prédiction est impossible.
    df.dropna(subset=cols_critiques, inplace=True)
    
    # 2. Remplissage de tous les autres NaN (Group 1...424) par 0
    # C'est plus simple que de cibler les colonnes une par une.
    if df.isnull().any().any():
        df.fillna(0, inplace=True)
        
    return df

# *************************************************************
# 3.3 Normaliser canonique des SMILES et supprimer les doublons
# *************************************************************
# Vérification et installation automatique de RDKit
try:
    from rdkit import Chem
except ImportError:
    print("Installation de rdkit...")
    os.system(f"{sys.executable} -m pip install rdkit")
    from rdkit import Chem

def canoniser_smiles(smiles):
    """Convertit un SMILES en sa forme canonique. Retourne None si invalide."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # canonical=True assure une représentation textuelle unique
            return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None
    return None

def appliquer_normalisation_canonique(df):
    """Applique la canonisation et supprime les doublons/invalides."""
    print(f"Taille initiale : {len(df)}")
    
    # 1. Création de la colonne SMILES_canonique
    df['SMILES_canonique'] = df['SMILES'].apply(canoniser_smiles)
    
    # 2. Suppression des SMILES invalides (ceux que RDKit n'a pas pu lire)
    invalides = df['SMILES_canonique'].isnull().sum()
    if invalides > 0:
        print(f"Suppression de {invalides} SMILES invalides.")
        df.dropna(subset=['SMILES_canonique'], inplace=True)
    
    # 3. Suppression des doublons basés sur le SMILES canonique
    # On garde la première occurrence
    doublons = df.duplicated(subset=['SMILES_canonique']).sum()
    if doublons > 0:
        print(f"Suppression de {doublons} doublons moléculaires.")
        df.drop_duplicates(subset=['SMILES_canonique'], keep='first', inplace=True)
    
    return df

# **************
# Point d'entrée
# **************
if __name__ == '__main__':
    
    # 1. Chargement des fichiers téléchargés
    print("\n=== Chargement des fichiers téléchargés ===")
    try:
        # Tentative de lecture des fichiers
        df_train = pd.read_csv('train.csv')
        df_test = pd.read_csv('test.csv')
        
        print("✅ Fichiers chargés avec succès !")
        print(f"Dimensions de Train : {df_train.shape}")
        print(f"Dimensions de Test : {df_test.shape}")
    
    except FileNotFoundError as e:
        print(f"❌ Erreur : Le fichier n'a pas été trouvé. Vérifiez le chemin.\nDétails : {e}")    
    except pd.errors.EmptyDataError:
        print("❌ Erreur : L'un des fichiers est vide.")    
    except pd.errors.ParserError:
        print("❌ Erreur : Problème lors du parsing (le fichier n'est probablement pas un CSV valide).")    
    except Exception as e:
        print(f"❌ Une erreur inattendue est survenue : {e}")
    
    # 3.1 Normaliser les colonnes Group 1...Group 424
    print("\n=== Normaliser les colonnes Group 1...Group 424 ===")
    df_train = normalize_group_columns(df_train)
    df_test = normalize_group_columns(df_test)
    print(f"Colonnes renommées dans Train. Exemple : {df_train.columns[3]}")
    print(f"Colonnes renommées dans Test.  Exemple : {df_test.columns[2]}")
    
    # 3.2 Traitement les valeurs manquantes
    print("\n=== Traitement les valeurs manquantes ===")
    df_train = nettoyer_valeurs_manquantes(df_train, est_test=False)
    df_test = nettoyer_valeurs_manquantes(df_test, est_test=True)
    print(f"Nettoyage terminé. Train: {df_train.shape}, Test: {df_test.shape}")
    
    # 3.3 Normaliser canonique des SMILES et supprimer les doublons
    print("\n=== Normaliser canonique des SMILES et supprimer les doublons ===")
    df_train = appliquer_normalisation_canonique(df_train)
    # Note : Pour le test.csv, on canonise mais on ne supprime JAMAIS de lignes 
    # pour ne pas fausser le fichier de soumission.
    df_test['SMILES_canonique'] = df_test['SMILES'].apply(canoniser_smiles)
    print(f"Taille finale après canonisation (Train) : {len(df_train)}")
    print(f"Taille finale après canonisation (Test) : {len(df_test)}")
    
    # Sauvegarder df_train et df_test à train_cleaned.csv et test_cleaned.csv
    df_train.to_csv("train_cleaned.csv", index=False, encoding='utf-8')
    print("\ndf_train a été sauvegardé en tant que : train_cleaned.csv.")
    df_test.to_csv("test_cleaned.csv", index=False, encoding='utf-8')
    print("df_test a été sauvegardé en tant que : test_cleaned.csv.")
    
    
    

    
    
    