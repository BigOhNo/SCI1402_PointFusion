# 3_comparaison_modeles_apprentissage.py
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error

# On s'assure que les bibliothèques spécifiques sont installées
try:
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
except ImportError:
    print("Installation des moteurs de boosting...")
    os.system(f"{sys.executable} -m pip install xgboost lightgbm scikit-learn matplotlib seaborn")
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor

# *********************************************
# 5. Construction du modèle / Choix de méthodes
# *********************************************

def comparer_modeles(df):
    """Entraîne plusieurs modèles sur les colonnes GroupX et compare leurs performances."""
        
    # 1. Définition de X (features) et y (cible)
    # On utilise uniquement les colonnes GroupX pour ce premier test
    feature_cols = [col for col in df.columns if col.startswith('Group')]
    # Ajouter explicitement les descripteurs chimiques s'ils existent dans le df
    #chimiques = ['MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 'TPSA']
    #for col in chimiques:
    #    if col in df.columns:
    #        feature_cols.append(col)            
    X = df[feature_cols]
    y = df['Tm']

    # 2. Stratification pour assurer une distribution équilibrée de Tm
    # On crée 10 bacs (bins) basés sur les quantiles de Tm
    tm_bins = pd.qcut(df['Tm'], q=10, labels=False, duplicates='drop')

    # 3. Division Train/Test (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=tm_bins
    )

    # 4. Mise à l'échelle (Scaling) avec QuantileTransformer 
    # (Très efficace pour les données chimiques souvent non-linéaires)
    scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    # On transforme puis on remet dans un DataFrame avec les noms de colonnes
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)

    # 5. Liste des modèles à tester
    models = [
        ("Linear Regression", LinearRegression()),
        ("Ridge", Ridge(alpha=1.0)),
        ("k-NN", KNeighborsRegressor(n_neighbors=5)),
        ("SVR", SVR(C=100, gamma='scale')),
        ("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        ("XGBoost", XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        ("LightGBM", LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1))
    ]

    # 6. Boucle d'entraînement
    results = []
    for name, model in models:
        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time
        
        y_pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        results.append({"Modèle": name, "R2": r2, "MAE": mae, "Temps (s)": training_time})
        #print(f"✅ {name:<20} terminé (R²: {r2:.4f})")

    # 7. Création du DataFrame de résultats et Visualisation
    df_results = pd.DataFrame(results).sort_values(by="R2", ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x="R2", y="Modèle", data=df_results, palette="viridis", hue="Modèle", legend=False)
    plt.title("Comparaison des Performances (Score R²)")
    plt.axvline(x=0.8, color='r', linestyle='--', label='Excellent')
    plt.tight_layout()
    plt.show()

    return df_results

# ********************************
# 5.1 Ajout de features chimiques
# ********************************
from rdkit import Chem
from rdkit.Chem import Descriptors

def ajouter_features_chimiques(df):
    """Calcule MolWt, LogP, etc. pour booster le score R2."""
    print(f"Calcul des descripteurs pour {len(df)} molécules...")
    
    # Initialisation des colonnes
    df['MolWt'] = 0.0
    df['LogP'] = 0.0
    df['NumHDonors'] = 0
    df['NumHAcceptors'] = 0
    df['TPSA'] = 0.0
    
    # On utilise une liste pour plus de rapidité
    data = []
    for smiles in df['SMILES_canonique']:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            data.append([
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.TPSA(mol)
            ])
        else:
            data.append([0, 0, 0, 0, 0])
            
    df[['MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 'TPSA']] = data
    return df

# **************
# Point d'entrée
# **************
if __name__ == '__main__':
    
    df_train_cleaned = pd.read_csv("train_cleaned.csv")
    #df_test_cleaned = pd.read_csv("test_cleaned.csv")
    
    df_comparaison = comparer_modeles(df_train_cleaned)
    print("\n=== Classement final des modèles (uniquement avec GroupX) ===")
    print(df_comparaison)
    
    #print("\n=== Classement final des modèles (avec ajout de features chimiques) ===")
    #df_train_cleaned_features = ajouter_features_chimiques(df_train_cleaned)
    #df_test_cleaned_features = ajouter_features_chimiques(df_test_cleaned)
    #df_comparaison = comparer_modeles(df_train_cleaned_features)
    #print(df_comparaison)