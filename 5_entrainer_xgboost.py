# 5_entrainer_xgboost.py
import pandas as pd
import numpy as np
import time
import joblib # Pour sauvegarder le modèle et le scaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

def entrainer(df_features):
    print(f"Démarrage de l'entraînement sur {len(df_features)} molécules...")

    # 1. Sélection rigoureuse des colonnes numériques uniquement
    # On élimine SMILES et toute autre colonne textuelle résiduelle
    X = df_features.select_dtypes(include=[np.number])
    
    # Ensuite on retire la cible 'Tm' et les colonnes d'index
    cols_to_drop = ['Tm', 'Unnamed: 0']
    X = X.drop(columns=[c for c in cols_to_drop if c in X.columns])
    
    y = df_features['Tm']
    
    # Vérification de sécurité
    print(f"Nombre de features initiales (numériques) : {X.shape[1]}")

    # Gestion des NaN/Infinis
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # 2. Suppression des features constantes
    selector = VarianceThreshold(threshold=(.99 * (1 - .99)))
    X_high_variance = selector.fit_transform(X)
    selected_cols = X.columns[selector.get_support()]
    X = pd.DataFrame(X_high_variance, columns=selected_cols, index=df_features.index)
    print(f"Features conservées : {len(selected_cols)} / {X_high_variance.shape[1]}")

    # 3. Split Stratifié (pour garder la distribution de température)
    tm_bins = pd.qcut(y, q=10, labels=False, duplicates='drop')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=tm_bins
    )

    # 4. Scaling
    scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Entraînement XGBoost
    model = XGBRegressor(
        n_estimators=500, # Augmenté car XGBoost est rapide
        learning_rate=0.053, 
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        tree_method='hist' # Accélère l'entraînement
    )
    
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    print(f"Temps d'entraînement : {time.time() - start_time:.2f} s")
    
    # 6. Évaluation
    y_pred = model.predict(X_test_scaled)
    print(f"\n--- Résultats ---")
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f} °K")
    
    # 7. Importance des features
    importances = pd.Series(model.feature_importances_, index=selected_cols)
    print("\nTop 10 des features les plus influentes :")
    print(importances.sort_values(ascending=False).head(10))
    
    return model, scaler

if __name__ == '__main__':
    df = pd.read_csv("train_features.csv")
    
    model, scaler = entrainer(df)
    
    # Sauvegarde des objets pour la prédiction future
    joblib.dump(model, "xgboost_tm_model.pkl")
    joblib.dump(scaler, "scaler_tm.pkl")
    print("\nModèle et Scaler sauvegardés !")