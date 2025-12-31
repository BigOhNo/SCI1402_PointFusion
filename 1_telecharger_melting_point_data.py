# 1_telecharger_melting_point_data.py
import os
import sys

# 1. FORCER LE CHEMIN AVANT TOUT IMPORT KAGGLE
# On définit le dossier courant comme dossier de configuration Kaggle
current_dir = os.getcwd()
os.environ['KAGGLE_CONFIG_DIR'] = current_dir

# On s'assure que gdown est disponible pour le téléchargement
try:
    import gdown
except ImportError:
    print("Installation de gdown...")
    os.system(f"{sys.executable} -m pip install gdown")
    import gdown

# 2. TÉLÉCHARGEMENT DE KAGGLE.JSON DEPUIS GOOGLE DRIVE
file_id = '1m3ebmCd6ffn7wDsEKi3aYLf4aCShGBZi'
url = f'https://drive.google.com/uc?id={file_id}'
target_path = os.path.join(current_dir, 'kaggle.json')

if not os.path.exists(target_path):
    print(f"Téléchargement de kaggle.json dans : {current_dir}")
    # Utilisation de fuzzy=True au cas où le lien Google Drive change légèrement
    gdown.download(url, target_path, quiet=False, fuzzy=True)
    print(f"kaggle.json est téléchargé dans : {current_dir}")
else:
    print("kaggle.json est déjà présent dans le répertoire de travail.")

# 3. IMPORT ET AUTHENTIFICATION (SEULEMENT MAINTENANT)
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    
    api = KaggleApi()
    api.authenticate()
    print("Authentification réussie avec le fichier local !")

    # 4. TÉLÉCHARGEMENT DES DONNÉES DE LA COMPÉTITION
    print("Téléchargement de train.csv et test.csv...")
    api.competition_download_file('melting-point', 'train.csv', path='.')
    api.competition_download_file('melting-point', 'test.csv', path='.')
    print("✅ Opération terminée. Les fichiers sont téléchargés et prêts pour le nettoyage.")

except Exception as e:
    print(f"❌ Erreur persistante : {e}")
    print(f"Vérifiez manuellement la présence de 'kaggle.json' dans : {current_dir}")
