import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download_melting_point_data():
    # Initialisation de l'API
    api = KaggleApi()
    api.authenticate()

    # Définition de la compétition
    competition = "melting-point"
    
    # Liste des fichiers à télécharger
    files = ['train.csv', 'test.csv']
    
    print("Début du téléchargement...")
    for file in files:
        # Télécharge dans le répertoire courant du projet Eclipse
        api.competition_download_file(competition, file, path='.')
        print(f"{file} téléchargé avec succès.")

if __name__ == "__main__":
    try:
        download_melting_point_data()
    except Exception as e:
        print(f"Erreur lors du téléchargement : {e}")
