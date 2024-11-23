# README pour le Dashboard d’Analyse de Sentiments Audio

---

## Table des matières

1. [Description]
2. [Fonctionnalités]
3. [Pré-requis]
5. [Installation]
6. [Utilisation]
7. [Méthodologie]
8. [Structure du Projet]
9. [Améliorations futures]
10. [Licence]
---

## Description

Ce projet propose un dashboard d’analyse de sentiments audio développé avec Streamlit. Il utilise un modèle ResNet-18 personnalisé pour classifier des fichiers audio en catégories de 
sentiments à partir de leurs caractéristiques acoustiques. Le dashboard offre des fonctionnalités d’analyse exploratoire des données (EDA) ainsi qu’une prédiction de sentiment en temps réel.

Les fonctionnalités principales incluent :

- La visualisation des caractéristiques du dataset via des graphiques interactifs.
- L’extraction des spectrogrammes Mel et des caractéristiques de bas niveau (LLF) pour une représentation enrichie des audios.
- La combinaison de ces caractéristiques pour une classification basée sur le modèle wav2vec2-base (facebook).
- Une interface conviviale pour télécharger des fichiers audio et obtenir une prédiction de sentiment.

---

## Fonctionnalités

### Analyse exploratoire des données
- Analysez la répartition des émotions, des intensités et des genres dans le dataset.
- Examinez les corrélations entre catégories d’émotions réduites, genre et intensité émotionnelle.
### Prédiction de sentiment
- Téléchargez un fichier audio au format .wav ou .mp3.
- Le fichier est traité pour extraire les caractéristiques, générer un spectrogramme combiné et classifier le sentiment en neutre, positif ou négatif.
### Modèle de classification
- Le modèle ResNet-18 préentraîné est personnalisé pour adapter les dernières couches.
- Les couches convolutionnelles initiales sont gelées, tandis que les couches finales sont ajustées pour classer les audios en six classes émotionnelles, réduites en trois sentiments : neutre, positif et négatif.

---

## Prérequis

- **Python 3.8+** doit être installé.
- Librairies : PyTorch, Streamlit, Librosa, Plotly, Pandas, Torchvision, Matplotlib, Scikit-learn, Pillow

---

## Installation

1. Clonez ce dépôt :
   ```bash
      git clone https://github.com/votre-repo/audio-sentiment-analysis
      cd audio-sentiment-analysis

2. Créez un environnement virtuel :

    python -m venv venv
    source venv/bin/activate  # Sur Linux/MacOS
    venv\Scripts\activate     # Sur Windows

3. Installez les dépendances :

    pip install -r requirements.txt
   
4. Lancez l'application :
    streamlit run audio_analysis_dashboard.py
   
6. Assurez-vous d’avoir le dataset et le modèle préentraîné. Placez le fichier CSV (ravdess_streamlit.csv) et le checkpoint du modèle (model_and_processor/model_resnet18_V2.pth) dans les répertoires correspondants

## Utilisation
### Lancement du dashboard

1. Exécutez l’application Streamlit :
   ```bash
      streamlit run app.py

2.Naviguez dans le dashboard via la barre latérale :
- Analyse exploratoire : pour visualiser les données du dataset.
- Prédiction de sentiment : pour classifier un fichier audio téléchargé.

### Prédiction de sentiment

- Téléchargez un fichier audio au format .wav ou .mp3.
- Le système traite le fichier en :
   - Générant un spectrogramme Mel.
   - Extrayant les caractéristiques de bas niveau (LLF).
   - Combinant ces caractéristiques en une image unique.
   - Classifiant l’audio en fonction du sentiment prédominant.

## Méthodologie
### Extraction des caractéristiques
- Spectrogramme Mel
  - Librosa est utilisé pour produire des spectrogrammes qui montrent la distribution de l’énergie en fonction des fréquences et du temps.

- Caractéristiques de bas niveau (LLF)
  - Des indicateurs comme l’énergie RMS sont calculés pour capturer les variations dans le signal audio.

- Combinaison des caractéristiques
  - Les spectrogrammes Mel et les images LLF sont combinés pour créer une entrée visuelle adaptée au modèle ResNet-18.
### Modèle
- Le modèle wav2vec2-base est préentraîné par facebook. Les dernières couches sont ajustées pour classifier six classes émotionnelles réduites en trois catégories de sentiments : neutre, positif et négatif.
### Évaluation
- Le modèle est évalué à l’aide de métriques standards : précision, rappel, F1-score et exactitude.

1. Exécutez l’application Streamlit :
   ```bash
        audio-sentiment-analysis/
        │
        ├── app.py                     # Application principale Streamlit
        ├── ravdess_streamlit.csv       # Fichier de données
        ├── model_and_processor/        # Répertoire contenant le modèle préentraîné
        ├── images/                     # Bannière et images des sentiments
        ├── processed_images/           # Répertoire des images temporaires générées
        ├── requirements.txt            # Liste des dépendances
        ├── README.md                   # Documentation du projet
   

## Améliorations futures
- Étendre la classification à davantage de catégories émotionnelles.
- Ajouter des techniques avancées de traitement du signal audio.
- Supporter plusieurs langues pour les fichiers audio.
- Déployer l’application via Streamlit Cloud ou un conteneur Docker.

## Licence
Ce projet est sous licence MIT. Consultez le fichier LICENSE pour plus de détails.
