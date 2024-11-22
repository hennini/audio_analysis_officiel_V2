import streamlit as st
import pandas as pd
import plotly.express as px
import torch
import librosa
from PIL import Image
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
# Import des fonctions fournies
from sklearn.preprocessing import StandardScaler
import librosa.display
import os
import torchvision

class CustomResNet18(nn.Module):
    def __init__(self, num_classes=6):
        super(CustomResNet18, self).__init__()

        # Charger ResNet18
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Remplacer la couche fully connected
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

        # Fine-tuning des dernières couches
        for param in self.resnet.parameters():
            param.requires_grad = False

        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.resnet(x)
    

# Fonction pour charger un modèle sauvegardé
def load_model(checkpoint_path, num_classes=6):
    # Initialiser le modèle
    model = CustomResNet18(num_classes=num_classes)

    # Charger le checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Si les poids sont quantifiés, les déquantifier
    for key, value in checkpoint.items():
        if isinstance(value, torch.Tensor) and value.is_quantized:
            checkpoint[key] = value.dequantize()

    # Charger les poids en ignorant les clés manquantes/inattendues
    model.load_state_dict(checkpoint, strict=False)

    return model


# Définir les paramètres nécessaires
sample_rate = 16000
n_mels = 128
hop_length = 512
fmax = 8000
output_dir = "processed_images"  # Répertoire pour les images générées

# Définir les transformations pour ResNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensionner
    transforms.ToTensor(),           # Convertir en tenseur
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_mel_spectrogram(file_path, sample_rate, output_file, n_mels, hop_length, fmax, label):
    try:
        audio, sr = librosa.load(file_path, sr=sample_rate)
        mel_spect = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length, fmax=fmax)
        mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)

        # Générer un nom de fichier cohérent avec le label
        output_file = os.path.join(output_dir, f"{label}.png")
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_spect_db, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='mel')
        plt.axis('off') 
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
        plt.close()
        return output_file
    except Exception as e: 
        print(f"Error processing file {file_path}: {e}")
        return None

def extract_llf_features(audio_data, sr, n_fft, win_length, hop_length, label):
    try:
        # Assurez-vous que toutes les valeurs nécessaires sont calculées correctement
        rms = librosa.feature.rms(y=audio_data, frame_length=win_length, hop_length=hop_length, center=False)
        
        # Si vous empilez des caractéristiques LLF, assurez-vous qu'aucun élément ne contient "..."
        feats = np.vstack([rms])  # Ajoutez toutes les caractéristiques calculées ici
        feats = librosa.power_to_db(feats)
        
        # Normalisation
        scaler = StandardScaler()
        feats = scaler.fit_transform(feats.T).T

        # Sauvegarder l'image
        output_file = os.path.join(output_dir, f"{label}_llf.png")
        plt.figure(figsize=(10, 4))
        plt.imshow(feats, aspect='auto', origin='lower', cmap='viridis')
        plt.axis('off')
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
        plt.close()

        return output_file
    except Exception as e:
        print(f"Error extracting LLF features: {e}")
        return None


def combine_images_if_same_filename(mel_spectrogram_path, llf_path, output_dir):
    try:
        # Vérifiez si les fichiers existent
        if mel_spectrogram_path is None or llf_path is None:
            print("One or both files do not exist. Skipping combination.")
            return None

        mel_filename = os.path.basename(mel_spectrogram_path).split('_')[0]
        llf_filename = os.path.basename(llf_path).split('_')[0]

        if mel_filename == llf_filename:
            print(f"Combining {mel_filename} and {llf_filename}")

            # Charger les images
            mel_img = Image.open(mel_spectrogram_path)
            llf_img = Image.open(llf_path)

            # Ajustement des dimensions et combinaison
            new_width = max(mel_img.width, llf_img.width)
            new_height = max(mel_img.height, llf_img.height)
            mel_img = mel_img.resize((new_width, new_height))
            llf_img = llf_img.resize((new_width, new_height))

            # Combinaison
            combined_img = np.vstack((np.array(mel_img), np.array(llf_img)))
            combined_img = Image.fromarray(combined_img)

            # Sauvegarde
            output_file = os.path.join(output_dir, f"{mel_filename}_combined.png")
            combined_img.save(output_file)

            return output_file
        else:
            print(f"Filenames do not match: {mel_filename} and {llf_filename}. Skipping.")
            return None
    except Exception as e:
        print(f"Error combining images: {e}")
        return None



# Fonction pour préparer l'entrée du modèle
def prepare_audio_for_resnet(audio_path, output_dir, model):
    try:

        label = os.path.basename(audio_path).split('.')[0]

        # Étape 1 : Générer le spectrogramme Mel
        mel_file = extract_mel_spectrogram(audio_path, sample_rate, None, n_mels, hop_length, fmax, label)

        
        # Étape 2 : Extraire les caractéristiques LLF
        audio_data, sr = librosa.load(audio_path, sr=sample_rate)
        n_fft = 2048
        win_length = n_fft
        llf_file = extract_llf_features(audio_data, sr, n_fft, win_length, hop_length, label)
        
        # Étape 3 : Combiner les images Mel spectrogram et LLF
        combined_file  = combine_images_if_same_filename(mel_file, llf_file, output_dir)
        if not combined_file :
            raise ValueError("Impossible de combiner les images.")
        
        # Étape 4 : Charger et transformer l'image combinée
        combined_image = Image.open(combined_file ).convert("RGB")
        transformed_image = transform(combined_image).unsqueeze(0)  # Ajouter une dimension batch
        
        # Étape 5 : Faire une prédiction avec le modèle ResNet
        model.eval()
        with torch.no_grad():
            outputs = model(transformed_image)
            predicted_class = outputs.argmax(dim=1).item()

        class_map_bis = {0: 'neutral', 1: 'positif', 2: 'positif', 3: 'negatif', 4: 'negatif', 5: 'negatif'}
        return class_map_bis[predicted_class]
    
    except Exception as e:
        print(f"Erreur dans le traitement de l'audio : {e}")
        return None


# Fonction d'analyse exploratoire
def exploratory_analysis(df):
    st.subheader("🔍 Analyse exploratoire des données")
    st.write("Statistiques descriptives des données :")
    st.write(df.describe(include='all'))

    emotion_counts = df['Emotion'].value_counts()
    fig_emotion = px.pie(values=emotion_counts, names=emotion_counts.index, title="Répartition des émotions", color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig_emotion, use_container_width=True)

    intensity_counts = df['Emotion intensity'].value_counts()
    fig_intensity = px.bar(x=intensity_counts.index, y=intensity_counts.values, title="Intensité des émotions", labels={'x': 'Intensité', 'y': 'Nombre'}, color=intensity_counts.index, color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_intensity, use_container_width=True)

    gender_counts = df['Gender'].value_counts()
    fig_gender = px.bar(x=gender_counts.index, y=gender_counts.values, title="Répartition des genres", labels={'x': 'Genre', 'y': 'Nombre'}, color=gender_counts.index, color_discrete_sequence=px.colors.qualitative.Prism)
    st.plotly_chart(fig_gender, use_container_width=True)

    fig_emotion_intensity = px.histogram(df, x='Emotion', color='Emotion intensity', barmode='group', title="Répartition des émotions par intensité", color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig_emotion_intensity, use_container_width=True)

    fig_emotion_gender = px.histogram(df, x='Emotion', color='Gender', barmode='group', title="Répartition des émotions par genre", color_discrete_sequence=px.colors.qualitative.Dark2)
    st.plotly_chart(fig_emotion_gender, use_container_width=True)

    reduced_emotion_counts = df['Emotion_Category'].value_counts().sort_index()
    fig_reduced_emotion = px.pie(values=reduced_emotion_counts, names=reduced_emotion_counts.index, title="Répartition des émotions réduites", color_discrete_sequence=px.colors.qualitative.Vivid)
    st.plotly_chart(fig_reduced_emotion, use_container_width=True)

    fig_reduced_emotion_gender = px.histogram(df, x='Emotion_Category', color='Gender', barmode='group', title="Répartition des émotions réduites par genre", color_discrete_sequence=px.colors.qualitative.Safe)
    st.plotly_chart(fig_reduced_emotion_gender, use_container_width=True)

    st.write("Exemples de données :")
    st.dataframe(df.sample(5))

# Fonction principale pour afficher le dashboard
def main():
    st.set_page_config(page_title="Dashboard d'Analyse de Sentiment Audio", layout="wide")
    left_co, cent_co, last_co = st.columns(3)
    with cent_co:
        st.image("images/banner.jpg", width=400)

    st.sidebar.title("Options du Dashboard")
    st.sidebar.write("Utilisez cette barre pour naviguer dans les options.")
    
    # Charger les données fictives
    df = pd.read_csv('ravdess_streamlit.csv')
    df_audio = df.head(43)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Charger le modèle
    checkpoint_path = "model_and_processor/model_resnet18_V2.pth"
    model = load_model(checkpoint_path)

    # Mettre le modèle en mode évaluation
    model.eval()
   
    # Dictionnaire pour mapper les classes à des sentiments
    class_map = {0: 'neutral', 1: 'positif', 2: 'positif', 3: 'negatif', 4: 'negatif', 5: 'positif'}
    
    # Afficher un widget pour sélectionner l'analyse ou la prédiction
    option = st.sidebar.selectbox("Choisissez une option :", ["Analyse exploratoire", "Prédire sentiment sur fichier audio"])

    if option == "Analyse exploratoire":
        exploratory_analysis(df)

    elif option == "Prédire sentiment sur fichier audio":
        st.subheader("🎤 Prédiction de sentiment pour un fichier audio uploadé")
        audio_file = st.file_uploader("Téléchargez un fichier audio", type=["wav", "mp3"])

        if audio_file is not None:
            with open("uploaded_audio.wav", "wb") as f:
                f.write(audio_file.getbuffer())
            st.audio(audio_file, format="audio/wav")

            if st.button("Prédire le sentiment"):
                sentiment = prepare_audio_for_resnet("uploaded_audio.wav", output_dir, model)
                st.write(f"### Le sentiment prédit pour cet audio est : **{sentiment}**")
                if sentiment:
                    sentiment_images = {
                        "positif": Image.open("images/positif.jpg"),
                        "neutral": Image.open("images/neutre.jpg"),
                        "negatif": Image.open("images/negatif.jpg")
                    }
                    st.image(sentiment_images[sentiment], width=150, caption=f"Sentiment : {sentiment}")

if __name__ == "__main__":
    main()