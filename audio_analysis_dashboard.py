import streamlit as st
import pandas as pd
import plotly.express as px
import torch
import librosa
from PIL import Image
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

# Charger le modèle et le processeur pour la prédiction
def load_model_and_processor(model_path, processor_path='facebook/wav2vec2-base'):
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
    processor = Wav2Vec2Processor.from_pretrained(processor_path)
    model.eval()
    return model, processor

# Prédiction de sentiment à partir d'un fichier audio
def predict_sentiment(audio_path, model, processor, inverse_label_map, max_length=32000):
    # Charger et traiter l'audio
    speech, sr = librosa.load(audio_path, sr=16000)
    speech = speech[:max_length] if len(speech) > max_length else np.pad(speech, (0, max_length - len(speech)), 'constant')
    inputs = processor(speech, sampling_rate=16000, return_tensors='pt', padding=True)
    input_values = inputs.input_values
    # Obtenir les prédictions du modèle
    with torch.no_grad():
        outputs = model(input_values)
    logits = outputs.logits
    predicted_class = logits.argmax(dim=-1).item()
    print(f"le predicted_class est ::::::::::::::::{predicted_class}")
    sentiment = inverse_label_map[predicted_class]
    print(f"le sentiment est ::::::::::::::::{sentiment}")
    return sentiment

def predict_sentiment_load(audio, model, processor, inverse_label_map_audio, max_length=32000):
    # Charger et traiter l'audio
    speech = librosa.load(audio, sr=16000)
    speech = speech[:max_length] if len(speech) > max_length else np.pad(speech, (0, max_length - len(speech)), 'constant')
    inputs = processor(speech, sampling_rate=16000, return_tensors='pt', padding=True)
    input_values = inputs.input_values
    # Obtenir les prédictions du modèle
    with torch.no_grad():
        outputs = model(input_values)
    logits = outputs.logits
    predicted_class = logits.argmax(dim=-1).item()
    print(f"le predicted_class est ::::::::::::::::{predicted_class}")
    sentiment = inverse_label_map_audio[predicted_class]
    print(f"le sentiment est ::::::::::::::::{sentiment}")
    return sentiment

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

    # Charger le modèle et le processeur pour les prédictions
    checkpoint_dir = "./results/checkpoint-459"  # Remplacez par le chemin de votre modèle
    model, processor = load_model_and_processor(checkpoint_dir)
    inverse_label_map = {0: 'neutral', 1: 'positif', 2: 'positif', 3: 'negatif', 4: 'negatif', 5: 'negatif'}  # Remplacez par votre map
    #inverse_label_map_audio = {'neutral': 0, 'calm': 'positif', 'happy': 'positif', 'sad': "negatif", 'angry': "negatif", 'fear': "negatif"}  # Remplacez par votre map

    # Afficher un widget pour sélectionner l'analyse ou la prédiction
    option = st.sidebar.selectbox("Choisissez une option :", ["Analyse exploratoire", "Prédiction de sentiment", "Prédire sentiment sur fichier audio"])

    if option == "Analyse exploratoire":
        exploratory_analysis(df)

    elif option == "Prédiction de sentiment":
        st.subheader("🎧 Prédiction du sentiment pour un fichier audio")
        audio_id = st.sidebar.selectbox("Choisir un ID audio :", df_audio['Path'].unique())
        audio_info = df[df['Path'] == audio_id].iloc[0]

        st.write("### Informations sur l'audio sélectionné :")
        st.write(f"- **Genre :** {audio_info['Gender']}")
        st.write(f"- **Emotion réelle :** {audio_info['Emotion_Category']}")

        if st.button("Prédire le sentiment"):
            sentiment = predict_sentiment(audio_id, model, processor, inverse_label_map)
            st.write(f"### Le sentiment prédit pour cet audio est : **{sentiment}**")
            # Afficher une image de sentiment si la prédiction est effectuée
            if sentiment:
                # Charger les images locales pour chaque sentiment
                sentiment_images = {
                    "positif": Image.open("images/positif.jpg"),
                    "neutral": Image.open("images/neutre.jpg"),
                    "negatif": Image.open("images/negatif.jpg")
                }
                st.image(sentiment_images[sentiment], width=150, caption=f"Sentiment : {sentiment}")


    elif option == "Prédire sentiment sur fichier audio":
        st.subheader("🎤 Prédiction de sentiment pour un fichier audio uploadé")
        audio_file = st.file_uploader("Téléchargez un fichier audio", type=["wav", "mp3"])

        if audio_file is not None:
            with open("uploaded_audio.wav", "wb") as f:
                f.write(audio_file.getbuffer())
            st.audio(audio_file, format="audio/wav")

            if st.button("Prédire le sentiment"):
                sentiment = predict_sentiment("uploaded_audio.wav", model, processor, inverse_label_map)
                st.write(f"### Le sentiment prédit pour cet audio est : **{sentiment}**")
                # Afficher une image de sentiment si la prédiction est effectuée
                if sentiment:
                    # Charger les images locales pour chaque sentiment
                    sentiment_images = {
                        "positif": Image.open("images/positif.jpg"),
                        "neutral": Image.open("images/neutre.jpg"),
                        "negatif": Image.open("images/negatif.jpg")
                    }
                    st.image(sentiment_images[sentiment], width=150, caption=f"Sentiment : {sentiment}")


if __name__ == "__main__":
    main()