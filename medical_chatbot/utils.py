import base64
import requests
import io
from PIL import Image
from dotenv import load_dotenv
import os
import logging
import streamlit as st

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chargement des variables d'environnement
load_dotenv()

# URL et clé API de Groq
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the .env file")

@st.cache_data
def process_image_query(image_content, query):
    """
    Traite une image et une requête, puis envoie à l'API GROQ pour analyse avec le modèle Llama 4 Scout
    
    Args:
        image_content (bytes): Contenu de l'image en bytes
        query (str): Question de l'utilisateur sur l'image
        
    Returns:
        dict: Réponse du modèle d'IA
    """
    try:
        # Encoder l'image en base64
        encoded_image = base64.b64encode(image_content).decode("utf-8")
        
        # Vérifier que l'image est valide
        try:
            img = Image.open(io.BytesIO(image_content))
            img.verify()
        except Exception as e:
            logger.error(f"Invalid image format: {str(e)}")
            return {"error": f"Invalid image format: {str(e)}"}
        
        # Préparer les messages pour l'API avec un prompt médical spécifique
        system_prompt = "You are an expert medical assistant. Analyze the medical image and provide detailed, professional medical insights about what you observe. Include possible diagnoses, noteworthy features, and relevant medical context. Be thorough but clear in your explanation."
        
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                ]
            }
        ]
        
        # Utiliser le modèle Llama 4 Scout pour remplacer les modèles décommissionnés
        response = requests.post(
            GROQ_API_URL, 
            json={
                "model": "meta-llama/llama-4-scout-17b-16e-instruct", 
                "messages": messages, 
                "max_tokens": 2000
            },
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            timeout = 60  # Augmenter le timeout pour les analyses d'images complexes
        )
            
        # Traiter la réponse
        if response.status_code == 200:
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            logger.info(f"Processed response from model: {answer[:100]}...")
            return {"scout": answer}
        else:
            logger.error(f"Error from API: {response.status_code} - {response.text}")
            return {"error": f"Error from API: {response.status_code} - {response.text}"}

    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        return {"error": f"An unexpected error occurred: {str(e)}"}
