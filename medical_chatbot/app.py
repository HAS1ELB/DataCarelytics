import streamlit as st
import base64
from PIL import Image
import io
import os
from medical_chatbot.utils import process_image_query

def main():
    # Configuration de la page Streamlit
    '''st.set_page_config(
        page_title="AI-DOCTOR - Medical Chatbot",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Chargement des styles personnalisés
    with open("medical_chatbot/assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)'''

    # Titre de l'application
    st.markdown("""
        <div class="header">
            <h1>🩺 AI-DOCTOR (MEDICAL CHATBOT) ANALYZE IMAGE APPLICATION</h1>
        </div>
        """, unsafe_allow_html=True)

    # Interface utilisateur en deux colonnes
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📤 Upload Image")
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Afficher l'image téléchargée
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        st.markdown("### 💬 Ask Question")
        query = st.text_area("Enter your question about the image", height=150)
        submit_button = st.button("🚀 Submit Query", type="primary")

    # Traitement de la requête
    if submit_button:
        if uploaded_file is not None and query:
            with st.spinner("Processing your request..."):
                try:
                    # Lire le contenu de l'image
                    image_bytes = uploaded_file.getvalue()
                    
                    # Traiter l'image et la requête
                    responses = process_image_query(image_bytes, query)
                    
                    # Afficher les résultats
                    st.markdown("### Results")
                    
                    if "error" in responses:
                        st.error(responses["error"])
                    else:
                        # Afficher la réponse du modèle
                        st.markdown(responses["scout"])
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            if not uploaded_file:
                st.error("Please upload an image")
            if not query:
                st.error("Please enter a question")
                
if __name__ == "__main__":
       main()
