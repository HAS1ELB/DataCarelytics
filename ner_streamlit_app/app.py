import joblib
import tempfile
import streamlit as st
from io import BytesIO
from fpdf import FPDF
from ner_streamlit_app.utils import predict_entities, extract_features
from PIL import Image
import io
from datetime import datetime

@st.cache_resource
def load_model(path):
    return joblib.load(path)

def structure_entities(pairs):
    structured = {}
    current_entity = None
    for token, label in pairs:
        if label == "O":
            continue
        label_clean = label.split("-")[-1]
        if label.startswith("B-"):
            current_entity = label_clean
            if current_entity not in structured:
                structured[current_entity] = [token]
            else:
                structured[current_entity].append(token)
        elif label.startswith("I-") and current_entity:
            structured[current_entity].append(token)
    return structured

import tempfile

def generate_pdf(name, surname, photo, structured_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Titre
    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(30, 60, 90)
    pdf.cell(0, 10, "Dossier Médical Structuré", ln=True, align="C")
    pdf.ln(10)

    # Photo patient (optionnelle)
    if photo is not None:
        # Sauvegarde temporaire de l'image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmpfile:
            img = Image.open(photo)
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            img.save(tmpfile.name, format='JPEG')
            tmpfile.flush()

            # Ajout dans PDF via chemin temporaire
            pdf.image(tmpfile.name, x=150, y=20, w=40, h=40)

    pdf.ln(10)

    # Infos patient + date
    pdf.set_font("Arial", "", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(40, 10, f"Nom : {surname}", ln=0)
    pdf.cell(60, 10, f"Prénom : {name}", ln=0)
    from datetime import datetime
    pdf.cell(0, 10, f"Date : {datetime.today().strftime('%d/%m/%Y')}", ln=1)
    pdf.ln(10)

    # Entités médicales
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(0, 102, 204)
    pdf.cell(0, 10, "Entités Médicales Identifiées :", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "", 12)
    for ent, words in structured_data.items():
        pdf.set_text_color(80, 80, 80)
        pdf.cell(0, 8, f"{ent}:", ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(0, 8, ' '.join(words))
        pdf.ln(3)

    # Génération du buffer PDF
    pdf_output = pdf.output(dest='S').encode('latin1')
    buffer = BytesIO(pdf_output)
    buffer.seek(0)
    return buffer


# --- Interface Streamlit ---
def main():
    st.title("🧠 NER Médical Structuré (CRF)")

    # Champs Nom, Prénom, Photo
    name = st.text_input("Prénom du patient")
    surname = st.text_input("Nom du patient")
    photo = st.file_uploader("📸 Photo du patient (optionnel)", type=["jpg", "jpeg", "png"])

    uploaded_file = st.file_uploader("📄 Téléverser un fichier texte (.txt)", type=["txt"])

    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")
        st.text_area("Texte extrait :", text, height=150)

        if st.button("Analyser le texte"):
            if not name or not surname:
                st.warning("Merci de renseigner le nom et le prénom du patient.")
            else:
                model = load_model("ner_streamlit_app/model/crf_ner_model.joblib")
                entities = predict_entities(text, model)
                structured = structure_entities(entities)

                st.subheader("Dossier Structuré :")
                for ent, tokens in structured.items():
                    st.markdown(f"**{ent}** → {' '.join(tokens)}")

                pdf_data = generate_pdf(name, surname, photo, structured)
                st.download_button(
                    "📄 Télécharger le dossier PDF",
                    data=pdf_data,
                    file_name="dossier_medical.pdf",
                    mime="application/pdf"
                )


if __name__ == "__main__":
    main()