import streamlit as st
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
from io import BytesIO
import datetime

def medical_image_analysis():
    st.title("📸 Analyse d'Images Médicales")
    st.markdown("**Importez et analysez des images médicales avec des outils de vision par ordinateur**", unsafe_allow_html=True)
    
    # Session state initialization
    if 'image' not in st.session_state:
        st.session_state.image = None
        st.session_state.processed_images = {}
    
    # Image loading container
    with st.container():
        st.markdown("### 📂 Chargement d'Image")
        uploaded_file = st.file_uploader("Choisissez une image médicale", type=["png", "jpg", "jpeg", "dicom", "dcm"])
        
        col1, col2 = st.columns(2)
        with col1:
            use_example = st.checkbox("Utiliser une image d'exemple")
        
        if use_example:
            example_type = st.selectbox(
                "Type d'image d'exemple",
                ["Radiographie pulmonaire", "IRM cérébrale", "Échographie", "Tomographie", "Mammographie"]
            )
            st.info("Utilisation de l'image d'exemple: " + example_type)
            st.session_state.image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
            st.session_state.image_name = f"exemple_{example_type.lower().replace(' ', '_')}"
            st.session_state.image_type = example_type
            
        elif uploaded_file is not None:
            try:
                if uploaded_file.name.lower().endswith(('.dcm', '.dicom')):
                    dicom_data = pydicom.dcmread(uploaded_file)
                    st.session_state.image = apply_modality_lut(dicom_data.pixel_array, dicom_data)
                    st.session_state.image_type = "DICOM"
                    st.session_state.image_name = uploaded_file.name
                    
                    with st.expander("Métadonnées DICOM"):
                        metadata = {
                            "Patient ID": getattr(dicom_data, "PatientID", "Non disponible"),
                            "Patient Name": getattr(dicom_data, "PatientName", "Non disponible"),
                            "Modality": getattr(dicom_data, "Modality", "Non disponible"),
                            "Study Date": getattr(dicom_data, "StudyDate", "Non disponible"),
                            "Image Size": f"{dicom_data.pixel_array.shape[0]} x {dicom_data.pixel_array.shape[1]}"
                        }
                        for key, value in metadata.items():
                            st.write(f"**{key}:** {value}")
                else:
                    image = Image.open(uploaded_file)
                    st.session_state.image = np.array(image)
                    st.session_state.image_name = uploaded_file.name
                    st.session_state.image_type = "Standard"
                
                st.success("Image chargée avec succès !")
            except Exception as e:
                st.error(f"Erreur lors du chargement de l'image: {e}")
    
    # Check if an image is loaded
    if st.session_state.image is not None:
        image = st.session_state.image.copy()
        
        st.markdown("### 🖼️ Image Originale")
        st.image(image, caption=f"Image originale: {st.session_state.image_name}", use_column_width=True)
        
        # Image preprocessing container
        with st.container():
            st.markdown("### 🛠️ Prétraitement de l'Image")
            preprocessing_options = st.multiselect(
                "Sélectionnez les prétraitements à appliquer",
                ["Normalisation", "Amélioration du contraste", "Réduction du bruit", "Segmentation", "Redimensionnement"]
            )
            
            processed_image = image.copy()
            
            if preprocessing_options:
                if "Normalisation" in preprocessing_options:
                    if processed_image.max() > 0:
                        processed_image = cv2.normalize(processed_image, None, 0, 255, cv2.NORM_MINMAX)
                    st.session_state.processed_images["Normalisation"] = processed_image.copy()
                
                if "Amélioration du contraste" in preprocessing_options:
                    if len(processed_image.shape) == 3:
                        gray_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2GRAY)
                    else:
                        gray_image = processed_image
                    processed_image = cv2.equalizeHist(gray_image)
                    st.session_state.processed_images["Amélioration du contraste"] = processed_image.copy()
                
                if "Réduction du bruit" in preprocessing_options:
                    kernel_size = st.slider("Taille du noyau (impair)", 3, 15, 5, 2)
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    processed_image = cv2.medianBlur(processed_image, kernel_size)
                    st.session_state.processed_images["Réduction du bruit"] = processed_image.copy()
                
                if "Segmentation" in preprocessing_options:
                    threshold_type = st.selectbox(
                        "Type de seuillage",
                        ["Binaire", "Otsu", "Adaptatif"]
                    )
                    if len(processed_image.shape) == 3:
                        gray_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2GRAY)
                    else:
                        gray_image = processed_image
                    
                    if threshold_type == "Binaire":
                        threshold_value = st.slider("Valeur de seuil", 0, 255, 127)
                        _, processed_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
                    elif threshold_type == "Otsu":
                        _, processed_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    else:
                        block_size = st.slider("Taille du bloc", 3, 99, 11, 2)
                        if block_size % 2 == 0:
                            block_size += 1
                        c_value = st.slider("Valeur C", -10, 10, 2)
                        processed_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                              cv2.THRESH_BINARY, block_size, c_value)
                    st.session_state.processed_images["Segmentation"] = processed_image.copy()
                
                if "Redimensionnement" in preprocessing_options:
                    scale_percent = st.slider("Pourcentage de redimensionnement", 10, 200, 100)
                    width = int(processed_image.shape[1] * scale_percent / 100)
                    height = int(processed_image.shape[0] * scale_percent / 100)
                    processed_image = cv2.resize(processed_image, (width, height), interpolation=cv2.INTER_AREA)
                    st.session_state.processed_images["Redimensionnement"] = processed_image.copy()
                
                st.markdown("### 🔄 Image Prétraitée")
                st.image(processed_image, caption="Image après prétraitement", use_column_width=True)
        
        # Advanced analysis container
        with st.expander("🔍 Analyse Avancée", expanded=False):
            analysis_type = st.selectbox(
                "Type d'analyse",
                ["Contours et caractéristiques", "Histogramme", "Classification (simulée)", "Segmentation avancée"]
            )
            
            if analysis_type == "Contours et caractéristiques":
                if len(processed_image.shape) == 3:
                    gray_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2GRAY)
                else:
                    gray_image = processed_image
                
                edge_method = st.selectbox("Méthode de détection", ["Canny", "Sobel", "Laplacien"])
                
                if edge_method == "Canny":
                    low_threshold = st.slider("Seuil bas", 0, 255, 50)
                    high_threshold = st.slider("Seuil haut", 0, 255, 150)
                    edges = cv2.Canny(gray_image, low_threshold, high_threshold)
                elif edge_method == "Sobel":
                    ksize = st.slider("Taille du noyau Sobel", 1, 7, 3, 2)
                    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=ksize)
                    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=ksize)
                    edges = cv2.magnitude(sobelx, sobely)
                    edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                else:
                    ksize = st.slider("Taille du noyau Laplacien", 1, 7, 3, 2)
                    edges = cv2.Laplacian(gray_image, cv2.CV_64F, ksize=ksize)
                    edges = np.uint8(np.absolute(edges))
                    edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                
                st.image(edges, caption=f"Détection de contours ({edge_method})", use_column_width=True)
                
                if st.checkbox("Afficher les contours sur l'image"):
                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contour_img = np.zeros_like(edges)
                    cv2.drawContours(contour_img, contours, -1, 255, 1)
                    st.image(contour_img, caption="Contours détectés", use_column_width=True)
                    
                    st.write(f"Nombre de contours détectés: {len(contours)}")
                    
                    if len(contours) > 0 and st.checkbox("Analyser les contours principaux"):
                        contours = sorted(contours, key=cv2.contourArea, reverse=True)
                        num_contours = min(5, len(contours))
                        contour_data = []
                        
                        for i in range(num_contours):
                            contour = contours[i]
                            area = cv2.contourArea(contour)
                            perimeter = cv2.arcLength(contour, True)
                            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                            contour_data.append({
                                "Contour": i+1,
                                "Aire (pixels²)": int(area),
                                "Périmètre (pixels)": int(perimeter),
                                "Circularité": round(circularity, 3)
                            })
                        
                        st.write("**Analyse des principaux contours:**")
                        st.table(contour_data)
            
            elif analysis_type == "Histogramme":
                fig, ax = plt.subplots(figsize=(10, 4))
                if len(processed_image.shape) == 3:
                    color = ('b', 'g', 'r')
                    for i, col in enumerate(color):
                        hist = cv2.calcHist([processed_image], [i], None, [256], [0, 256])
                        ax.plot(hist, color=col)
                    ax.set_title('Histogramme (BGR)')
                else:
                    hist = cv2.calcHist([processed_image], [0], None, [256], [0, 256])
                    ax.plot(hist)
                    ax.set_title('Histogramme (Niveaux de gris)')
                
                ax.set_xlabel('Intensité des pixels')
                ax.set_ylabel('Nombre de pixels')
                ax.grid(True)
                st.pyplot(fig)
                
                st.write("**Statistiques d'intensité de pixels:**")
                if len(processed_image.shape) == 3:
                    for i, channel in enumerate(['Bleu', 'Vert', 'Rouge']):
                        stats = {
                            "Canal": channel,
                            "Min": int(processed_image[:,:,i].min()),
                            "Max": int(processed_image[:,:,i].max()),
                            "Moyenne": round(processed_image[:,:,i].mean(), 2),
                            "Écart-type": round(processed_image[:,:,i].std(), 2)
                        }
                        st.write(f"**{channel}:** Min={stats['Min']}, Max={stats['Max']}, Moyenne={stats['Moyenne']}, Écart-type={stats['Écart-type']}")
                else:
                    stats = {
                        "Min": int(processed_image.min()),
                        "Max": int(processed_image.max()),
                        "Moyenne": round(processed_image.mean(), 2),
                        "Écart-type": round(processed_image.std(), 2)
                    }
                    st.write(f"Min={stats['Min']}, Max={stats['Max']}, Moyenne={stats['Moyenne']}, Écart-type={stats['Écart-type']}")
            
            elif analysis_type == "Classification (simulée)":
                st.warning("Cette fonctionnalité simule la classification d'images médicales.")
                import time
                import random
                
                if st.button("Lancer la classification"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    steps = ["Prétraitement", "Extraction des caractéristiques", "Classification par CNN", "Évaluation des résultats"]
                    
                    for i, step in enumerate(steps):
                        progress_value = (i / len(steps))
                        progress_bar.progress(progress_value)
                        status_text.text(f"Étape {i+1}/{len(steps)}: {step}")
                        time.sleep(1)
                    
                    progress_bar.progress(1.0)
                    status_text.text("Classification terminée!")
                    
                    if st.session_state.image_type == "DICOM":
                        classes = ["Normal", "Pneumonie", "COVID-19", "Nodule pulmonaire", "Autre pathologie"]
                    else:
                        classes = ["Normal", "Anomalie de type 1", "Anomalie de type 2", "Indéterminé"]
                    
                    probabilities = np.random.dirichlet(np.ones(len(classes)), size=1)[0]
                    
                    st.subheader("Résultats de la classification")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    bars = ax.bar(classes, probabilities, color='skyblue')
                    max_idx = np.argmax(probabilities)
                    bars[max_idx].set_color('navy')
                    ax.set_ylabel('Probabilité')
                    ax.set_title('Probabilités de classification')
                    plt.ylim(0, 1)
                    plt.xticks(rotation=45, ha='right')
                    for i, prob in enumerate(probabilities):
                        ax.text(i, prob + 0.02, f'{prob:.2f}', ha='center', va='bottom')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.markdown(f"**Conclusion:** L'image est classée comme **{classes[max_idx]}** avec une probabilité de {probabilities[max_idx]:.2f} (simulée).")
                    st.info("Note: Ces résultats sont simulés.")
            
            elif analysis_type == "Segmentation avancée":
                if len(processed_image.shape) == 3:
                    gray_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2GRAY)
                else:
                    gray_image = processed_image
                
                segmentation_method = st.selectbox(
                    "Méthode de segmentation",
                    ["Watershed", "K-Means", "Seuillage multi-niveaux"]
                )
                
                if segmentation_method == "Watershed":
                    blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
                    threshold_value = st.slider("Valeur de seuil", 0, 255, 127)
                    _, thresh = cv2.threshold(blur, threshold_value, 255, cv2.THRESH_BINARY_INV)
                    kernel = np.ones((3, 3), np.uint8)
                    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
                    sure_bg = cv2.dilate(opening, kernel, iterations=3)
                    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
                    rel_dist_thresh = st.slider("Seuil relatif pour premiers plans", 0.1, 0.9, 0.7)
                    _, sure_fg = cv2.threshold(dist_transform, rel_dist_thresh * dist_transform.max(), 255, 0)
                    sure_fg = np.uint8(sure_fg)
                    unknown = cv2.subtract(sure_bg, sure_fg)
                    _, markers = cv2.connectedComponents(sure_fg)
                    markers = markers + 1
                    markers[unknown == 255] = 0
                    
                    if len(processed_image.shape) == 3:
                        markers = cv2.watershed(processed_image, markers.copy())
                    else:
                        colored_img = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
                        markers = cv2.watershed(colored_img, markers.copy())
                    
                    watershed_result = np.zeros_like(gray_image)
                    watershed_result[markers == -1] = 255
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(thresh, caption="Image seuillée", use_column_width=True)
                    with col2:
                        st.image(watershed_result, caption="Résultat Watershed (contours)", use_column_width=True)
                
                elif segmentation_method == "K-Means":
                    if len(processed_image.shape) == 3:
                        reshaped_img = processed_image.reshape((-1, 3))
                    else:
                        reshaped_img = processed_image.reshape((-1, 1))
                    
                    reshaped_img = np.float32(reshaped_img)
                    n_clusters = st.slider("Nombre de clusters", 2, 10, 4)
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    kmeans.fit(reshaped_img)
                    labels = kmeans.labels_
                    centers = np.uint8(kmeans.cluster_centers_)
                    segmented_img = centers[labels]
                    
                    if len(processed_image.shape) == 3:
                        segmented_img = segmented_img.reshape(processed_image.shape)
                    else:
                        segmented_img = segmented_img.reshape(processed_image.shape)
                    
                    st.image(segmented_img, caption=f"Segmentation K-Means ({n_clusters} clusters)", use_column_width=True)
                    st.write("**Centres des clusters:**")
                    for i, center in enumerate(centers):
                        if len(center) == 1:
                            st.write(f"Cluster {i+1}: Intensité = {center[0]}")
                        else:
                            st.write(f"Cluster {i+1}: BGR = {center}")
                
                else:
                    n_levels = st.slider("Nombre de niveaux", 2, 8, 4)
                    thresholds = np.linspace(0, 255, n_levels+1)[1:-1].astype(np.uint8)
                    segmented_img = np.zeros_like(gray_image)
                    
                    for i in range(len(thresholds) + 1):
                        if i == 0:
                            mask = gray_image < thresholds[i]
                            value = 0
                        elif i == len(thresholds):
                            mask = gray_image >= thresholds[i-1]
                            value = 255
                        else:
                            mask = (gray_image >= thresholds[i-1]) & (gray_image < thresholds[i])
                            value = int(255 * (i / len(thresholds)))
                        segmented_img[mask] = value
                    
                    st.image(segmented_img, caption=f"Seuillage multi-niveaux ({n_levels} niveaux)", use_column_width=True)
                    st.write("**Valeurs de seuil utilisées:**", thresholds)
        
        # Export container
        with st.container():
            st.markdown("### 💾 Exporter les Résultats")
            export_format = st.selectbox("Format d'exportation", ["PNG", "JPEG", "TIFF"])
            
            if st.button("Préparer l'exportation"):
                img_pil = Image.fromarray(processed_image)
                buf = BytesIO()
                
                if export_format == "PNG":
                    img_pil.save(buf, format="PNG")
                    mime_type = "image/png"
                    file_ext = "png"
                elif export_format == "JPEG":
                    img_pil.save(buf, format="JPEG", quality=95)
                    mime_type = "image/jpeg"
                    file_ext = "jpg"
                else:
                    img_pil.save(buf, format="TIFF")
                    mime_type = "image/tiff"
                    file_ext = "tiff"
                
                buf.seek(0)
                now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"image_med_{now}.{file_ext}"
                
                st.download_button(
                    label="Télécharger l'image traitée",
                    data=buf,
                    file_name=filename,
                    mime=mime_type
                )
                
                if st.checkbox("Inclure un rapport textuel"):
                    report = f"""
                    # Rapport d'Analyse d'Image Médicale

                    **Date:** {datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}
                    **Nom de l'image:** {st.session_state.image_name}
                    **Type d'image:** {st.session_state.image_type}
                    **Dimensions:** {processed_image.shape[0]} x {processed_image.shape[1]} pixels
                    
                    ## Prétraitements appliqués
                    {', '.join(preprocessing_options) if preprocessing_options else "Aucun"}
                    
                    ## Statistiques d'image
                    - **Min:** {processed_image.min()}
                    - **Max:** {processed_image.max()}
                    - **Moyenne:** {processed_image.mean():.2f}
                    - **Écart-type:** {processed_image.std():.2f}
                    
                    ## Notes supplémentaires
                    Analyste: _________________
                    Observations: _________________
                    
                    *Ce rapport a été généré automatiquement.*
                    """
                    st.download_button(
                        label="Télécharger le rapport",
                        data=report,
                        file_name=f"rapport_{now}.md",
                        mime="text/markdown"
                    )