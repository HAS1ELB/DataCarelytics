import streamlit as st
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
from io import BytesIO
import datetime
from sklearn.cluster import KMeans

def medical_image_analysis():
    st.title("📸 Analyse d'Images Médicales")
    st.markdown("**Importez et analysez des images médicales avec des outils de vision par ordinateur.**", unsafe_allow_html=True)

    if 'image' not in st.session_state:
        st.session_state.image = None
        st.session_state.processed_images = {}

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
                st.error(f"Erreur lors du chargement de l'image : {e}")

    if st.session_state.image is not None:
        image = st.session_state.image.copy()

        st.markdown("### 🖼️ Image Originale")
        st.image(image, caption=f"Image originale : {st.session_state.image_name}", use_container_width=True)

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
                st.image(processed_image, caption="Image après prétraitement", use_container_width=True)

        with st.expander("🔍 Analyse Avancée", expanded=False):
            analysis_type = st.selectbox(
                "Type d'analyse",
                ["Contours et caractéristiques", "Histogramme", "Segmentation avancée"]
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

                st.image(edges, caption=f"Détection de contours ({edge_method})", use_container_width=True)

                if st.checkbox("Afficher les contours sur l'image"):
                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contour_img = np.zeros_like(edges)
                    cv2.drawContours(contour_img, contours, -1, 255, 1)
                    st.image(contour_img, caption="Contours détectés", use_container_width=True)
                    st.write(f"Nombre de contours détectés : {len(contours)}")

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

            elif analysis_type == "Segmentation avancée":
                if len(processed_image.shape) == 3:
                    gray_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2GRAY)
                else:
                    gray_image = processed_image

                segmentation_method = st.selectbox(
                    "Méthode de segmentation",
                    ["Watershed", "K-Means"]
                )

                if segmentation_method == "Watershed":
                    blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
                    threshold_value = st.slider("Valeur de seuil", 0, 255, 127)
                    _, thresh = cv2.threshold(blur, threshold_value, 255, cv2.THRESH_BINARY_INV)
                    kernel = np.ones((3, 3), np.uint8)
                    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
                    sure_bg = cv2.dilate(opening, kernel, iterations=3)
                    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
                    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
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
                    st.image(watershed_result, caption="Résultat Watershed (contours)", use_container_width=True)

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

                    st.image(segmented_img, caption=f"Segmentation K-Means ({n_clusters} clusters)", use_container_width=True)

        with st.container():
            st.markdown("### 💾 Exporter les Résultats")
            export_format = st.selectbox("Format d'exportation", ["PNG", "JPEG"])
            if st.button("Exporter l'image"):
                img_pil = Image.fromarray(processed_image)
                buf = BytesIO()
                img_pil.save(buf, format=export_format)
                buf.seek(0)
                st.download_button(
                    label="Télécharger l'image",
                    data=buf,
                    file_name=f"processed_{st.session_state.image_name}.{export_format.lower()}",
                    mime=f"image/{export_format.lower()}"
                )