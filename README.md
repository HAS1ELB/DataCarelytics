https://has1elb-medicalanalysisapp-main-xojuse.streamlit.app/

# DataCarelytics

## Overview

DataCarelytics is a comprehensive medical data science platform designed to analyze and process medical datasets, including tabular data, medical images, and clinical text. Leveraging advanced machine learning, deep learning, and natural language processing (NLP) techniques, it provides tools for clinical prediction, image classification, and text mining. The platform is tailored for researchers, clinicians, and data scientists, emphasizing usability, security, and advanced analytics for sensitive medical data.

### Key Features

* **Tabular Data Analysis** : Advanced preprocessing, feature engineering, and machine learning with algorithms like XGBoost, LightGBM, and CatBoost.
* **Medical Image Processing** : Analyze MRI and X-ray images using pre-trained deep learning models for tumor classification and disease detection.
* **Medical Text Analysis** : Extract structured information from clinical notes using Named Entity Recognition (NER) with CRF models.
* **Chatbot Functionality** :
* **Image-based Chatbot** : Analyzes medical images using Groq’s Llama 4 Scout model.
* **Text-based RAG Chatbot** : Answers medical queries using a retrieval-augmented generation (RAG) system with FAISS vector storage.
* **Interactive Visualizations** : Dynamic charts, heatmaps, and model performance metrics powered by Plotly.
* **Data Privacy** : Local processing with secure handling of sensitive medical data, as outlined in the [Privacy Policy](https://grok.com/chat/politique_confidentialite.md).

## Deployment

The DataCarelytics application is live and accessible at:
[**https://has1elb-medicalanalysisapp-main-xojuse.streamlit.app/**](https://has1elb-medicalanalysisapp-main-xojuse.streamlit.app/)
Visit the URL to interact with the platform directly in your browser.

## Project Structure

```
has1elb-datacarelytics/
├── README.md                     # Project overview and instructions
├── main.py                       # Main Streamlit application entry point
├── politique_confidentialite.md  # Privacy policy for data handling
├── requirements.txt              # Python dependencies
├── style.css                    # Global CSS styles for the Streamlit app
├── medical_chatbot/             # Image-based medical chatbot
│   ├── __init__.py
│   ├── app.py                   # Streamlit app for image-based chatbot
│   ├── utils.py                 # Utility functions for image processing and API calls
│   └── assets/
│       └── style.css            # Custom CSS for the image chatbot
├── medical_chatbot2/            # Text-based RAG chatbot
│   ├── README.md                # Instructions for the RAG chatbot
│   ├── create_vectorstore.py    # Script to create FAISS vector store from PDFs
│   ├── main.py                  # Streamlit app for the RAG chatbot
│   └── vectorstore/
│       └── db_faiss/           # FAISS vector store for document embeddings
├── ner_streamlit_app/           # Named Entity Recognition (NER) for clinical text
│   ├── __init__.py
│   ├── app.py                   # Streamlit app for NER and PDF report generation
│   ├── requirements.txt         # NER-specific dependencies
│   ├── utils.py                 # Utility functions for NER processing
│   └── model/
│       └── crf_ner_model.joblib # Pre-trained CRF model for NER
├── tabular_data/                # Tabular data analysis and machine learning
│   ├── __init__.py
│   ├── config.py                # Streamlit page configuration
│   ├── data_processing.py       # Data preprocessing and visualization
│   ├── ml_models.py             # Machine learning model training and evaluation
│   └── utils.py                 # Utility functions for visualizations
└── trt_image/                   # Medical image processing
    ├── advanced_tumor_classification_model.h5  # Pre-trained MRI tumor classification model
    ├── chest_xray_model.h5                    # Pre-trained X-ray classification model
    └── main.py                                # Streamlit app for image analysis
```

## Prerequisites

* Python 3.8 or higher
* A modern web browser (e.g., Chrome, Firefox)
* Groq API key for chatbot functionalities (obtain from [Groq Console](https://console.groq.com/))
* Optional: CUDA-enabled GPU for faster deep learning model inference

## Setup Instructions

1. **Clone the Repository** :

```bash
   git clone https://github.com/HAS1ELB/DataCarelytics.git
   cd has1elb-datacarelytics
```

1. **Create a Virtual Environment** :

```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
```

1. **Install Dependencies** :

```bash
   pip install -r requirements.txt
```

1. **Set Environment Variables** :
   Create a `.env` file in the project root:

```plaintext
   GROQ_API_KEY=your_groq_api_key_here
```

   Replace `your_groq_api_key_here` with your Groq API key from [Groq Console](https://console.groq.com/).

1. **Prepare FAISS Vector Store for Text-based Chatbot** :

* Place medical PDF documents in the `medical_chatbot2/data/` directory.
* Run the following command to create the FAISS vector store:
  ```bash
  python medical_chatbot2/create_vectorstore.py
  ```

1. **Run the Application Locally** :

```bash
   streamlit run main.py
```

   The application will open in your default web browser. Alternatively, access the deployed version at [https://has1elb-medicalanalysisapp-main-xojuse.streamlit.app/](https://has1elb-medicalanalysisapp-main-xojuse.streamlit.app/).

## Usage

1. **Access the Application** :

* **Deployed Version** : Visit [https://has1elb-medicalanalysisapp-main-xojuse.streamlit.app/](https://has1elb-medicalanalysisapp-main-xojuse.streamlit.app/).
* **Local Version** : Run `streamlit run main.py`. A consent popup will appear due to the handling of sensitive medical data. Check the consent box to proceed, agreeing to the [Privacy Policy](https://grok.com/chat/politique_confidentialite.md).

1. **Navigate Modules** :
   Use the sidebar to select a module:

* **Home** : Overview of features and example applications.
* **Tabular Data Analysis** : Upload or load sample datasets (e.g., Diabetes, Heart Disease) for preprocessing, visualization, and machine learning.
* **Image Processing** : Analyze MRI or X-ray images using pre-trained deep learning models.
* **Text Analysis** : Extract entities from clinical text and generate structured PDF reports.
* **Chatbot** : Choose between:
  * **Image-based Chatbot** : Upload medical images and ask questions, powered by Groq’s Llama 4 Scout.
  * **Text-based RAG Chatbot** : Query medical knowledge from PDF documents using a FAISS vector store.

1. **Export Results** :

* Tabular Data: Export trained machine learning models as `.joblib` files or as a ZIP archive.
* Text Analysis: Download structured medical reports as PDFs.
* Visualizations: Save interactive Plotly charts for reports or publications.

## Modules

### 1. Tabular Data Analysis

* **Functionality** : Load datasets (CSV/Excel or sample datasets), preprocess data (handle missing values, outliers, feature engineering), and train machine learning models (e.g., Random Forest, XGBoost).
* **Features** :
* Data visualization (histograms, correlation heatmaps, scatter plots).
* Feature selection (ANOVA, RFE, L1-based).
* Model evaluation with metrics (accuracy, precision, RMSE, etc.).
* SHAP-based model interpretability.
* Cross-validation and learning curves.
* **Usage** : Select "Tabular Data Analysis" in the sidebar, upload a dataset or use a sample, then preprocess and train models under the appropriate sections.

### 2. Image Processing

* **Functionality** : Analyze MRI and X-ray images using pre-trained TensorFlow models for tumor classification or pneumonia detection.
* **Models** :
* `advanced_tumor_classification_model.h5`: Classifies MRI images into glioma, meningioma, no tumor, or pituitary tumor.
* `chest_xray_model.h5`: Classifies X-ray images as normal or pneumonia (binary classification).
* **Usage** : Select "Image Processing," choose analysis type (MRI or X-ray), upload an image, and view predictions with confidence scores.

### 3. Medical Text Analysis

* **Functionality** : Extract medical entities (e.g., symptoms, diagnoses) from unstructured text using a CRF-based NER model and generate structured PDF reports.
* **Usage** : Select "Text Analysis," input patient details, upload a text file, and download the generated PDF report.

### 4. Medical Chatbots

* **Image-based Chatbot** :
* Upload medical images and ask questions.
* Uses Groq’s Llama 4 Scout model for detailed medical insights.
* Usage: Select "Chatbot," choose "Image-based Chatbot," upload an image, and enter a query.
* **Text-based RAG Chatbot** :
* Query medical knowledge from PDF documents stored in a FAISS vector store.
* Usage: Select "Chatbot," choose "Text-based RAG Chatbot," and enter a medical question.

## Dependencies

See `requirements.txt` for a complete list of Python packages. Key dependencies include:

* `streamlit`: For the web interface.
* `pandas`, `numpy`, `scikit-learn`: For data processing and machine learning.
* `tensorflow`: For deep learning models.
* `langchain`, `langchain-groq`, `sentence-transformers`, `faiss-cpu`: For the RAG chatbot.
* `spacy`, `sklearn-crfsuite`: For NER.
* `plotly`: For interactive visualizations.

## Data Privacy

* **Local Processing** : All data processing occurs locally to ensure privacy (except for API calls to Groq for chatbots).
* **Sensitive Data** : Medical datasets, images, and text are handled securely, with data deleted at the end of the user session.
* **Consent** : Users must consent to data processing as per the [Privacy Policy](https://grok.com/chat/politique_confidentialite.md).
* **API Usage** : The Groq API is used for chatbot functionalities, with secure API key management via environment variables.

## Example Applications

* **Clinical Prediction** : Build models to predict patient outcomes (e.g., diabetes risk, heart disease).
* **Image Classification** : Detect tumors in MRI scans or pneumonia in X-rays.
* **Text Mining** : Extract structured data from clinical notes for research or reporting.
* **Medical Q&A** : Use chatbots to interpret images or answer queries based on medical literature.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

Please ensure code follows PEP 8 style guidelines and includes appropriate tests.

## License

This project is licensed under the MIT License. See the [LICENSE](https://grok.com/chat/LICENSE) file for details.

## Contact

For issues, questions, or to exercise data privacy rights (access, rectification, deletion), contact  **[elbahraouihassan54@gmail.com](mailto:elbahraouihassan54@gmail.com)** . For API-related queries, visit [xAI API](https://x.ai/api).
