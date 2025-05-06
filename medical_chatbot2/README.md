# Medical Chatbot

A medical chatbot that integrates a retrieval-augmented generation (RAG) system with Llama 4 via Groq API, providing reasoned medical answers based on PDF documents.

## Setup Instructions

1. **Clone the Repository** :

```bash
   git clone <repository-url>
   cd medical-chatbot-combined
```

1. **Create Virtual Environment** :

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

```
   GROQ_API_KEY=your_groq_api_key_here
```

   Replace `your_groq_api_key_here` with your Groq API key from `https://console.groq.com`.

1. **Create FAISS Vector Store** :
   Place medical PDFs in the `data/` directory, then run:

```bash
   python create_vectorstore.py
```

1. **Run the Application** :

```bash
   streamlit run main.py
```

## Usage

* Open the Streamlit app in your browser.
* Enter medical questions in the chat input.
* View responses with source documents from the FAISS vector store.

## Dependencies

See `requirements.txt` for a complete list of packages.

## Notes

* Ensure PDFs are placed in the `data/` directory before running `create_vectorstore.py`.
* The chatbot uses Llama 4 (`llama-4-scout-17b-16e-instruct`) via Groq’s API for fast inference.
