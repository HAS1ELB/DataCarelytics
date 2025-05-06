import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

DB_FAISS_PATH = "medical_chatbot2/vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt():
       custom_prompt_template = """
       Below is a task description along with additional context provided in the input section. Your goal is to provide a well-reasoned response that effectively addresses the request.

       Before crafting your answer, take a moment to carefully analyze the question. Develop a clear, step-by-step thought process to ensure your response is both logical and accurate.

       ### Task:
       You are a medical expert specializing in clinical reasoning, diagnostics, and treatment planning. Answer the medical question below using your advanced knowledge and the provided context.

       ### Context:
       {context}

       ### Query:
       {question}

       ### Answer:
       <think>
       Let's break down the query step-by-step to provide a precise and accurate response based on the context.
       </think>
       """
       prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
       return prompt

def load_llm():
       groq_api_key = os.environ.get("GROQ_API_KEY")
       llm = ChatGroq(
           groq_api_key=groq_api_key,
           model_name="meta-llama/llama-4-scout-17b-16e-instruct",
           temperature=0.5,
           max_tokens=512
       )
       return llm

def main():
       st.title("MediBot: AI Medical Assistant")

       if 'messages' not in st.session_state:
           st.session_state.messages = []

       for message in st.session_state.messages:
           st.chat_message(message['role']).markdown(message['content'])

       prompt = st.chat_input("Ask a medical question:")

       if prompt:
           st.chat_message('user').markdown(prompt)
           st.session_state.messages.append({'role': 'user', 'content': prompt})

           try:
               vectorstore = get_vectorstore()
               if vectorstore is None:
                   st.error("Failed to load the vector store")

               qa_chain = RetrievalQA.from_chain_type(
                   llm=load_llm(),
                   chain_type="stuff",
                   retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                   return_source_documents=True,
                   chain_type_kwargs={'prompt': set_custom_prompt()}
               )

               response = qa_chain.invoke({'query': prompt})
               result = response["result"]
               source_documents = response["source_documents"]
               result_to_show = f"{result}\n\n**Source Documents:**\n{source_documents}"

               st.chat_message('assistant').markdown(result_to_show)
               st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

           except Exception as e:
               st.error(f"Error: {str(e)}")

if __name__ == "__main__":
       main()