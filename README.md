# 🤖 AI Voice Document Chatbot

An intelligent Streamlit-based web app that allows users to upload documents (PDF, TXT, DOC, DOCX), ask questions about them using **voice or text**, and receive **AI-powered answers**. Built using powerful NLP models like **Sentence Transformers** and **FAISS** for vector search, it also supports voice response via **gTTS**.

---

## 🚀 Features

- 📂 Upload multiple document types: PDF, TXT, DOC, DOCX
- 🔍 AI-powered question answering based on document content
- 🗣️ Voice input support (SpeechRecognition)
- 🔊 Voice response using gTTS (optional toggle)
- 🧠 Embedding with `all-MiniLM-L6-v2` model
- ⚡ Fast semantic search using FAISS vector database
- 📊 Real-time analytics: document count, size, and chat messages
- 🧪 Customizable settings: Max results, voice toggle, etc.

---

## 🖼️ UI Preview

![AI Voice Document Chatbot](./33c0a25c-7800-4774-b411-9e900b5555bd.png)
<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/951fad95-bd62-48a6-b559-79247de9138b" />


---

## 📦 Tech Stack

- **Frontend/UI:** Streamlit
- **Voice Input:** SpeechRecognition
- **Voice Output:** gTTS
- **NLP Model:** SentenceTransformers (`all-MiniLM-L6-v2`)
- **Search Engine:** FAISS
- **Document Handling:** PyPDF2, python-docx
- **Data Processing:** NumPy, Pandas
- **Backend:** Python

