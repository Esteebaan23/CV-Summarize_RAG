# 📄 CV Summarizer & Q&A (RAG) — PDF Chatbot with LangChain + FAISS + Watsonx

This project is a **Retrieval-Augmented Generation (RAG)** application that allows users to **upload a PDF (e.g., a resume/CV)** and instantly:

✅ Generate an **automatic summary** of the candidate as soon as the PDF is uploaded  
✅ Ask questions about the document and receive **grounded answers** based on the PDF content  
✅ Easily adapt the same pipeline to summarize and query **other private company documents** (reports, policies, manuals, SOPs, etc.)

Built with:
- **Gradio** (UI)
- **LangChain** (RAG pipeline utilities)
- **FAISS** (vector store for fast similarity search)
- **IBM watsonx.ai** (LLM + embeddings)

---

## 🚀 Features

- **Auto-summary on upload** (no button required)
- **Regenerate summary** button (re-run summarization anytime)
- **PDF Q&A chatbot** powered by RAG (retrieval + LLM generation)
- **Session-safe scaling** using `gr.State()` (no global variables → supports multiple users safely)
- Uses **FAISS indexing per session** for fast retrieval during Q&A

---

## 🧠 How It Works (RAG Flow)

1. User uploads a **PDF**
2. The app extracts text from the PDF
3. The text is split into chunks using `RecursiveCharacterTextSplitter`
4. Chunks are embedded using **IBM Slate embeddings**
5. Embeddings are stored in **FAISS**
6. For each question:
   - Retrieve top-k relevant chunks
   - Answer using **Watsonx LLM** with the retrieved context

---

## 📁 Project Structure
├── app.py # Main Gradio app
└── README.md # Documentation



