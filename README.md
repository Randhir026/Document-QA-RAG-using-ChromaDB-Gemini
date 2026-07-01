# 📄 Document QA RAG using ChromaDB + Gemini

A simple Retrieval-Augmented Generation (RAG) application that answers questions from PDF and HTML documents using semantic search and Google Gemini.

This project demonstrates a low-cost RAG pipeline with document ingestion, vector embeddings, semantic retrieval, grounded answer generation, citations, and evaluation.

---

## 🚀 Features

* 📄 Load PDF documents using PyMuPDF
* 🌐 Load HTML documents using BeautifulSoup
* ✂️ Split documents into chunks using LangChain
* 🧠 Generate embeddings using Sentence Transformers (`all-MiniLM-L6-v2`)
* 🗂️ Store embeddings in ChromaDB
* 🔍 Perform semantic similarity search
* 🤖 Generate grounded answers using Google Gemini (Free Tier)
* 📚 Display document citations
* 📊 Evaluate retrieval quality, latency, and cost

---

## 🛠️ Tech Stack

* Python
* ChromaDB
* Sentence Transformers
* LangChain
* Google Gemini API
* PyMuPDF
* BeautifulSoup

---

## 📂 Project Structure

```text
document_qa_rag/
│
├── config.py
├── loader.py
├── ingest.py
├── rag.py
├── evaluation.py
├── requirements.txt
├── .env
├── README.md
│
├── documents/
│      AI Training Document.pdf
│
└── chroma_db/
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/randhir026/document_qa_rag.git
cd document_qa_rag
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it.

Windows

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔑 Configure API Key

Create a `.env` file.

```text
GOOGLE_API_KEY=YOUR_GEMINI_API_KEY
```

Get your free Gemini API key from Google AI Studio.

---

## 📄 Add Documents

Place your PDF and/or HTML documents inside the `documents` folder.

Example:

```text
documents/
│
├── AI Training Document.pdf
├── sample.html
```

---

## 📥 Ingest Documents

Generate embeddings and store them in ChromaDB.

```bash
python ingest.py
```

Example output:

```text
Loading documents...

Stored 120 chunks in ChromaDB
```

---

## ❓ Ask Questions

Run the QA system:

```bash
python rag.py
```

Example:

```text
Question:
What law governs the eBay User Agreement?
```

Output:

```text
Answer:

The laws of the State of Utah govern the eBay User Agreement.

Sources

AI Training Document.pdf (Page 13)
```

---

## 📊 Evaluation

Run:

```bash
python evaluation.py
```

The evaluation reports:

* Retrieval Quality
* Answer Quality
* Average Latency
* Cost

---

## 📚 Example Questions

* What law governs the eBay User Agreement?
* What is Informal Dispute Resolution?
* Who pays arbitration fees?
* What is Batch Arbitration?
* Where should an Opt-Out Notice be mailed?
* Summarize the Legal Disputes section.

---

## 📦 Dependencies

* chromadb
* sentence-transformers
* langchain
* langchain-text-splitters
* google-genai
* PyMuPDF
* beautifulsoup4
* python-dotenv

---

## 📈 Future Improvements

* FastAPI REST API
* Streamlit Web Interface
* Hybrid Search (BM25 + Vector Search)
* OCR Support for Scanned PDFs
* Retrieval Evaluation Metrics
* Docker Deployment

---

## 👨‍💻 Author

Randhir Kumar

MCA Graduate | AI & Machine Learning Enthusiast

---

## 📄 License

This project is released under the MIT License.
