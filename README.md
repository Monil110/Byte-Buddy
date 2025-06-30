# Byte Buddy 🤖

Byte Buddy is an AI-powered assistant designed to help students of RVCE (Rashtreeya Vidyalaya College of Engineering) with college-related queries. It leverages advanced language models and vector search to answer questions about academics, campus resources, events, and more.

## Technical Overview

**Byte Buddy** is built using a Retrieval-Augmented Generation (RAG) architecture, combining document retrieval with generative AI for accurate, context-aware answers. Here's how it works:

- **PDF Ingestion:** All PDF files in the specified knowledge base directory are loaded and parsed using `PyMuPDFLoader` from LangChain.
- **Text Chunking:** Documents are split into overlapping chunks (default: 2000 characters, 200 overlap) using `RecursiveCharacterTextSplitter` to preserve context across sections.
- **Embeddings:** Each chunk is converted into a vector embedding using Google Generative AI Embeddings (`GoogleGenerativeAIEmbeddings`).
- **Vector Store:** Embeddings and their corresponding text chunks are stored in a FAISS vector index for efficient similarity search.
- **Retrieval:** When a user asks a question, the app retrieves the top-k most relevant chunks (default: 15) from the vector store.
- **Prompting:** The retrieved context and the user's question are combined in a prompt template, which is sent to a Google Gemini model (`ChatGoogleGenerativeAI`) for answer generation.
- **UI:** The app is built with Streamlit, providing a modern, interactive web interface. Folium is used for map visualization when campus directions are requested.

**Key Libraries Used:**
- Streamlit (UI)
- LangChain (RAG pipeline, document loaders, text splitters)
- FAISS (vector similarity search)
- Google Generative AI (embeddings and chat model)
- PyMuPDF (PDF parsing)
- Folium & streamlit-folium (map rendering)
- python-dotenv (environment variable management)

**Scalability:**
- Byte Buddy can handle dozens of PDFs efficiently on a typical laptop. For very large document sets, consider increasing system RAM or preprocessing the index offline.

## Features

- **Ask Anything About RVCE:** Get answers about syllabus, timetable, registration, campus facilities, and more.
- **PDF Knowledge Base:** The bot uses information from college-related PDF documents.
- **Conversational AI:** Friendly, professional, and context-aware responses.
- **Campus Map:** View the RVCE campus map when you ask for directions or maps.
- **Modern UI:** Built with Streamlit for an interactive web experience.

## Setup Instructions

### 1. Clone or Download the Repository

Navigate to your project directory:

```sh
cd "C:\Users\MONIL\Desktop\Byte-Buddy"
```

### 2. Install Dependencies

Make sure you have Python 3.8+ installed. Then, install the required packages:

```sh
pip install streamlit langchain faiss-cpu python-dotenv PyMuPDF folium streamlit-folium google-generativeai
```

### 3. Set Up Google API Key

- Create a `.env` file in the project root with the following content:
  ```
  GOOGLE_API_KEY=your_google_api_key_here
  ```
- Replace `your_google_api_key_here` with your actual Google Generative AI API key.

### 4. Prepare PDF Knowledge Base

- Place all relevant college PDF files in the directory:
  ```
  C:/Users/MONIL/Desktop/aiml lab see
  ```
- The app will automatically load all PDFs from this folder.

### 5. Run the App

```sh
streamlit run app4.py
```

- The app will open in your browser at [http://localhost:8501](http://localhost:8501).

## Usage

- Enter your question in the text area (e.g., "What is the timetable for 2nd year CSE?").
- For campus directions or maps, ask questions like "Show me the campus map" or "How do I get to the library?".
- The bot will respond with helpful, RVCE-specific information.

## Troubleshooting

- **No API Key:** Make sure your `.env` file is present and contains a valid Google API key.
- **No PDFs Found:** Ensure your PDF files are in the correct directory (`C:/Users/MONIL/Desktop/Byte-Buddy`).
- **Missing Packages:** Install any missing Python packages as shown above.

## Customization

- To add or update knowledge, simply add more PDF files to the knowledge base directory and restart the app.
- You can customize the prompt and response style in `app4.py`.

## License

This project is for educational purposes at RVCE.
