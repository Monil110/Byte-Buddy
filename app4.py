import streamlit as st
import os
import warnings
from dotenv import load_dotenv
import faiss
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_core.prompts import ChatPromptTemplate
import folium
from streamlit_folium import st_folium
import google.generativeai as genai

st.set_page_config(page_title="RVSmartBot", page_icon=":robot_face:")

# Initialize environment variables and suppress warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore")
load_dotenv()

# Configure API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Please set your GOOGLE_API_KEY in the .env file")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# Load and process PDFs
pdf_directory = "C:/Users/MONIL/Desktop/aiml lab see"
pdfs = [os.path.join(pdf_directory, file) for file in os.listdir(pdf_directory) if file.endswith('.pdf')]

if not pdfs:
    raise ValueError(f"No PDF files found in directory {pdf_directory}.")

@st.cache_data
def load_and_process_pdfs():
    docs = []
    for pdf in pdfs:
        loader = PyMuPDFLoader(pdf)
        try:
            pages = loader.load()
            docs.extend(pages)
        except Exception as e:
            st.error(f"Error loading {pdf}: {e}")
    return docs

docs = load_and_process_pdfs()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)

# Initialize embeddings
@st.cache_resource
def initialize_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

embeddings = initialize_embeddings()

# Create FAISS index
@st.cache_resource
def create_vector_store():
    test_embedding = embeddings.embed_query("test query")
    d = len(test_embedding)
    index = faiss.IndexFlatL2(d)
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    vector_store.add_documents(documents=chunks)
    return vector_store

vector_store = create_vector_store()

# Define retriever and prompt
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

prompt = ChatPromptTemplate.from_template("""
You are an AI-powered assistant designed exclusively to assist students of RVCE (Rashtreeya Vidyalaya College of Engineering) with college-related tasks. Your tone should be friendly, approachable, and professional. Provide accurate and concise answers, and always stay within the scope of college-related queries.

You can assist students in the following areas:
- **Academic help:** Information about classes, schedules, assignments, exams, and study tips.
- **Campus resources:** Guidance on accessing campus facilities like the library, labs, or counseling services.
- **Events and activities:** Details about clubs, events, and extracurricular activities.
- **Administrative help:** Queries about registration, fee payment, academic records, and other administrative tasks.
- **Personal organization:** Suggestions for time management, stress handling, and productivity tools.

**Guidelines for responses:**
1. If the question is unrelated to RVCE or the scope above, respond politely by stating that you only provide information about RVCE and its students.
2. Avoid making assumptions or providing information you are unsure about. Direct users to appropriate college resources or offices for further assistance.
3. Always use a conversational tone to make the interaction welcoming and helpful.

**Example interactions:**

**Student:** "How do I register for a workshop?"
**Chatbot:** "You can register for workshops through the RVCE student portal or by contacting the specific department organizing it. Let me know if you need more details!"

**Student:** "What is the population of India?"
**Chatbot:** "I'm here to assist with queries related to RVCE and its students. For general questions, you might want to try a search engine."

Question: {question} 
Context: {context} 
Answer:
""")

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

@st.cache_resource
def initialize_model():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3,
        max_tokens=1000
    )

model = initialize_model()

# Define RAG chain
rag_chain = (
    RunnableMap({
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    })
    | prompt
    | model
    | StrOutputParser()
)

# Optional header image
if os.path.exists("rvce.jpg"):
    st.image("rvce.jpg", width=150)

# Custom CSS
st.markdown(f"""
    <style>
    body {{
        font-family: 'Arial', sans-serif;
        background-image: url("assets/bg.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .stApp {{
        background-color: rgba(0, 0, 0, 0.5);
    }}
    .header {{
        color: #007BFF;
        font-size: 36px;
        text-align: center;
        margin-top: 20px;
    }}
    .question-input {{
        background-color: #ffffff;
        border-radius: 8px;
        padding: 10px;
    }}
    .response-section {{
        background-color: #ffffff;
        border-radius: 8px;
        padding: 15px;
        margin-top: 20px;
    }}
    .error {{
        color: red;
        font-weight: bold;
    }}
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="header">RVSmartBot <span style="font-size: 40px;">🤖</span></h1>', unsafe_allow_html=True)
st.markdown('<h3>Ask questions about your syllabus, timetable, or anything related to your college experience.</h3>', unsafe_allow_html=True)

# Text input only (voice input removed)
question = st.text_area(
    "Enter your question:", 
    key="question_input", 
    help="Ask about your classes, exams, or any other college-related queries.", 
    height=150
)

if question:
    with st.spinner("Processing your query..."):
        try:
            answer = rag_chain.invoke(question)
            st.markdown("### Response:")
            st.write(answer)

            # Show map if user asks for it
            if "direction" in question.lower() or "map" in question.lower():
                rvce_location = [12.9237, 77.4987]
                m = folium.Map(location=rvce_location, zoom_start=17)
                st.title("RVCE Campus Map")
                st.write("Explore the map to locate different departments and amenities on the RVCE campus.")
                st_folium(m, width=800, height=600)
        except Exception as e:
            st.markdown(f'<div class="error">Error: {e}</div>', unsafe_allow_html=True)
