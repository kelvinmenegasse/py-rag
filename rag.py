import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    /* Chat Input Styling */
    .stChatInput input {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
        border: 1px solid #3A3A3A !important;
    }
    
    /* User Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #1E1E1E !important;
        border: 1px solid #3A3A3A !important;
        color: #E0E0E0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Assistant Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #2A2A2A !important;
        border: 1px solid #404040 !important;
        color: #F0F0F0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Avatar Styling */
    .stChatMessage .avatar {
        background-color: #00FFAA !important;
        color: #000000 !important;
    }
    
    /* Text Color Fix */
    .stChatMessage p, .stChatMessage div {
        color: #FFFFFF !important;
    }
    
    .stFileUploader {
        background-color: #1E1E1E;
        border: 1px solid #3A3A3A;
        border-radius: 5px;
        padding: 15px;
    }
    
    h1, h2, h3 {
        color: #00FFAA !important;
    }
    </style>
    """, unsafe_allow_html=True)

selected_model= "deepseek-r1:8b"

PROMPT_TEMPLATE = """
Você é um assistente de pesquisa especializado. Use o contexto fornecido para responder à consulta.
Se a consulta não está relacionada ao contexto, diga que você não sabe. Seja conciso e fácil de entender.

Query: {user_query} 
Context: {document_context} 
Answer:
"""
PDF_STORAGE_PATH = 'document_store/pdfs/' # Caminho para armazenar os PDFs carregados
EMBEDDING_MODEL = OllamaEmbeddings(model=selected_model) # Modelo de Embeddings (embeddings são representações vetoriais de documentos)
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL) # Banco de dados de vetores para armazenar os documentos
LANGUAGE_MODEL = OllamaLLM(model=selected_model) # Modelo de linguagem para gerar respostas


# Funções auxiliares
# Função para salvar o arquivo PDF carregado
def save_uploaded_file(uploaded_file):
    file_path = PDF_STORAGE_PATH + uploaded_file.name
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

# Função para carregar os documentos PDF
def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

# Função para dividir os documentos em chunks
def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

# Função para indexar os documentos
def index_documents(document_chunks):
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)

# Função para encontrar documentos relacionados
def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)

# Função para gerar a resposta
def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})

# UI Configuration

st.title("📘 Pesquisa em documentos usando IA")
st.markdown("### Seu assistente de pesquisa de documentos")
st.markdown("---")

# File Upload Section
uploaded_pdf = st.file_uploader(
    "Carregar documento de pesquisa (PDF)",
    type="pdf",
    help="Selecione um documento no formato PDF para análise",
    accept_multiple_files=False

)

# Processamento do arquivo PDF
if uploaded_pdf:
    saved_path = save_uploaded_file(uploaded_pdf)
    raw_docs = load_pdf_documents(saved_path)
    processed_chunks = chunk_documents(raw_docs)
    index_documents(processed_chunks)
    
    # Display success message
    st.success("✅ Documento processado com sucesso! Faça perguntas sobre o documento.")
    
    # Chatbot Section
    user_input = st.chat_input("Faça sua pergunta relacionada ao documento...")
    
    # Display user input and AI response
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        
        # AI Response
        with st.spinner("Analisando documento..."):
            relevant_docs = find_related_documents(user_input)
            ai_response = generate_answer(user_input, relevant_docs)
            
        with st.chat_message("assistant", avatar="🤖"):
            st.write(ai_response)