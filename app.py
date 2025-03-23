import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    AIMessagePromptTemplate,
)

# Custom CSS styling
st.markdown("""
<style>
    /* Existing styles */
    .main {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #2d2d2d;
    }
    .stTextInput textarea {
        color: #ffffff !important;
    }
    
    /* Add these new styles for select box */
    .stSelectbox div[data-baseweb="select"] {
        color: white !important;
        background-color: #3d3d3d !important;
    }
    
    .stSelectbox svg {
        fill: white !important;
    }
    
    .stSelectbox option {
        background-color: #2d2d2d !important;
        color: white !important;
    }
    
    /* For dropdown menu items */
    div[role="listbox"] div {
        background-color: #2d2d2d !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)
st.title("DeepSeek Code")
st.caption("🚀 Your AI Pair Programmer")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox(
        "Choose Model",
        ["deepseek-r1:8b"],
        index=0
    )
    st.divider()
    st.markdown("### Model Capabilities")
    st.markdown("""
    - 🐍 Python Expert
    - 🐞 Debugging Assistant
    - 📝 Code Documentation
    - 💡 Solution Design
    """)
    st.divider()
    st.markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)")


# Initialize Ollama
llm_engine= ChatOllama(
    model=selected_model,
    temperature=0.1, # Quanto mais próximo de 0, mais coerente e previsível será a resposta. Quanto mais próximo de 1, mais aleatório/criativo será a resposta.
    max_tokens=2048, # Quanto maior o número, mais detalhada e complexa será a resposta, mas também mais demorada.
    verbose=True, # Se True, o modelo irá imprimir informações de depuração durante a execução.
    base_url="http://localhost:11434", # URL base do servidor Ollama.
)

# System prompt configuration
# Alterar o sistema para ser mais específico, atribuindo um contexto e instruções específicas para o modelo.
system_prompt = SystemMessagePromptTemplate.from_template(
    "Você é um assistente de programação que ajuda os usuários a escrever código Python. "
    "Você é um especialista em Python e pode ajudar os usuários a escrever código Python de alta qualidade. "
    "Você deve responder utilizando padrões de código Python válidos e seguir as melhores práticas de programação. "
    "Você pode responder melhorando o codigo com princípios CLEAN CODE e SOLID. "   
)

# Session state for management
if "message_log" not in st.session_state:
    st.session_state.message_log = [{
        "role": "ai",
        "content": "Olá! Como posso ajudar?"
    }]

# Chat container
chat_container = st.container()

# Display chat messages
with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


# Chat input and processing
user_query = st.chat_input("Digite sua pergunta aqui")

# Process user input
def generate_ai_response(prompt_chain):
    processing_pipeline=prompt_chain | llm_engine | StrOutputParser()
    return processing_pipeline.invoke({})

# Handle user input
def build_prompt_chain():
    prompt_sequence = [system_prompt]
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

# Process user input
if user_query:
    # Add user message to log
    st.session_state.message_log.append({"role": "user", "content": user_query})
    
    # Generate AI response
    with st.spinner("🧠 Processando..."):
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)
    
    # Add AI response to log
    st.session_state.message_log.append({"role": "ai", "content": ai_response})
    
    # Rerun to update chat display
    st.rerun()