import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.llms import ollama
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ----------------------------
# Load Environment Variables
# ----------------------------
load_dotenv()
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = ""
os.environ["LANGSMITH_PROJECT"] = ""

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="🦙 LLaMA 3.2 Chat", page_icon="🧠", layout="centered")

# ----------------------------
# Dark Mode CSS Styling
# ----------------------------
st.markdown("""
    <style>
    body, .main, .stApp {
        background-color: #121212;
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
    }

    .stChatMessage {
        background-color: #1f1f1f;
        border-radius: 10px;
        padding: 12px;
        margin: 10px 0;
        box-shadow: 1px 1px 8px rgba(0,0,0,0.3);
    }

    .stTextInput > div > div > input, .stChatInput > div > div > input {
        background-color: #2a2a2a;
        color: #ffffff;
        border: 1px solid #444;
        border-radius: 8px;
        padding: 10px;
    }

    .stMarkdown h1, .stMarkdown h2 {
        color: #14b8a6;
    }

    .stButton > button {
        background-color: #14b8a6;
        color: #000;
        border: none;
        border-radius: 8px;
    }

    hr {
        border: 0;
        height: 1px;
        background: linear-gradient(to right, #14b8a6, transparent);
        margin: 20px 0;
    }

    .stSelectbox, .stSlider {
        background-color: #1f1f1f;
        color: #fff;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Header
# ----------------------------
st.markdown("""
    <h1 style='text-align: center;'>🦙 Chat with Umar's Assistant</h1>
    <p style='text-align: center; color: #aaaaaa;'>Dark Mode | LangChain + Streamlit + Ollama</p>
    <hr/>
""", unsafe_allow_html=True)

# ----------------------------
# Sidebar Settings
# ----------------------------
with st.sidebar:
    st.header("⚙️ Model Settings")
    model_name = st.selectbox("Choose Ollama Model", options=["llama3.2"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, step=0.05)
    max_tokens = st.slider("Max Tokens", 64, 2048, 512, step=64)
    st.caption("Use `ollama pull llama3` for the model.")

# ----------------------------
# Load LLaMA Model
# ----------------------------
@st.cache_resource(show_spinner="🚀 Loading LLaMA 3.2...")
def load_llm(model, temp, tokens):
    return ollama.Ollama(model=model, temperature=temp, num_predict=tokens)

llm = load_llm(model_name, temperature, max_tokens)

# ----------------------------
# Prompt Setup
# ----------------------------
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are a helpful and intelligent assistant powered by LLaMA 3.2."),
    HumanMessagePromptTemplate.from_template("Question: {question}")
])
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# ----------------------------
# Chat History
# ----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ----------------------------
# Chat Input & Response Handling
# ----------------------------
user_input = st.chat_input("💬 Type your message...")

if user_input:
    with st.spinner("🧠 Thinking..."):
        response = chain.invoke({"question": user_input})
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("LLaMA", response))

# ----------------------------
# Show Chat History
# ----------------------------
for speaker, msg in st.session_state.chat_history:
    with st.chat_message("user" if speaker == "You" else "assistant"):
        if speaker == "You":
            st.markdown(f"**{speaker}:** {msg}")
        else:
            st.markdown(f"🦙 **Assistant:** {msg}")
