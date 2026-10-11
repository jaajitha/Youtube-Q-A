import os
import re
import hashlib
from typing import Optional, List, Any
import streamlit as st
from dotenv import load_dotenv, set_key

# LLM Providers
import groq
from groq import Groq as GroqClient
from langchain_groq import ChatGroq

import together
from together import Together as TogetherClient

# LangChain and Vectorstore imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.language_models.llms import LLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from transcript_loader import extract_video_id, get_video_transcript_data, get_video_transcript

# Load environment variables from .env
load_dotenv()

# Streamlit Page Setup
st.set_page_config(
    page_title="YouTube Video Q&A",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #FF0033 0%, #FF6B6B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #718096;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .provider-pill {
        display: inline-block;
        padding: 4px 10px;
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 9999px;
        font-size: 0.85rem;
        color: #166534;
        font-weight: 600;
        margin-bottom: 10px;
    }
    .answer-card {
        background-color: rgba(255, 0, 51, 0.06);
        border: 1px solid rgba(255, 0, 51, 0.25);
        border-left: 5px solid #FF0033;
        border-radius: 8px;
        padding: 16px 20px;
        margin-top: 10px;
        margin-bottom: 14px;
        color: inherit !important;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    .source-box {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 6px;
        padding: 12px 14px;
        margin-bottom: 8px;
        font-size: 0.92rem;
        line-height: 1.5;
        color: inherit !important;
    }
</style>
""", unsafe_allow_html=True)

# Directory to cache FAISS vector stores
VECTOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectorstores")
os.makedirs(VECTOR_DIR, exist_ok=True)
ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


# Modern Together AI LLM wrapper for LangChain LCEL
class TogetherLLM(LLM):
    api_key: str
    model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
    temperature: float = 0.3
    max_tokens: int = 700

    @property
    def _llm_type(self) -> str:
        return "together"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        clean_key = self.api_key.strip().strip("'\"")
        client = TogetherClient(api_key=clean_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=stop,
        )
        return response.choices[0].message.content or ""


# Cache embeddings model in memory
@st.cache_resource(show_spinner="Loading HuggingFace embedding model...")
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# Initialize Session State
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []
if "current_video_id" not in st.session_state:
    st.session_state.current_video_id = None
if "input_url" not in st.session_state:
    st.session_state.input_url = ""

# Sidebar: Provider & Model Configuration
st.sidebar.markdown("### ⚙️ LLM Provider & Settings")

provider_choice = st.sidebar.radio(
    "Select AI Provider:",
    ["Groq (Free & Blazing Fast)", "Together AI"],
    index=0,
    help="Groq offers a generous free tier and ultra-low latency."
)

is_groq = provider_choice.startswith("Groq")

if is_groq:
    raw_key = os.environ.get("GROQ_API_KEY", "")
    # Fallback if key was previously saved as TOGETHER_API_KEY
    if not raw_key and os.environ.get("TOGETHER_API_KEY", "").strip().strip("'\"").startswith("gsk_"):
        raw_key = os.environ.get("TOGETHER_API_KEY", "")

    key_env = raw_key.strip().strip("'\"")
    key_input = st.sidebar.text_input(
        "Groq API Key",
        value=key_env,
        type="password",
        help="Free API key from https://console.groq.com/keys"
    )
    api_key = key_input.strip().strip("'\"")

    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
        st.sidebar.success("✅ Groq API key active")
    else:
        st.sidebar.warning("⚠️ No Groq key set.")
        st.sidebar.info("👉 [Get a Free Groq API Key (instant)](https://console.groq.com/keys)")

    # Dynamic Model Selection for Groq
    def get_groq_models(key: str):
        fallbacks = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
        if not key:
            return fallbacks
        try:
            g_client = GroqClient(api_key=key)
            models_data = g_client.models.list().data
            chat_models = [m.id for m in models_data if not ("whisper" in m.id or "guard" in m.id or "orpheus-arabic" in m.id)]
            if "openai/gpt-oss-120b" in chat_models:
                chat_models.remove("openai/gpt-oss-120b")
                chat_models.insert(0, "openai/gpt-oss-120b")
            return chat_models if chat_models else fallbacks
        except Exception:
            return fallbacks

    model_options = get_groq_models(api_key)
    selected_model = st.sidebar.selectbox("Groq Model", model_options, index=0)

    # Key actions
    col_save_env, col_test_key = st.sidebar.columns([1, 1])
    with col_save_env:
        if st.button("💾 Save .env"):
            if api_key:
                try:
                    set_key(ENV_PATH, "GROQ_API_KEY", api_key)
                    st.sidebar.success("Saved GROQ_API_KEY to .env!")
                except Exception:
                    with open(ENV_PATH, "a") as f:
                        f.write(f"\nGROQ_API_KEY={api_key}\n")
                    st.sidebar.success("Saved GROQ_API_KEY to .env!")
            else:
                st.sidebar.error("Enter a key first.")

    with col_test_key:
        if st.button("🔍 Test Key"):
            if not api_key:
                st.sidebar.error("Enter a key first.")
            else:
                with st.sidebar.spinner("Testing Groq key..."):
                    try:
                        g_client = GroqClient(api_key=api_key)
                        g_client.models.list()
                        st.sidebar.success("🎉 Groq key verified successfully!")
                    except groq.AuthenticationError:
                        st.sidebar.error("❌ Invalid Groq key. Verify at https://console.groq.com/keys")
                    except Exception as e:
                        st.sidebar.warning(f"Verification error: {e}")

else:
    # Together AI configuration
    raw_key = os.environ.get("TOGETHER_API_KEY", "")
    key_env = raw_key.strip().strip("'\"")
    key_input = st.sidebar.text_input(
        "Together AI API Key",
        value=key_env,
        type="password",
        help="API key from https://api.together.ai/settings/api-keys"
    )
    api_key = key_input.strip().strip("'\"")

    if api_key:
        os.environ["TOGETHER_API_KEY"] = api_key
        if "." in api_key:
            st.sidebar.warning("⚠️ The key contains a '.' — this appears to be a web session token rather than an API key. API keys from api.together.ai/settings/api-keys are 64-character hex strings.")
        else:
            st.sidebar.success("✅ Together API key entered")
    else:
        st.sidebar.warning("⚠️ No Together key set.")
        st.sidebar.info("👉 [Get Together AI Key](https://api.together.ai/settings/api-keys)")

    # Model Selection for Together
    model_options = [
        "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "Qwen/Qwen2.5-72B-Instruct"
    ]
    selected_model = st.sidebar.selectbox("Together Model", model_options, index=0)

    # Key actions
    col_save_env, col_test_key = st.sidebar.columns([1, 1])
    with col_save_env:
        if st.button("💾 Save .env"):
            if api_key:
                try:
                    set_key(ENV_PATH, "TOGETHER_API_KEY", api_key)
                    st.sidebar.success("Saved TOGETHER_API_KEY to .env!")
                except Exception:
                    with open(ENV_PATH, "a") as f:
                        f.write(f"\nTOGETHER_API_KEY={api_key}\n")
                    st.sidebar.success("Saved TOGETHER_API_KEY to .env!")
            else:
                st.sidebar.error("Enter a key first.")

    with col_test_key:
        if st.button("🔍 Test Key"):
            if not api_key:
                st.sidebar.error("Enter a key first.")
            else:
                with st.sidebar.spinner("Testing Together key..."):
                    try:
                        t_client = TogetherClient(api_key=api_key)
                        t_client.models.list()
                        st.sidebar.success("🎉 Together key verified successfully!")
                    except together.AuthenticationError:
                        st.sidebar.error("❌ Key rejected (401 Unauthorized). Check https://api.together.ai/settings/api-keys")
                    except Exception as e:
                        st.sidebar.warning(f"Verification error: {e}")

# Common generation parameters
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
max_tokens = st.sidebar.slider("Max Tokens", min_value=128, max_value=1500, value=700, step=64)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 RAG Chunking Parameters")
chunk_size = st.sidebar.number_input("Chunk Size (characters)", min_value=300, max_value=3000, value=1000, step=100)
chunk_overlap = st.sidebar.number_input("Chunk Overlap", min_value=0, max_value=500, value=150, step=25)

if st.sidebar.button("🧹 Reset Vector Cache"):
    st.session_state.qa_history = []
    st.session_state.current_video_id = None
    st.cache_resource.clear()
    st.sidebar.success("Cache cleared!")

# Main Page Header
st.markdown('<div class="main-title">🎬 YouTube Video Q&A Assistant</div>', unsafe_allow_html=True)
provider_label = "⚡ Groq (Ultra-fast)" if is_groq else "Together AI"
st.markdown(
    f'<div class="sub-title">Extract transcripts from YouTube videos, build high-speed vector embeddings, and ask questions using LangChain & <strong>{provider_label}</strong>.</div>',
    unsafe_allow_html=True
)

# Quick sample video pills
st.markdown("##### 💡 Try with a sample video:")
col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    if st.button("🧠 3Blue1Brown - Neural Networks"):
        st.session_state.input_url = "https://www.youtube.com/watch?v=aircAruvnKk"
with col_s2:
    if st.button("🤖 Andrej Karpathy - Intro to LLMs"):
        st.session_state.input_url = "https://www.youtube.com/watch?v=zjkBMFhNj_g"
with col_s3:
    if st.button("🎓 Steve Jobs - Stanford Speech"):
        st.session_state.input_url = "https://www.youtube.com/watch?v=UF8uR6Z6KLc"

# URL Input Field
video_url = st.text_input(
    "Enter YouTube Video URL:",
    value=st.session_state.input_url,
    placeholder="https://www.youtube.com/watch?v=... or https://youtu.be/...",
    key="url_input"
)

# Helper function to get consistent hash
def get_video_hash(vid_id: str) -> str:
    return hashlib.md5(vid_id.encode()).hexdigest()

if video_url:
    try:
        video_id = extract_video_id(video_url)
        clean_watch_url = f"https://www.youtube.com/watch?v={video_id}"

        # If user changed URL, reset QA history
        if st.session_state.current_video_id != video_id:
            st.session_state.current_video_id = video_id
            st.session_state.qa_history = []

        col_player, col_stats = st.columns([1.2, 1])
        with col_player:
            st.video(clean_watch_url)

        with col_stats:
            video_hash = get_video_hash(video_id)
            store_path = os.path.join(VECTOR_DIR, video_hash)

            embeddings = get_embeddings_model()

            vectorstore = None
            transcript_text = ""
            segments = []

            # Check if vectorstore exists
            if os.path.exists(store_path):
                with st.spinner("Loading cached vector embeddings..."):
                    vectorstore = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
                    st.success("⚡ Loaded indexed vectorstore from cache.")
            else:
                with st.spinner("Extracting transcript and generating embeddings..."):
                    video_id, segments = get_video_transcript_data(video_url)
                    transcript_text = " ".join([s["text"] for s in segments if s.get("text")])

                    if not transcript_text.strip():
                        raise ValueError("No transcript text found for this video.")

                    # Split text
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        length_function=len
                    )
                    chunks = splitter.split_text(transcript_text)

                    # Create FAISS vectorstore
                    vectorstore = FAISS.from_texts(chunks, embeddings)
                    vectorstore.save_local(store_path)
                    st.success(f"✅ Indexed {len(chunks)} text chunks successfully!")

            st.info(f"**Video ID:** `{video_id}`")

        # Collapsible Transcript Preview
        with st.expander("📄 View Full Video Transcript"):
            if not transcript_text:
                try:
                    transcript_text = get_video_transcript(video_url)
                except Exception:
                    transcript_text = "(Transcript cached in vectorstore)"
            st.write(transcript_text)

        st.markdown("---")
        st.markdown("### 💬 Ask Questions About the Video")

        # Quick question prompt buttons
        q_col1, q_col2, q_col3 = st.columns(3)
        suggested_q = None
        with q_col1:
            if st.button("📌 Summarize Key Takeaways"):
                suggested_q = "Summarize the key takeaways and main concepts of this video in 3 to 5 clear bullet points."
        with q_col2:
            if st.button("🎯 Main Arguments / Purpose"):
                suggested_q = "What is the primary argument, goal, or motivation explained in this video?"
        with q_col3:
            if st.button("💡 Practical Examples Given"):
                suggested_q = "What specific examples or demonstrations are described in the video?"

        # Question input
        user_question = st.text_input(
            "Your Question:",
            value=suggested_q if suggested_q else "",
            placeholder="e.g. What is the central message of this video?",
            key="user_question_input"
        )

        col_ask, col_clear_qa = st.columns([1, 4])
        with col_ask:
            ask_pressed = st.button("🚀 Ask AI", type="primary")

        if ask_pressed and user_question:
            if not api_key:
                provider_name = "Groq" if is_groq else "Together AI"
                st.error(f"❌ Please provide a {provider_name} API Key in the left sidebar to generate answers.")
            else:
                provider_desc = f"Groq ({selected_model})" if is_groq else f"Together AI ({selected_model})"
                with st.spinner(f"Querying {provider_desc}..."):
                    try:
                        if is_groq:
                            llm = ChatGroq(
                                groq_api_key=api_key,
                                model_name=selected_model,
                                temperature=temperature,
                                max_tokens=max_tokens
                            )
                        else:
                            llm = TogetherLLM(
                                api_key=api_key,
                                model=selected_model,
                                temperature=temperature,
                                max_tokens=max_tokens
                            )

                        # Retrieve relevant documents via FAISS
                        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
                        source_docs = retriever.invoke(user_question)
                        context_text = "\n\n".join(doc.page_content for doc in source_docs)

                        # Prompt Template
                        prompt_template = """You are a helpful assistant answering questions about a YouTube video based strictly on its transcript.
Use the following pieces of context from the transcript to answer the question at the end.
If the context does not contain enough information to answer the question, state that the video does not provide enough details rather than making up facts.

Context:
{context}

Question: {question}

Helpful Answer:"""
                        prompt = PromptTemplate.from_template(prompt_template)

                        # LCEL Chain
                        chain = prompt | llm | StrOutputParser()
                        answer = chain.invoke({"context": context_text, "question": user_question})

                        # Save to history
                        st.session_state.qa_history.insert(0, {
                            "question": user_question,
                            "answer": answer,
                            "sources": [doc.page_content for doc in source_docs]
                        })

                    except groq.AuthenticationError:
                        st.error("🔑 **Invalid Groq API Key (401 Unauthorized)**: Please check your Groq key. You can create a free key at https://console.groq.com/keys")
                    except groq.RateLimitError:
                        st.error("⏳ **Groq Rate Limit Exceeded**: Please wait a few seconds and try again.")
                    except together.AuthenticationError:
                        st.error("🔑 **Invalid Together AI API Key (401 Unauthorized)**: Together AI rejected your API key. Please check your key at https://api.together.ai/settings/api-keys")
                    except together.RateLimitError:
                        st.error("⏳ **Together AI Rate Limit Exceeded**: Please wait a moment and try again.")
                    except Exception as e:
                        st.error(f"Error querying AI model: {e}")

        # Display Q&A History
        if st.session_state.qa_history:
            st.markdown("#### 📝 Conversation & Answers")
            for idx, item in enumerate(st.session_state.qa_history):
                with st.chat_message("user", avatar="👤"):
                    st.markdown(f"**{item['question']}**")

                with st.chat_message("assistant", avatar="💡"):
                    st.markdown(item["answer"])

                with st.expander(f"🔍 Relevant Transcript Excerpts (Sources #{idx+1})"):
                    for s_idx, src in enumerate(item["sources"]):
                        st.markdown(f"""
                        <div class="source-box">
                            <strong>Excerpt {s_idx+1}:</strong><br/>
                            {src}
                        </div>
                        """, unsafe_allow_html=True)
                st.markdown("---")

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")

else:
    st.info("👆 Enter a YouTube video URL above or click one of the sample buttons to get started!")
