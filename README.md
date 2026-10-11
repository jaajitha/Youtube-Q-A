# 🎬 YouTube Video Q&A Assistant

An AI-powered application built with **LangChain**, **Streamlit**, **Groq**, and **Together AI** that allows you to extract transcripts from any YouTube video and ask questions using Retrieval Augmented Generation (RAG).

---

## 🚀 Features

- **Multi-Provider Support**: Choose between **Groq** (free, ultra-low latency LLaMA 3.3) and **Together AI**.
- **Instant Transcript Retrieval**: Extracts transcripts and captions from YouTube videos (supports normal URLs, youtu.be, shorts, and embeds).
- **RAG Architecture**: Automatically splits transcripts, embeds them using HuggingFace (`all-MiniLM-L6-v2`), and indexes them locally with FAISS vector stores.
- **Interactive Q&A**: Ask any question about the video and receive sourced answers with cited excerpts.
- **Embedded Player**: Watch the video directly alongside the chat interface.

---

## 🛠️ How to Run

1. **Activate the Virtual Environment**:
   ```powershell
   .\venv\Scripts\activate
   ```

2. **Launch the Application**:
   ```powershell
   streamlit run main.py
   ```
   or:
   ```powershell
   .\venv\Scripts\python.exe -m streamlit run main.py
   ```

3. **Open in Browser**:
   Navigate to [http://localhost:8501](http://localhost:8501)

4. **Configure API Key in the Sidebar**:
   - **Groq (Recommended)**: Create a free key at [console.groq.com/keys](https://console.groq.com/keys) (takes 10 seconds, no credit card required).
   - **Together AI**: Or use an active key from [api.together.ai/settings/api-keys](https://api.together.ai/settings/api-keys).
   - Click **🔍 Test Key** to verify, and **💾 Save .env** to persist.
