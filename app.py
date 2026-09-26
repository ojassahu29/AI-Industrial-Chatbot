import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

# pyrefly: ignore [missing-import]
from rag import IndustrialRAG

st.set_page_config(
    page_title="AI Industrial Chatbot",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def get_rag():
    rag = IndustrialRAG()
    rag.load_index()
    return rag

st.title("AI-Powered Industrial Chatbot")
st.caption("Hugging Face LLM + semantic retrieval over industrial documents")

with st.sidebar:
    st.header("System")
    st.write("LLM: Qwen2.5-1.5B-Instruct")
    st.write("Embeddings: all-MiniLM-L6-v2")
    st.write("Retrieval: cosine similarity")
    st.write("Top-K context: 4")
    st.divider()
    st.info("Run `python src/build_index.py` after adding or changing documents.")

try:
    rag = get_rag()
except Exception as e:
    st.error(str(e))
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask an industrial/technical question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving documents and generating answer..."):
            answer, contexts = rag.answer(question)
        st.markdown(answer)

        with st.expander("Retrieved document evidence"):
            for i, c in enumerate(contexts, start=1):
                st.markdown(
                    f"**Source {i}: {c['source']} — {c['section']} "
                    f"(similarity {c['score']:.3f})**"
                )
                st.write(c["text"])
    st.session_state.messages.append({"role": "assistant", "content": answer})
