# app.py
import streamlit as st
import tempfile
from rag import build_rag_chain, ask_question


st.set_page_config(page_title="PDF Chat", page_icon="📄")
st.title("PDF Chat")

#upload pdf
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:

    # Build chain once, store in session
    if "chain" not in st.session_state:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with st.spinner("Indexing PDF..."):
            st.session_state.chain = build_rag_chain(tmp_path)
            st.session_state.messages = []  # reset chat on new upload

  #for saving chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

#input for asking question
    if question := st.chat_input("Ask anything from the PDF..."):
        st.chat_message("user").write(question)
        st.session_state.messages.append({"role": "user", "content": question})

        with st.spinner("Thinking..."):
            result = ask_question(st.session_state.chain, question)

        answer = f"{result['answer']}\n\n *Pages: {result['source_pages']}*"
        st.chat_message("assistant").write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    # Show this when no PDF is uploaded yet
    st.info("Upload a PDF to get started")