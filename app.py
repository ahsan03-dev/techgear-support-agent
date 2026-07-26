import streamlit as st
from rag_chain import get_relevant_documents, generate_streaming_response, load_infrastructure, load_models

st.set_page_config(page_title="TechGear Support", layout="wide")

with st.spinner("Initializing RAG pipeline..."):
    load_infrastructure()
    load_models()

with st.sidebar:
    st.title("🛠️ Project Architecture")
    st.markdown(
        "This RAG system demonstrates real-time AI customer support."
    )
    st.subheader("Tech Stack")
    st.markdown(
        "- **Language:** Python\n"
        "- **Framework:** Streamlit\n"
        "- **Database:** Supabase (pgvector)\n"
        "- **AI Models:** Groq Llama 3.3, Qwen3-Embedding\n"
        "- **Pipeline:** Hybrid Search with Reranking"
    )
    
    st.divider()
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

st.title("TechGear Customer Support (RAG) 🤖")
st.markdown("Welcome! I am the TechGear AI assistant. How can I help you today?")

if "messages" not in st.session_state:
    st.session_state.messages = []

if len(st.session_state.messages) == 0:
    st.info(
        "**Try asking:**\n"
        "- How do I reset my wireless earbuds?\n"
        "- What is the return policy for laptops?\n"
        "- How do I file a warranty claim if my product stops working?"
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            top_docs = get_relevant_documents(prompt)
            
        stream = generate_streaming_response(prompt, top_docs)
        full_response = st.write_stream(stream)

        if top_docs:
            with st.expander("View Reference Articles"):
                for doc in top_docs:
                    st.markdown(f"**{doc['title']}** (Relevance Score: {doc['rerank_score']:.2f})")
                    st.caption(doc['content'])
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})