import os
import streamlit as st
from dotenv import load_dotenv
from supabase import create_client, Client
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from groq import Groq

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
load_dotenv()

@st.cache_resource
def load_infrastructure():
    supabase_client: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    groq_api_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return supabase_client, groq_api_client

@st.cache_resource
def load_models():
    embed_model = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")
    rerank_model = CrossEncoder('BAAI/bge-reranker-base')
    return embed_model, rerank_model

def get_relevant_documents(user_query):
    supabase, _ = load_infrastructure()
    embedding_model, reranker_model = load_models()

    query_embedding = embedding_model.embed_query(user_query)

    search_results = supabase.rpc(
        'match_support_articles_hybrid',
        {
            'query_text': user_query,
            'query_embedding': query_embedding,
            'match_count': 10
        }
    ).execute()

    documents = search_results.data
    if not documents:
        return []

    pairs = [[user_query, doc['content']] for doc in documents]
    scores = reranker_model.predict(pairs)
    
    for i, doc in enumerate(documents):
        doc['rerank_score'] = scores[i]
    
    ranked_docs = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
    return ranked_docs[:3]

def generate_streaming_response(user_query, top_docs):
    _, groq_client = load_infrastructure()
    
    if not top_docs:
        context_text = "No helpful articles found in the database."
    else:
        context_text = "\n\n".join([f"Article: {doc['title']}\n{doc['content']}" for doc in top_docs])
    
    system_prompt = (
        "You are a helpful customer support bot for TechGear. "
        "Use ONLY the following context to answer the user's question. "
        "If the answer is not in the context, clearly say you do not know.\n\n"
        f"Context:\n{context_text}"
    )

    chat_completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        stream=True
    )

    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content