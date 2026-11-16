
import streamlit as st
import re
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import pipeline

st.set_page_config(
    page_title="RAG Document Chat",
    page_icon="📚"
)

st.title("📚 RAG Document Chat")
st.write("Upload your PDF and ask questions!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-small"
    )

    return embedding_model, generator

model, generator = load_models()

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

if uploaded_file:

    reader = PdfReader(uploaded_file)
    pdf_text = ""

    for page in reader.pages:
        pdf_text += page.extract_text() or ""

    chunks = re.split(r'(?<=[.!?])\s+', pdf_text)

    chunks = [
        chunk.strip()
        for chunk in chunks
        if chunk.strip()
    ]

    if not chunks:
        st.error("No readable text found in PDF.")

    else:

        client = chromadb.Client()

        collection = client.get_or_create_collection(
            "rag_documents"
        )

        embeddings = model.encode(chunks)

        collection.upsert(
            ids=[str(i) for i in range(len(chunks))],
            documents=chunks,
            embeddings=embeddings.tolist()
        )

        st.success("PDF processed successfully!")

        question = st.text_input(
            "Ask a question about your PDF"
        )

        if st.button("Get Answer"):

            if question.strip():

                query_embedding = model.encode(
                    [question]
                ).tolist()

                results = collection.query(
                    query_embeddings=query_embedding,
                    n_results=min(3, len(chunks))
                )

                retrieved_chunks = results["documents"][0]

                context = "\n".join(retrieved_chunks)

                st.subheader("🔍 Retrieved Document Text")
                st.write(context)

                # Generate answer
                prompt = f"""
                Answer the question using the context.
                Give a complete sentence.

                Context: {context}

                Question: {question}

                Answer:
                """

                response = generator(
                    prompt,
                    max_new_tokens=100,
                    do_sample=False
                )

                answer = response[0]["generated_text"].strip()

                # Simple fallback for incomplete answers
                if len(answer.split()) < 3:
                    matching_sentences = [
                        chunk for chunk in retrieved_chunks
                        if any(
                            word in chunk.lower()
                            for word in question.lower().split()
                            if len(word) > 3
                        )
                    ]

                    if matching_sentences:
                        answer = matching_sentences[0]

                st.subheader("🤖 AI Answer")
                st.write(answer)

                st.session_state.chat_history.append({
                    "question": question,
                    "answer": answer
                })

            else:
                st.warning("Please enter a question.")

        st.divider()
        st.subheader("💬 Chat History")

        for chat in st.session_state.chat_history:
            st.write("**You:**", chat["question"])
            st.write("**AI:**", chat["answer"])
            st.divider()

        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
