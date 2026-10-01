import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import ollama


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="📄"
)


# ==========================================
# TITLE
# ==========================================

st.title("📄 Chat with Your PDF")

st.write(
    "Ask questions about your PDF using RAG + Ollama"
)


# ==========================================
# TEST MESSAGE
# ==========================================

st.info("Application is running successfully!")


# ==========================================
# PDF UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


# ==========================================
# PROCESS PDF
# ==========================================

if uploaded_file is not None:

    st.success("PDF uploaded successfully!")

    # Read PDF

    reader = PdfReader(uploaded_file)

    st.write(
        "Number of pages:",
        len(reader.pages)
    )

    # Extract text

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"


    # Check text

    if not text.strip():

        st.error(
            "No text could be extracted from this PDF."
        )

        st.stop()


    st.write(
        "Total characters:",
        len(text)
    )


    # ==========================================
    # CREATE CHUNKS
    # ==========================================

    chunk_size = 500

    chunks = []

    for i in range(
        0,
        len(text),
        chunk_size
    ):

        chunk = text[
            i:i + chunk_size
        ]

        if chunk.strip():

            chunks.append(chunk)


    st.write(
        "Number of chunks:",
        len(chunks)
    )


    # ==========================================
    # LOAD EMBEDDING MODEL
    # ==========================================

    with st.spinner(
        "Loading embedding model..."
    ):

        embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )


    st.success(
        "Embedding model loaded!"
    )


    # ==========================================
    # CREATE EMBEDDINGS
    # ==========================================

    with st.spinner(
        "Creating embeddings..."
    ):

        embeddings = embedding_model.encode(
            chunks
        )


    st.success(
        "Embeddings created!"
    )


    # ==========================================
    # CREATE CHROMADB
    # ==========================================

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = client.get_or_create_collection(
        name="documents"
    )


    # ==========================================
    # STORE DOCUMENT
    # ==========================================

    ids = []

    for i in range(len(chunks)):

        ids.append(
            "chunk_" + str(i)
        )


    # Remove old data

    try:

        old_data = collection.get()

        if old_data["ids"]:

            collection.delete(
                ids=old_data["ids"]
            )

    except Exception:

        pass


    collection.add(

        ids=ids,

        documents=chunks,

        embeddings=embeddings.tolist()

    )


    st.success(
        "PDF stored successfully in ChromaDB!"
    )


    st.divider()


    # ==========================================
    # QUESTION
    # ==========================================

    question = st.text_input(
        "Ask a question about the PDF"
    )


    # ==========================================
    # ANSWER
    # ==========================================

    if question:

        with st.spinner(
            "Searching document..."
        ):

            # Create question embedding

            question_embedding = (
                embedding_model.encode(
                    [question]
                )
            )


            # Search ChromaDB

            results = collection.query(

    query_embeddings=
    question_embedding.tolist(),

    n_results=len(chunks)

)


            # Get documents

            relevant_chunks = (
                results["documents"][0]
            )


            # Combine context

            context = "\n\n".join(
                relevant_chunks
            )


        # ==========================================
        # SHOW RETRIEVED DATA
        # ==========================================

        with st.expander(
            "View Retrieved Context"
        ):

            st.write(context)


        # ==========================================
        # CREATE PROMPT
        # ==========================================

        prompt = f"""

You are a helpful document assistant.

Answer the user's question using the information
available in the document context.

If the user asks for a summary, overview, or brief
description, summarize the important information
from the document context.

If the answer cannot be found in the document,
say:

"I could not find the answer in the document."

Do not make up information.

Document Context:

{context}

User Question:

{question}

Answer:

"""

        # ==========================================
        # OLLAMA
        # ==========================================

        with st.spinner(
            "Ollama is generating the answer..."
        ):

            response = ollama.chat(

                model="llama3.2",

                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]

            )


        # ==========================================
        # DISPLAY ANSWER
        # ==========================================

        answer = response["message"]["content"]

        st.subheader("🤖 Answer")

        st.write(answer)
