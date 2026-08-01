import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


def _require_env(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Variável de ambiente obrigatória não definida: {name}")
    return value


def ingest_pdf():
    pdf_path = _require_env("PDF_PATH")
    database_url = _require_env("DATABASE_URL")
    collection_name = _require_env("PG_VECTOR_COLLECTION_NAME")
    embedding_model = _require_env("GOOGLE_EMBEDDING_MODEL")

    docs = PyPDFLoader(pdf_path).load()

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    ).split_documents(docs)

    enriched_chunks = [
        Document(
            page_content=chunk.page_content,
            metadata={k: v for k, v in chunk.metadata.items() if v not in ("", None)},
        )
        for chunk in chunks
    ]
    ids = [f"doc-{i}" for i in range(len(enriched_chunks))]

    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)

    store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=database_url,
        use_jsonb=True,
    )
    store.add_documents(documents=enriched_chunks, ids=ids)


if __name__ == "__main__":
    ingest_pdf()
