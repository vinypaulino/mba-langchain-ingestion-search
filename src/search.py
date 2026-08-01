import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def _require_env(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Variável de ambiente obrigatória não definida: {name}")
    return value


def search_prompt():
    database_url = _require_env("DATABASE_URL")
    collection_name = _require_env("PG_VECTOR_COLLECTION_NAME")
    embedding_model = _require_env("GOOGLE_EMBEDDING_MODEL")
    chat_model = _require_env("GOOGLE_CHAT_MODEL")

    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
    store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=database_url,
        use_jsonb=True,
    )
    llm = ChatGoogleGenerativeAI(model=chat_model, temperature=0)

    def ask(question):
        results = store.similarity_search_with_score(question, k=10)
        contexto = "\n\n".join(doc.page_content for doc, _score in results)
        prompt = PROMPT_TEMPLATE.format(contexto=contexto, pergunta=question)
        return llm.invoke(prompt).content

    return ask
