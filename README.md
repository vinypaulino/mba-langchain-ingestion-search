# Desafio MBA Engenharia de Software com IA - Full Cycle

Ingestão de um PDF em PostgreSQL + pgVector e busca semântica via CLI, usando LangChain e embeddings do Google Gemini.

## Pré-requisitos

- Python 3.11+
- Docker e Docker Compose
- Chave de API do Google AI (Gemini)

## Configuração

1. Crie e ative o ambiente virtual:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Copie `.env.example` para `.env` e preencha `GOOGLE_API_KEY`:

   ```bash
   cp .env.example .env
   ```

## Execução

1. Subir o banco de dados:

   ```bash
   docker compose up -d
   ```

2. Executar a ingestão do PDF (a partir da raiz do projeto):

   ```bash
   python src/ingest.py
   ```

3. Rodar o chat:

   ```bash
   python src/chat.py
   ```

4. Fazer perguntas sobre o conteúdo de `document.pdf`. Digite `sair` para encerrar.

## Estrutura

- `src/ingest.py` — lê o PDF, divide em chunks (1000 caracteres, overlap de 150), gera embeddings e grava no PGVector.
- `src/search.py` — vetoriza a pergunta, busca os 10 chunks mais relevantes (`k=10`) e monta o prompt para a LLM.
- `src/chat.py` — CLI de interação com o usuário.
