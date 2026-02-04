# RAG - Consulta Legislação Municipal

Este é um sistema distribuído para consulta de legislação municipal usando RAG (Retrieval-Augmented Generation). O sistema é composto por três componentes:

1. VectorStore API (Backend) - Gerencia o índice FAISS e busca chunks relevantes
2. Query API (Backend) - Processa consultas usando LangChain e OpenAI
3. Frontend - Interface de chat em Streamlit

## Estrutura do Projeto

```
RAG/
├── data/                    # PDFs da legislação
├── vectorstore_api/        
│   └── main.py             # API FastAPI para FAISS
├── query_api/
│   └── main.py             # API FastAPI para processamento RAG
├── frontend/
│   └── app.py              # Interface Streamlit
└── requirements.txt         # Dependências do projeto
```

## Instalação

1. Crie e ative um ambiente virtual:

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows/Git Bash
# ou
.\.venv\Scripts\Activate.ps1   # Windows/PowerShell
```

2. Instale as dependências:

```bash
pip install -r requirements.txt
```

3. Configure o arquivo `.env` na raiz do projeto:

```
OPENAI_API_KEY=sua-chave-aqui
```

## Como Executar

### Usando Docker (Recomendado)

1. Certifique-se de ter Docker e Docker Compose instalados
2. Configure o arquivo `.env` na raiz do projeto
3. Execute:

```bash
docker-compose up --build
```

Os serviços estarão disponíveis em:
- Frontend: http://localhost:8501
- VectorStore API: http://localhost:8001/docs
- Query API: http://localhost:8002/docs

### Execução Local

Alternativamente, você pode executar os componentes localmente em terminais separados:

1. Inicie o VectorStore API (Terminal 1):

```bash
cd vectorstore_api
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

2. Inicie o Query API (Terminal 2):

```bash
cd query_api
uvicorn main:app --host 0.0.0.0 --port 8002 --reload
```

3. Inicie o Frontend Streamlit (Terminal 3):

```bash
cd frontend
streamlit run app.py
```

O frontend estará disponível em `http://localhost:8501`.

## APIs

### VectorStore API
- GET `/search/?query=string`
  - Recebe: query (string)
  - Retorna: `{"chunks": ["texto1", "texto2", ...]}`

### Query API
- GET `/ask/?query=string`
  - Recebe: query (string)
  - Retorna: `{"answer": "resposta gerada"}`

## Observações
- Certifique-se de que os PDFs estejam em `RAG/data/`
- As APIs usam FastAPI e têm documentação automática em `/docs`
- O frontend mantém histórico de chat na sessão
- Ajuste os modelos e parâmetros em `query_api/main.py`

## Troubleshooting
- Se FAISS der erro no Windows, tente instalar via conda: `conda install -c conda-forge faiss-cpu`
- Verifique se todas as portas (8000, 8001, 8501) estão livres
- Em caso de erro de CORS, ajuste as configurações de middleware nas APIs