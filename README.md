# alura-courses
Repositório de projetos demonstrativos do curso 'LangChain e Python: Criando ferramentas com a OpenAI' da plataforma Alura.

Este repositório contém exemplos em Python que demonstram usos básicos da biblioteca LangChain com modelos de conversação (via OpenAI). O foco é um conjunto de pequenas aplicações que gerenciam roteiros, histórico de chat e cadeias de prompts simples para gerar recomendações de viagem.

## Sumário
- Visão geral
- Requisitos
- Instalação
- Configuração (.env)
- Exemplos implementados
- Como executar

## Visão geral
Há quatro exemplos no diretório raiz do projeto:

- `AgentRouting/` — Demonstra um roteador de agentes: um prompt decide qual agente (praia, aventura, gastronomia) deve responder à consulta e então invoca o agente apropriado.
- `ChatHistory/` — Demonstra como manter histórico de conversas em memória para conversas multi-turno com um modelo (usando `RunnableWithMessageHistory`).
- `SingleChain/` — Cadeia simples que gera uma sugestão de CNAE dada uma descrição resumida das atividades exercidas pela empresa usando parsers de saída (JSON e string).
- `RAG/` — Exemplo de Retrieval-Augmented Generation (RAG). O script carrega PDFs, divide em chunks, cria embeddings, indexa em FAISS e faz consultas com contexto recuperado para fundamentar as respostas.

## Requisitos
- Python 3.10+ (recomendado)
- `pip` para instalar dependências
- Uma chave de API da OpenAI com permissão para usar os modelos desejados

As dependências do projeto estão em `requirements.txt`.

## Instalação
1. Clone o repositório (se ainda não estiver local):

```bash
git clone <repo-url>
cd alura-courses
```

2. Crie e ative um ambiente virtual (exemplo para bash):

```bash
python -m venv .venv
source .venv/Scripts/activate   # no Windows usando bash.exe (Git Bash). Em PowerShell: .\.venv\Scripts\Activate.ps1
```

3. Instale as dependências:

```bash
pip install -r requirements.txt
```

## Configuração (.env)
Crie um arquivo `.env` na raiz do projeto com a sua chave de API da OpenAI:

```
OPENAI_API_KEY=sk-...
```

Certifique-se de nunca commitar sua chave para o repositório.

## Exemplos implementados

1) AgentRouting
- Local: `AgentRouting/main.py`
- O que faz: recebe uma consulta do usuário, usa um roteador (prompt) para classificar a consulta como `praia`, `aventura` ou `gastronomia`, e então encaminha para o agente especializado que responde.

2) ChatHistory
- Local: `ChatHistory/main.py`
- O que faz: demonstra como conservar o histórico de mensagens em memória (`InMemoryChatMessageHistory`) e reutilizá-lo em chamadas subsequentes ao modelo para conversas multi-turno.

3) SingleChain
- Local: `SingleChain/main.py`
- O que faz: exemplo de concatenação de prompts e parsers de saída (`JsonOutputParser`, `StrOutputParser`) para construir uma pequena cadeia que sugere códigos CNAEs para uma descrição de atividade exercida por uma empresa e também possíveis impostos.

4) RAG
- Local: `RAG/main.py`
- O que faz: demonstra uma pipeline de RAG:
	- Carrega documentos PDF em `RAG/data/` usando `PyPDFLoader`.
	- Divide os documentos em trechos (chunks) com `RecursiveCharacterTextSplitter`.
	- Gera embeddings (`OpenAIEmbeddings`) e indexa os trechos em um índice FAISS.
	- Para uma consulta, recupera os trechos mais relevantes e passa como contexto para o modelo gerar uma resposta fundamentada.
- Dados: coloque os PDFs na pasta `RAG/data/` (o script já referencia alguns nomes de exemplo). Ajuste os nomes se necessário.
- Dependências principais: `faiss` (ex.: `faiss-cpu`), `langchain_community` (PyPDFLoader), `langchain_text_splitters`, `langchain_openai` e suas dependências. Verifique `requirements.txt`.

## Como executar
Antes de executar, certifique-se de ativar o ambiente virtual e definir o `.env` com `OPENAI_API_KEY`.

- Executar o exemplo AgentRouting:

```bash
python AgentRouting/main.py
```

Será solicitado que você descreva o tipo de viagem; o programa roteará e imprimirá a resposta do agente selecionado.

- Executar o exemplo ChatHistory:

```bash
python ChatHistory/main.py
```

O script executa um pequeno fluxo de perguntas (preenchido no código) e imprime as respostas, demonstrando manutenção de histórico.

-- Executar o exemplo SingleChain:

```bash
python SingleChain/main.py
```

O script fará prompts sequenciais (pede uma descrição resumida da atividade exercida via input) e imprime a resposta composta pela cadeia.

- Executar o exemplo RAG:

```bash
python RAG/main.py
```

Observações para RAG:
- Certifique-se de que os PDFs referenciados em `RAG/main.py` existem em `RAG/data/`.
- FAISS pode demandar a instalação de `faiss-cpu` (ou `faiss-gpu` se quiser usar GPU). No Windows, instalar `faiss-cpu` via pip pode ou não estar disponível — em caso de problemas, considere usar um ambiente Linux/WSL ou instalar via conda:

```bash
# exemplo com conda
conda install -c conda-forge faiss-cpu
```

- Ajuste o `model`, `temperature` e parâmetros de recuperação (k) no arquivo `RAG/main.py` conforme sua necessidade e limites da conta OpenAI.

## Observações
- Ajuste o nome do modelo e a temperatura nos scripts conforme sua conta e limites de uso.
- Os exemplos usam APIs experimentais/abstrações via `langchain_openai` e `langchain_core`; verifique a compatibilidade das versões no `requirements.txt`.

## Licença
Consulte o arquivo `LICENSE` para detalhes.

