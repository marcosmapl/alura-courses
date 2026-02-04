from typing import Dict, Any, List
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel

import httpx
import streamlit as st

models_list = ['gpt-5', 'gpt-5-mini', 'gpt-5-pro', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo']

class QueryResponse(BaseModel):
    answer: str

# Configure page
st.set_page_config(
    page_title="ALMa - Assistente de Legislação Tributária de Manaus",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Olá, Sou a ALMa, Assistente de Legislação Tributária de Manaus, no que posso ajudar?"
        }
    ]
if "show_chunks" not in st.session_state:
    st.session_state.show_chunks = False
    
if "k_chunks" not in st.session_state:
    st.session_state.k_chunks = 3
    
if "llm_model" not in st.session_state:
    st.session_state.llm_model = 'gpt-4o-mini'


# Retry-enabled API calls
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def search_vectorstore(prompt: str, k: int = 3) -> Dict[str, Any]:
    """Fetch relevant chunks from VectorStore API"""
    with httpx.Client(timeout=30.0) as client:
        response = client.get(
            "http://127.0.0.1:8001/search",
            params={"query": prompt, "k": k}
        )
        response.raise_for_status()
        return response.json()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def query_api(prompt: str, llm_model: str, chunks: List[Any]) -> Dict[str, Any]:
    """Send query and chunks to Query API.

    The Query API expects a list of plain text chunks (strings). This function
    extracts the `page_content` from chunk objects (dicts) and sends only the
    texts to avoid Pydantic validation errors (422).
    """
    # Prepare list of plain text chunks (only page_content)
    chunk_texts: List[str] = []
    for c in chunks or []:
        if isinstance(c, dict):
            chunk_texts.append(c.get("page_content", ""))
        else:
            chunk_texts.append(str(c))

    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            "http://127.0.0.1:8002/ask",
            json={"query": prompt, "llm_model": llm_model, "chunks": chunk_texts}
        )
        response.raise_for_status()
        return response.json()

# Page title
st.title("Assistente de Legislação Tributária de Manaus (ALMa) 📚")
st.markdown("---")

# Sidebar controls
with st.sidebar:
    st.title("Configurações")
    st.session_state.show_chunks = st.checkbox("Mostrar trechos relevantes", value=st.session_state.show_chunks)
    st.session_state.k_chunks = st.slider("Qtd. Referências", 1, 5, value=st.session_state.k_chunks)
    st.radio(
        "Modelo LLM", 
        key='llm_model',
        options=models_list
    )

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if st.session_state.show_chunks and "chunks" in message:
            with st.expander("Ver trechos relevantes"):
                for i, chunk in enumerate(message["chunks"], 1):
                            # chunk may be a dict with page_content and metadata
                            if isinstance(chunk, dict):
                                content = chunk.get("page_content", "")
                                metadata = chunk.get("metadata", {})
                            else:
                                content = str(chunk)
                                metadata = {}

                            st.markdown(f"**Trecho {i}:**")
                            st.html(f"<pre>{content}</pre>")
                            # display metadata below content
                            if metadata:
                                md_lines = []
                                for k, v in metadata.items():
                                    if k == 'url':
                                        md_lines.append(f'<span style="padding: 2px 5px; margin-right: 10px; background-color: #d6d6d6;"><strong>{k}</strong>:<a href="{v}">{v}</a></span>')
                                    else:
                                        md_lines.append(f'<span style="padding: 2px 5px; margin-right: 10px; background-color: #d6d6d6;"><strong>{k}</strong>: {v}</span>')
                                
                                st.html(''.join(md_lines))
                                
                            st.markdown("---")

# Chat input
if prompt := st.chat_input("Digite aqui sua pergunta sobre a legislação tributária de Manaus..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response with loading state
    with st.chat_message("assistant"):
        try:
            with st.status("Processando sua pergunta...", expanded=True) as status:
                st.write("Buscando trechos relevantes...")
                
                # Step 1: Call VectorStore API to get chunks
                vectorstore_response = search_vectorstore(prompt, st.session_state.k_chunks)
                st.session_state.chunks = vectorstore_response.get("chunks", [])
                
                status.write(f"Encontrados {len(st.session_state.chunks)} trechos relevantes.")
                status.write("Gerando resposta...")
                
                # Step 2: Call Query API with chunks
                query_response = query_api(prompt, st.session_state.llm_model, st.session_state.chunks)
                answer = query_response["answer"]
                
                # Display response
                st.markdown(answer)
                
                # Show chunks if enabled
                if st.session_state.show_chunks and st.session_state.chunks:
                    with st.expander("Ver trechos relevantes"):
                        for i, chunk in enumerate(st.session_state.chunks, 1):
                            # chunk may be a dict with page_content and metadata
                            if isinstance(chunk, dict):
                                content = chunk.get("page_content", "")
                                metadata = chunk.get("metadata", {})
                            else:
                                content = str(chunk)
                                metadata = {}

                            st.markdown(f"**Trecho {i}:**")
                            st.html(f"<pre>{content}</pre>")
                            # display metadata below content
                            if metadata:
                                md_lines = []
                                for k, v in metadata.items():
                                    if k == 'url':
                                        md_lines.append(f'<span style="padding: 2px 5px; margin-right: 10px; background-color: #d6d6d6;"><strong>{k}</strong>:<a href="{v}">{v}</a></span>')
                                    else:
                                        md_lines.append(f'<span style="padding: 2px 5px; margin-right: 10px; background-color: #d6d6d6;"><strong>{k}</strong>:{v}</span>')
                                
                                st.html(''.join(md_lines))
                                
                            st.markdown("---")
                
                status.update(label="Resposta gerada!", state="complete")
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "chunks": st.session_state.chunks
                })
                
        except httpx.HTTPError as e:
            error_msg = f"Erro ao chamar API: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"❌ {error_msg}"
            })
        except Exception as e:
            error_msg = f"Erro inesperado: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"❌ {error_msg}"
            })