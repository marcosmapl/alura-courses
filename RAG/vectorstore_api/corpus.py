from uuid import uuid4
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os

# Load and process all documents in the `data` folder
script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, "data")
print(f"Script directory: {script_dir}")
print(f"Data directory: {data_dir}")

if not os.path.isdir(data_dir):
    raise RuntimeError(f"Data directory not found: {data_dir}")

# Load documents and create corpus
file_names = [
    'decreto-3000-20150112.pdf',
    'decreto-3748-20170611.pdf',
    'decreto-4818-20000103.pdf',
    'decreto-5478-20230314.pdf',
    'decreto-5479-20230111.pdf',
    'decreto-5962-20240920.pdf',
    'lei-ordinaria-459-19981230.pdf',
    'lei-ordinaria-1628-20111230.pdf',
    'lei-ordinaria-2833-20211220.pdf'
]

docs_meta = [
    {'id': str(uuid4()), 'filename': 'decreto-3000-20150112.pdf', 'title': 'Decreto 3000/2015', 'date': '2015-01-12', 'type': 'decreto', 'url': 'http://leismunicipa.is/klomu'},
    {'id': str(uuid4()), 'filename': 'decreto-3748-20170611.pdf', 'title': 'Decreto 3748/2017', 'date': '2017-06-11', 'type': 'decreto', 'url': 'http://leismunicipa.is/kaovr'},
    {'id': str(uuid4()), 'filename': 'decreto-4818-20000103.pdf', 'title': 'Decreto 4818/2000', 'date': '2000-01-03', 'type': 'decreto', 'url': 'http://leismunicipa.is/rfhog'},
    {'id': str(uuid4()), 'filename': 'decreto-5478-20230314.pdf', 'title': 'Decreto 5478/2023', 'date': '2023-03-14', 'type': 'decreto', 'url': 'http://leismunicipa.is/0afpo'},
    {'id': str(uuid4()), 'filename': 'decreto-5479-20230111.pdf', 'title': 'Decreto 5479/2023', 'date': '2023-01-11', 'type': 'decreto', 'url': 'http://leismunicipa.is/0afpj'},
    {'id': str(uuid4()), 'filename': 'decreto-5962-20240920.pdf', 'title': 'Decreto 5962/2024', 'date': '2024-09-20', 'type': 'decreto', 'url': 'http://leismunicipa.is/1m3k3'},
    {'id': str(uuid4()), 'filename': 'lei-ordinaria-459-19981230.pdf', 'title': 'Lei Ordinária nº 459/1998', 'date': '1998-12-30', 'type': 'lei ordinária', 'url': 'http://leismunicipa.is/redoh'},
    {'id': str(uuid4()), 'filename': 'lei-ordinaria-1628-20111230.pdf', 'title': 'Lei Ordinária nº 1628/2011', 'date': '2011-12-30', 'type': 'lei ordinária', 'url': 'http://leismunicipa.is/gorhe'},
    {'id': str(uuid4()), 'filename': 'lei-ordinaria-2833-20211220.pdf', 'title': 'Lei Ordinária nº 2833/2021', 'date': '2021-12-20', 'type': 'lei ordinária', 'url': 'http://leismunicipa.is/uzqer'},
]

def build_corpus():
    corpus = []

    for file_name, metadata in zip(file_names, docs_meta):
        file_path = os.path.join(data_dir, file_name)
        try:
            docs = PyPDFLoader(file_path).load()
            print(f"Loaded {len(docs)} document from {file_name}.")
            # Split documents into chunks
            for idx_doc, doc in enumerate(docs):
                chunks = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=200
                ).split_documents([doc])
                print(f"\tDocument {idx_doc} splitted into {len(chunks)} chunks.")
                
                for idx_chunk, chunk in enumerate(chunks):
                    print('\t --- CHUNK ', idx_chunk, '---')
                    print(chunk)
                    # Merge metadata
                    chunk.metadata = {
                        **metadata,
                        'page': idx_doc,
                        'total_pages': len(docs),
                        'chunk_index': idx_chunk,
                    }
                    corpus.append(chunk)
        except Exception as e:
            print(f"Failed to load {file_name}: {e}")

    return corpus
