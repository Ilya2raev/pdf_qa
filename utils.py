import requests
import fitz

from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter


def download_url(url: str) -> None:
    """Downloads and saves URL file"""
    response = requests.get(url)
    filename = url.split('/')[-1]
    with open(filename, 'wb') as f:
        f.write(response.content)


def extract_text(filename: str) -> list:
    """Gets text from PDF as a string"""
    text = ''
    doc = fitz.open(filename)

    for page in doc:
        text += page.get_text()

    text_chunks = [text[i:i+100] for i in range(0, len(text), 100)]
    return text_chunks


def similarity_search(texts: list, query: str) -> str:
    """Performs similarity search between DB embeddings and query"""
    embeddings = GPT4AllEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.create_documents(texts)

    db = FAISS.from_documents(docs, embeddings)
    result = db.similarity_search(query)
    return result[0].page_content
