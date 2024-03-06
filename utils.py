import requests
import fitz
import faiss

from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer


def download_url(url: str) -> None:
    """Downloads and saves URL file"""
    response = requests.get(url)
    filename = url.split('/')[-1]
    with open(filename, 'wb') as f:
        f.write(response.content)


def extract_text(filename: str) -> str:
    """Gets text from PDF as a string"""
    text = ''
    doc = fitz.open(filename)

    for page in doc:
        text += page.get_text()

    text_chunks = [text[i:i+100] for i in range(0, len(text), 100)]
    return text_chunks


def create_embeddings(text: str, gpu: bool=False):
    model = SentenceTransformer(
        'sentence-transformers/sentence-t5-base', device='cuda' if gpu else 'cpu')

    embeddings = model.encode(text)
    return embeddings
