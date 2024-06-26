import requests
import fitz
from gpt4all import GPT4All
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter


def download_url(url: str) -> None:
    """Downloads and saves URL file"""
    response = requests.get(url)
    filename = url.split("/")[-1]
    with open(filename, "wb") as f:
        f.write(response.content)
    return filename


def extract_text(filename: str) -> list:
    """Gets text from PDF as a string"""
    text = ""
    doc = fitz.open(filename)

    for page in doc:
        text += page.get_text()

    return text


def similarity_search(text: str, query: str) -> str:
    """Performs similarity search between DB embeddings and query"""
    embeddings = GPT4AllEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_text(text)

    db = FAISS.from_documents(docs, embeddings)
    result = db.similarity_search(query)
    return result[0].page_content


def get_llm_answer(
    query: str,
    context: str,
    model_name: str = "mistral-7b-openorca.gguf2.Q4_0.gguf",
    model_path: str = ".model/",
    device: str = "cpu",
) -> str:
    """Gets model answer based on system prompt with query
    and FAISS context as input variables"""
    model = GPT4All(model_name=model_name, model_path=model_path, device=device)

    output = model.generate(
        f"You are a virtual assistant. Your task is to generate answers to the query based on the context. Context: {context}. Answer the question: {query} ",
        max_tokens=1000,
    )

    return output if len(output) > 2 else "No data to answer the question"
