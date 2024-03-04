import requests
import fitz

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
    return text
    