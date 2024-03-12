# PDF Question Answering with Retrieval Augmented Generation


The repository utilizes the concept of retrieval augmented generation (RAG). It uses FAISS to find information in the chunks from pre-downloaded PDF. It is then used as an input for optimized prompt template along with the query.

To make it work:

1. Clone the project
2. pip install -r requirements.txt
3. Download mistral-7b-openorca into model folder
4. python main.py --url --query --device

    - URL flag contains valid link to PDF file. It is downloaded in the work directory.
    - Query to the Large Language Model (mistral-7b-openorca) based on PDF data
    - Device to be used ('cpu' or 'gpu').