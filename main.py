import argparse

from utils import download_url, extract_text, similarity_search, get_llm_answer


def main(url, query, device):
    filename = download_url(url)
    text_chunks = extract_text(filename)
    context = similarity_search(text_chunks, query)
    output = get_llm_answer(query, context, device=device)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--device", type=str, required=True)
    args = parser.parse_args()
    print(main(**vars(args)))
