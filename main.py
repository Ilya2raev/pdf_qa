import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--link", type=str, required=True)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True)