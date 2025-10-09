import argparse
from sentence_transformers import SentenceTransformer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='/resources/sentence_transformer')
    parser.add_argument('--model_name', type=str, default='bert-base-nli-stsb-mean-tokens')
    args = parser.parse_args()

    model = SentenceTransformer(args.model_name)
    model.save(args.save_dir)
