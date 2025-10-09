import json
import argparse
import prior


def load_procthor_data(args):
    dataset = prior.load_dataset("procthor-10k")
    train_data = dataset['train']

    # File path where you want to save the JSON Lines file
    file_path = f'{args.save_dir}/data.jsonl'

    # Writing data to a JSONL file
    with open(file_path, 'w') as file:
        for entry in train_data:
            # Assuming 'entry' is a dictionary-like object; adjust if necessary
            json_string = json.dumps(entry)
            file.write(json_string + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='/resources/procthor-data')
    args = parser.parse_args()

    # Load the procthor-10k data and save it
    load_procthor_data(args)
