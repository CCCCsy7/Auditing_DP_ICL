import argparse
import os
from datasets import load_dataset, DatasetDict

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Dataset Preprocessing Script')
    parser.add_argument('--dataset', type=str, required=True, choices=['docvqa', 'samsum'],
                        help='Choose between "docvqa" and "samsum" dataset.')
    args = parser.parse_args()

    # Load the selected dataset
    if args.dataset == 'docvqa':
        # dataset = load_dataset('lmms-lab/DocVQA', 'DocVQA')
        pass
    elif args.dataset == 'samsum':
        dataset = load_dataset('samsum')

    # Randomly select 1000 samples from the training set
    train_dataset = dataset['train'].shuffle(seed=42).select(range(1000))

    # Randomly select 100 samples from the test set
    test_dataset = dataset['test'].shuffle(seed=42).select(range(100))

    # Create a new dataset dictionary
    new_dataset = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })

    # Save the new dataset to ./data directory
    save_path = os.path.join('./data', f'{args.dataset}_sampled')
    os.makedirs(save_path, exist_ok=True)
    new_dataset.save_to_disk(save_path)

if __name__ == '__main__':
    main()
