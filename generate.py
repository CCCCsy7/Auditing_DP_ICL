import argparse
import os
import json
import random
import pickle
from tqdm import tqdm

from utils.api import generate_text_with_anthropic, generate_text_with_gpt
from datasets import load_from_disk

def main():
    parser = argparse.ArgumentParser(description='Generate summaries for the SAMSum dataset using LLMs.')
    parser.add_argument('--exampler', type=int, required=True,
                        help='Number of exemplars (shots). If 0, skip Poisson subsampling.')
    parser.add_argument('--ensemble', type=int, default=10,
                        help='Number of ensembles.')
    parser.add_argument('--llm', type=str, choices=['claude', 'gpt'], required=True,
                        help='Choice of LLM to use: claude or gpt.')
    parser.add_argument('--canary_size', type=int, default=0,
                        help='Number of canaries to include in training data. If 0, no canaries are added.')
    parser.add_argument('--test_size', type=int, default=0,
                        help='Number of test samples to use. If 0, use all test data.')
    args = parser.parse_args()

    exampler = args.exampler
    ensemble = args.ensemble
    llm = args.llm
    canary_size = args.canary_size
    test_size = args.test_size

    # Load the pre-sampled SAMSum dataset using load_from_disk
    dataset = load_from_disk('./data/samsum_sampled')
    train_data = dataset['train']
    test_data = dataset['test']
    if test_size > 0:
        test_data = test_data.select(range(test_size))

    # If canary_size > 0, load canaries and replace items in training data
    if canary_size > 0:
        # Load canaries from the pickle file
        with open('./data/canaries/samsum_canaries_200.pkl', 'rb') as f:
            canaries_pool = pickle.load(f)
        
        # Check that canary_size does not exceed available canaries
        if canary_size > len(canaries_pool):
            raise ValueError(f"canary_size ({canary_size}) exceeds the number of available canaries ({len(canaries_pool)}).")
        
        # Randomly select canary_size canaries from the pool
        canaries_indices = random.sample(range(len(canaries_pool)), canary_size)
        canary_selection = {}
        selected_canaries = []
        
        # For each selected canary, decide whether to use it with 0.5 probability
        for original_idx in canaries_indices:
            is_selected = random.random() < 0.5
            canary_selection[original_idx] = is_selected
            if is_selected:
                selected_canaries.append(canaries_pool[original_idx])
        
        # Save the selection record
        output_dir = './data/selected_canaries'
        os.makedirs(output_dir, exist_ok=True)
        selection_filename = f'canary_selection_{llm}_{exampler}shot_{ensemble}ensemble_{canary_size}canary.json'
        selection_filepath = os.path.join(output_dir, selection_filename)
        with open(selection_filepath, 'w') as f:
            json.dump(canary_selection, f)

        # If any canaries were selected, insert them into training data
        if selected_canaries:
            # Randomly select indices in the training data
            train_indices = random.sample(range(len(train_data)), len(selected_canaries))
            
            # Create a mapping from index to canary
            index_to_canary = dict(zip(train_indices, selected_canaries))
            
            # Use map function to replace training data at specified indices
            def replace_with_canary(example, idx):
                if idx in index_to_canary:
                    canary_data = index_to_canary[idx]
                    return {
                        key: canary_data[key] if key in canary_data else example[key]
                        for key in example.keys()
                    }
                return example

        train_data = train_data.map(replace_with_canary, with_indices=True)

    # Ensure the output directory exists
    output_dir = './data/original_output'
    os.makedirs(output_dir, exist_ok=True)

    original_outputs = []

    for test_item in tqdm(test_data, desc="Processing test items", position=0):
        for ensemble_idx in tqdm(range(ensemble), desc="Running ensemble", position=1, leave=False):
            if exampler > 0:
                exemplars = poisson_subsample(train_data, exampler)
            else:
                exemplars = []
            
            # Build the prompt
            prompt = build_prompt(exemplars, test_item)

            # Query the LLM
            if llm == 'claude':
                prediction = generate_text_with_anthropic(prompt)
            elif llm == 'gpt':
                prediction = generate_text_with_gpt(prompt)
            else:
                raise ValueError('Invalid LLM choice')

            # Save the result
            original_output_item = {
                'origin_prompt': prompt,
                'output': prompt + prediction,
                'prediction': prediction
            }
            original_outputs.append(original_output_item)

    assert len(original_outputs) == len(test_data) * ensemble

    # Save outputs to a JSON file
    output_filename = f'{llm}_{exampler}shot_{ensemble}ensemble_{canary_size}canary.jsonl'
    output_filepath = os.path.join(output_dir, output_filename)
    with open(output_filepath, 'w', encoding='utf-8') as f:
        for item in original_outputs:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

def build_prompt(exemplars, test_item):
    """
    Build the prompt by combining exemplars and the test item.

    Parameters:
        exemplars (list): A list of exemplars.
        test_item (dict): A single test data item.

    Returns:
        str: The constructed prompt.
    """
    prompt = ""

    # Add exemplars to the prompt
    for exemplar in exemplars:
        dialogue = exemplar['dialogue']
        summary = exemplar['summary']
        prompt += f"Dialogue:\n{dialogue}\nSummary:\n{summary}\n\n"

    # Add the test dialogue
    prompt += f"Dialogue:\n{test_item['dialogue']}\nSummary:\n"

    return prompt

def poisson_subsample(data, expected_count):
    """
    Perform Poisson subsampling on the dataset.

    Parameters:
        data (Dataset): The dataset to sample from.
        expected_count (int): The expected number of samples.

    Returns:
        list: A list of sampled data items.
    """
    sampled_data = []
    # Calculate the sampling probability
    sampling_prob = expected_count / len(data)
    for item in data:
        if random.random() < sampling_prob:
            sampled_data.append(item)
    return sampled_data

if __name__ == "__main__":
    main()
