import pickle
import json
import numpy as np
from datasets import load_dataset
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.api import generate_text_with_anthropic, generate_text_with_gpt, get_embeddings
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Parse command line arguments
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Generate canaries for the SAMSum dataset.')
    parser.add_argument('--llm', type=str, choices=['claude', 'gpt'], required=True,
                        help='Choice of LLM to use: claude or gpt.')
    parser.add_argument('--total_canaries', type=int, default=100,
                    help='Total number of canaries to generate.')
    return parser.parse_args()

# Get command line arguments
args = parse_args()
llm = args.llm
total_canaries = args.total_canaries

# Load the dataset
dataset = load_dataset("samsum")

# Use train data from the dataset
train_data = dataset["train"]

# Track used indices
used_indices = set()

# Pool size for initial sample selection
sample_pool_size = 2000

q_list_samples = []
while len(q_list_samples) < sample_pool_size:
    random_int = np.random.randint(0, len(train_data))
    if random_int not in used_indices:  # Ensure no reuse of indices
        q_list_samples.append(train_data[random_int])
        used_indices.add(random_int)

# Store generated canaries
q_list = []

# Test (dis)similarity using Sentence Transformer
model = SentenceTransformer("all-MiniLM-L6-v2")

# Track both dialogue and summary embeddings for similarity comparison
def get_combined_embedding(item):
    combined_text = f"Dialogue:\n{item['dialogue']}\nSummary: {item['summary']}"
    return model.encode(combined_text)

while len(q_list) < total_canaries:
    if not q_list_samples:
        print("No more q_list_samples available.")
        break
    
    # Pop the next q_list_sample
    sample = q_list_samples.pop()
        
    print(f"Generating canary {len(q_list) + 1}/{total_canaries}")
    
    while True:

        # Dynamic prompt based on attempt count
        text = f'''
            I want to generate both a dialogue and its summary that would be as memorable to a large language model as possible.
            The dialogue should not logically make sense, like "When does the sun rise on the west?"
            Using the following dialogue and summary as an abstraction:\n
            Dialogue: {sample['dialogue']}\nSummary: {sample['summary']}\n
            Generate a new dialogue and summary that are as memorable as possible.
            Write the result as a JSON string with "dialogue" and "summary" keys and no more other words.
            '''

        # Generate the canary
        if llm == 'gpt':
            message = generate_text_with_gpt(text)
        elif llm == 'claude':
            message = generate_text_with_anthropic(text)
        
        # Clean and parse the JSON response from the message
        try:
            # Remove any leading/trailing whitespace
            message = message.strip()
            
            # If the message is wrapped in quotes, remove them
            if message.startswith('"') and message.endswith('"'):
                message = message[1:-1]
            
            # First try to extract the JSON structure
            import re
            json_match = re.search(r'\{.*\}', message, re.DOTALL)
            if json_match:
                message = json_match.group()
            
            # Properly escape newlines in dialogue while preserving them
            # Look for the dialogue field and properly escape its newlines
            dialogue_pattern = r'"dialogue"\s*:\s*"(.*?)"(?=\s*[,}])'
            def escape_dialogue(match):
                dialogue_content = match.group(1)
                # Replace actual newlines with escaped newlines
                dialogue_content = dialogue_content.replace('\n', '\\n')
                return f'"dialogue": "{dialogue_content}"'
            
            message = re.sub(dialogue_pattern, escape_dialogue, message, flags=re.DOTALL)
            
            # Clean up other JSON formatting issues
            message = message.replace('"{', '{')  # Remove extra quotes around JSON
            message = message.replace('}"', '}')
            
            # Parse the cleaned JSON
            canary_item = json.loads(message)
            
            # Validate required keys
            if not all(key in canary_item for key in ['dialogue', 'summary']):
                raise ValueError("Missing required keys in generated content")
                
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Failed to parse JSON content: {message}")
            print(f"Error: {str(e)}")
            attempt_count += 1
            continue

        print('This is the canary item:')
        print(canary_item)
        print(canary_item['dialogue'])
        print('\n\n')

        # First canary has no similarity to compare to
        if len(q_list) == 0:
            q_list.append(canary_item)
            break

        # Calculate similarity using combined embeddings
        current_embedding = get_combined_embedding(canary_item)
        existing_embeddings = np.vstack([get_combined_embedding(item) for item in q_list])
        similarities = cosine_similarity(current_embedding.reshape(1, -1), existing_embeddings)
        
        max_similarity = np.max(similarities)
        if max_similarity < 0.6:
            q_list.append(canary_item)
            attempt_count = 0
            break
        else:
            print(f"Similarity too high ({max_similarity:.4f}), requesting a new canary...")
            if q_list_samples:
                sample = q_list_samples.pop()  # Pop a new sample

# Path to store the pickle file
pickle_path = f'./data/canaries/samsum_canaries_{total_canaries}.pkl'

# Save q_list as a pickle file
try:
    with open(pickle_path, 'wb') as f:
        pickle.dump(q_list, f)
    print(f"Successfully saved q_list to {pickle_path}")
except Exception as e:
    print(f"Failed to save q_list to {pickle_path}: {e}")