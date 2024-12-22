import json
import os
import argparse
from openai import OpenAI
import numpy as np
from tqdm import tqdm

def get_embeddings(input_file):
    # Create OpenAI client
    client = OpenAI()
    
    # Create output directory if it doesn't exist
    output_dir = "./data/original_output_embeddings"
    os.makedirs(output_dir, exist_ok=True)
    
    # Read input JSON file
    input_path = os.path.join("./data/original_output", input_file)
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    # Get embeddings for each prediction
    embeddings = []
    for item in tqdm(data, desc="Getting embeddings"):
        prediction = item['prediction']
        response = client.embeddings.create(
            input=prediction,
            model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding
        embeddings.append(embedding)
    
    # Convert to numpy array and save
    embeddings_array = np.array(embeddings)
    output_filename = f"{os.path.splitext(input_file)[0]}_embeddings.npy"
    output_path = os.path.join(output_dir, output_filename)
    np.save(output_path, embeddings_array)
    
    print(f"Embeddings saved to {output_path}")
    print(f"Array shape: {embeddings_array.shape}")

def main():
    parser = argparse.ArgumentParser(description='Get embeddings for JSON file predictions')
    parser.add_argument('--input_file', type=str, help='Name of the input JSON file')
    args = parser.parse_args()
    
    get_embeddings(args.input_file)

if __name__ == "__main__":
    main()
