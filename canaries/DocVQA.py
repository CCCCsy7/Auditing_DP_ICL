import json
import pickle
import numpy as np
from utils.api import generate_text_with_anthropic
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

file_path = "INSERT PATH HERE"

# Load the NumPy array
data = np.load(file_path, allow_pickle=True)
q_list_samples = []

#Randomly select questions from dataset
for i in range(0, 150):
    random_int = np.random.randint(1, len(data))
    q_list_samples.append(data[random_int]['question'])

#Storing generated canaries
q_list = []

#Test (dis)similarity
model = SentenceTransformer("all-MiniLM-L6-v2")

import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

q_list = []

for i, q in enumerate(q_list_samples):
    attempt_count = 0  # Track the number of attempts to generate a unique canary
    
    while True:  # Loop to ensure similarity check passes
        # Dynamic prompt based on attempt count
        if attempt_count == 0:
            text = '''
            I want a sentence to be as memorable to a large language model as possible, a sentence that doesn\'t 
            logically make sense, for example: "When does the sun rise on the west?" Using the the same abstract form as:\n
            '''
        else:
            text = f'''
            The previous attempt was too similar to other questions. This time, generate a completely unique, unusual, and memorable sentence that a large language model would never forget. 
            Ensure it is significantly different from this question: "{canary_question}" and follows the same abstract form as:\n
            '''
        
        text += q
        text += '''\nGenerate a single question, stored in a python list, that is as memorable to a large language model as possible.
        '''
        
        # Call anthropic API to generate canary
        message = generate_text_with_anthropic(text)
        
        # Extract the JSON response from the message
        if '\n' not in message[0].text:
            json_content = message[0].text
        else:
            start_index = message[0].text.find("\n[")
            if start_index == -1:
                start_index = message[0].text.find("[")
            end_index = message[0].text.rfind("]\n") + 1
            json_content = message[0].text[start_index:end_index]
        try:
            parsed_json = json.loads(json_content)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON content: {json_content}")
            break  # If parsing fails, request a new canary question

        canary_question = parsed_json[0]  # Extract the canary question from the response

        # First question has no similarity to compare to
        if i == 0:
            q_list.append(canary_question)
            break  # Exit the while loop, as we don't need a similarity check for the first question
        
        # Calculate cosine similarity between the current canary and existing questions
        similarities = cosine_similarity(model.encode(canary_question).reshape(1, -1), model.encode(q_list))
        
        # Check if similarity is below 0.5
        max_similarity = np.max(similarities)
        if max_similarity < 0.5:
            q_list.append(canary_question)
            attempt_count = 0  # Reset the attempt count
            break  # Exit the while loop since the canary passes the similarity check
        else:
            print(f"Similarity too high ({max_similarity:.4f}), requesting a new canary...")
            attempt_count += 1  # Increase the attempt count

pickle_path = "INSERT PATH HERE"

# Save q_list as a pickle file
with open(pickle_path, 'wb') as f:
    pickle.dump(q_list, f)

#Can save list of questions to a file