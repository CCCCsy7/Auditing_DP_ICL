# api.py

import json
import anthropic
import openai
from openai import OpenAI
import os
import time
from tenacity import retry, stop_after_attempt, wait_exponential
import torch

@retry(stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_text_with_anthropic(text_input):
    """
    Call the Anthropic API to generate a response based on the input text.
    
    Parameters:
        text_input (str): The input text to send to the model.

    Returns:
        str: The content of the generated message.
    """
    client = anthropic.Anthropic()
    
    message = client.messages.create(
        model='claude-3-5-sonnet-20241022',
        max_tokens=300,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text_input
                    }
                ]
            }
        ]
    )
    return message.content[0].text

def generate_text_with_gpt(text_input, model="gpt-4o-mini"):
    """
    Call the OpenAI GPT API to generate a response based on the input text.
    
    Parameters:
        text_input (str): The input text to send to the model.

    Returns:
        str: The content of the generated message.
    """
    client = OpenAI()

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": text_input}]
            )
            return response.choices[0].message.content
        except openai.RateLimitError as e:
            sleep_time = (2 ** attempt)
            print(f"Rate limit hit. Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)
    return "Rate limit exceeded. Please try again later."


def get_embeddings(text_input):
    client = OpenAI()
    response = client.embeddings.create(
        input=text_input,
        model="text-embedding-3-small"
    )
    return torch.tensor(response.data[0].embedding)


if __name__ == "__main__":
    print(generate_text_with_gpt("What is the capital of France?"))