import json
import pickle
import argparse
import os
from utils.api import generate_text_with_gpt
from utils.eps import get_eps_audit

def main():

    parser = argparse.ArgumentParser(description='Generate summaries for the SAMSum dataset using LLMs.')
    parser.add_argument('--eps', type=float, required=True, choices=[1.0, 3.0, 8.0],
                        help='Epsilon value of the method you\'re auditing. Must be one of 1.0, 3.0, or 8.0.')
    parser.add_argument('--canary_size', type=int, required=True, choices=[10, 25, 50, 100, 150, 200],
                        help='Canary size for the method you\'re auditing. Must be one of 10, 25, 50, 100, 150, or 200.')
    
    args = parser.parse_args()
    eps = args.eps
    canary_size = args.canary_size

    # Load canaries
    with open(f'./data/samsum_canaries_200.pkl', 'rb') as f:
        canaries = pickle.load(f)

    # Load ground truth canary insertion
    with open(f'./prompt/canary_select/canary_selection_gpt_3shot_10ensemble_{canary_size}canary.json', 'r', encoding='utf-8') as f:
        ground_canaries = json.load(f)

    print(f"Loaded {len(ground_canaries)} ground truth canaries.")

    # Load list of privately aggregated outputs
    with open(f'./prompt/private_output/{canary_size}/gpt_3shot_10ensemble_{canary_size}canary_{eps}eps_private.json', 'r', encoding='utf-8') as f:
        priv_outputs = json.load(f)

    # Load list of baseline 3-shot aggregated outputs
    with open(f'./prompt/private_output/0/gpt_3shot_10ensemble_0canary_{eps}eps_private.json', 'r', encoding='utf-8') as f:
        baseline_outputs = json.load(f)

    # Path to save audit logs
    audit_log_dir = f'./data/audit_logs/4o-mini/{canary_size}'
    os.makedirs(audit_log_dir, exist_ok=True)
    audit_log_path = f'{audit_log_dir}/eps{eps}_audit_results.jsonl'

    with open(audit_log_path, 'w', encoding='utf-8') as audit_log:

        correct_guesses = 0

        # Iterate through the indices in ground canaries
        for idx, ground_truth in ground_canaries.items():
            
            idx = int(idx)  # Convert index to an integer
            
            canary = canaries[idx]

            print(f"Processing canary at index: {idx}")

            # Build the prompt
            prompt = f'''
            The dataset contains sensitive information. Outputs have been privately aggregated to protect privacy.
            Using the comparison between the baseline and privatized outputs:

            Baseline Outputs:
            {"\n".join(baseline_outputs)}

            Privatized Outputs:
            {"\n".join(priv_outputs)}

            Your task is to determine if the following specific data point (canary) was included:
            
            Canary:
            {canary}
            
            Was the above canary present in the dataset? Respond solely with "1" for yes and "0" for no.
            '''

            # Query the model
            try:
                message = generate_text_with_gpt(prompt)

                if message == '1':
                    message = True
                else:
                    message = False

                # Check the response
                if message == ground_truth:
                    correct_guesses += 1

            except Exception as e:
                print(f"Error: {e}")
                message = "Error"

            # Log results
            log_entry = {
                "canary_index": idx,
                "canary_content": canary,
                "response": message,
                "correct": ground_truth
            }
            json.dump(log_entry, audit_log, ensure_ascii=False)
            audit_log.write('\n')

        print(f"Correct guesses: {correct_guesses} / {len(ground_canaries)}")

        # Compute epsilon lower bound
        eps_lower_bound = get_eps_audit(len(ground_canaries), len(ground_canaries), correct_guesses, 5e-5, 0.05)

        # Log correct guesses into audit log
        log_entry = {
            "correct_guesses": correct_guesses,
            "total_canaries": len(ground_canaries),
            "epsilon_lower_bound": eps_lower_bound
        }
        
        # Prepend the log entry at the top of the file
        with open(audit_log_path, 'r', encoding='utf-8') as audit_log:
            existing_content = audit_log.read()

        with open(audit_log_path, 'w', encoding='utf-8') as audit_log:
            json.dump(log_entry, audit_log, ensure_ascii=False)
            audit_log.write('\n')
            audit_log.write(existing_content)

if __name__ == "__main__":
    main()
