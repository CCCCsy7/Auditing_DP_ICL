import json
import os
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from prv_accountant.dpsgd import find_noise_multiplier
import evaluate
from datasets import load_from_disk
import evaluate


def aggregate_and_evaluate(
    candidates_json, target_embeddings_f, ensemble, private, epsilon
):
    # Create output directories if they don't exist
    private_output_dir = "./output/private_output"
    results_dir = "./output/results"
    for dir_path in [private_output_dir, results_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Load candidates embeddings and predictions
    # Actually, the candidates_json is the predictions of the 0shot-10ensemble settings.
    candidates_embeddings_path = f"./data/original_output_embeddings/{os.path.splitext(candidates_json)[0]}_embeddings.npy"
    candidates_embeddings = np.load(candidates_embeddings_path)
    print(f"candidates_embeddings shape: {candidates_embeddings.shape}")

    candidates_predictions = []
    candidates_json_full = os.path.join("./data/original_output", candidates_json)
    with open(candidates_json_full, "r", encoding="utf-8") as f:
        for line in f:
            candidates_predictions.append(json.loads(line)["prediction"])
    print(f"len(candidates_predictions): {len(candidates_predictions)}")

    # Load target embeddings
    target_embeddings_path = f"./data/original_output_embeddings/{os.path.splitext(target_embeddings_f)[0]}.npy"
    target_embeddings = np.load(target_embeddings_path)
    print(f"target_embeddings shape: {target_embeddings.shape}")

    # Load ground truth
    dataset = load_from_disk("./data/samsum_sampled")
    ground_truth = dataset["test"]["summary"]  # type: list
    test_size = len(ground_truth)
    print(f"test_size: {test_size}")

    final_pred = []
    rouge1, rouge2, rougeL, rougeLsum = [], [], [], []
    metric = evaluate.load("rouge")

    if not private:
        indexs = []
        for i in range(test_size):
            # get mean embedding
            curr_emb = target_embeddings[i * ensemble : (i + 1) * ensemble] # shape: (ensemble, vector_dim)
            mean_emb = np.mean(curr_emb, axis=0, keepdims=True) # shape: (1, vector_dim)
            dist = cosine_similarity(
                mean_emb, candidates_embeddings[i * ensemble : (i + 1) * ensemble]
            ) # shape: (1, ensemble)
            indexs.append(np.argmax(dist) + i * ensemble)
        # get predictions from json fil
        for index in indexs:
            final_pred.append(candidates_predictions[index])
        scores = metric.compute(predictions=final_pred, references=ground_truth)
        rouge1.append(scores["rouge1"])
        rouge2.append(scores["rouge2"])
        rougeL.append(scores["rougeL"])
        rougeLsum.append(scores["rougeLsum"])

    elif private:
        # Calculate noise multiplier
        sampling_probability = ensemble * 3 / 1000  # Based on your example
        noise_multiplier = find_noise_multiplier(
            sampling_probability=sampling_probability,
            num_steps=ensemble,
            target_epsilon=epsilon,
            target_delta=5e-5,
            eps_error=0.01,
            mu_max=100, # experimental value
        )

        for _ in range(5):
            indexs = []
            for i in range(test_size):
                # get mean embedding
                curr_emb = target_embeddings[(i * ensemble) : (i * ensemble + ensemble)]
                curr_emb_mean = np.mean(curr_emb, axis=0, keepdims=True)
                curr_emb_mean += np.random.normal(
                    loc=0, 
                    scale=noise_multiplier/ensemble,  # Note: need to divide by ensemble here
                    size=curr_emb_mean.shape
                )
                dist = cosine_similarity(
                    curr_emb_mean,
                    candidates_embeddings[(i * ensemble) : (i * ensemble + ensemble)],
                )
                indexs.append(np.argmax(dist) + i * ensemble)

            final_pred = []
            for index in indexs:
                final_pred.append(candidates_predictions[index])

            score = metric.compute(predictions=final_pred, references=ground_truth)

            rouge1.append(score["rouge1"])
            rouge2.append(score["rouge2"])
            rougeL.append(score["rougeL"])
            rougeLsum.append(score["rougeLsum"])

    # Save private predictions
    output_filename = f"{os.path.splitext(target_embeddings_f)[0].replace('_embeddings', '')}_{epsilon}eps_private.json"
    output_path = os.path.join(private_output_dir, output_filename)
    print(f"output_path: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_pred, f)

    final_scores = {
        "rouge1": {
            "mean": np.mean(np.array(rouge1)),
            "std": np.std(np.array(rouge1)) if private else 0.0
        },
        "rouge2": {
            "mean": np.mean(np.array(rouge2)),
            "std": np.std(np.array(rouge2)) if private else 0.0
        },
        "rougeL": {
            "mean": np.mean(np.array(rougeL)),
            "std": np.std(np.array(rougeL)) if private else 0.0
        },
        "rougeLsum": {
            "mean": np.mean(np.array(rougeLsum)),
            "std": np.std(np.array(rougeLsum)) if private else 0.0
        }
    }

    print("eps :", epsilon)
    print(
        "rouge1's mean: ",
        final_scores["rouge1"]["mean"],
        "rouge1's std: ",
        final_scores["rouge1"]["std"],
    )
    print(
        "rouge2's mean: ",
        final_scores["rouge2"]["mean"],
        "rouge2's std: ",
        final_scores["rouge2"]["std"],
    )
    print(
        "rougeL's mean: ",
        final_scores["rougeL"]["mean"],
        "rougeL's std: ",
        final_scores["rougeL"]["std"],
    )
    print(
        "rougeLsum's mean: ",
        final_scores["rougeLsum"]["mean"],
        "rougeLsum's std: ",
        final_scores["rougeLsum"]["std"],
    )

    # Save metrics
    metrics_filename = f"{os.path.splitext(target_embeddings_f)[0].replace('_embeddings', '')}_{epsilon}eps_metrics.json"
    metrics_path = os.path.join(results_dir, metrics_filename)
    with open(metrics_path, "w") as f:
        json.dump(final_scores, f)

    print(f"Private predictions saved to: {output_path}")
    print(f"Metrics saved to: {metrics_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate embeddings and evaluate with privacy options"
    )
    parser.add_argument(
        "--candidates_json",
        type=str,
        required=True,
        help="This json file (only the name) contains the candidates for embedding2text. It is actually the predictions of the 0shot-10ensemble settings.",
    )
    parser.add_argument(
        "--target_embeddings_f",
        type=str,
        required=True,
        help="The target embeddings file (only the name) is the embeddings of the ones that need to be aggregated.",
    )
    parser.add_argument(
        "--ensemble",
        type=int,
        required=True,
        help="The number of ensemble predictions for each test example.",
    )
    parser.add_argument(
        "--private",
        type=bool,
        required=True,
        help="Whether to use private aggregation.",
    )
    parser.add_argument("--eps", type=float, default=1.0)
    args = parser.parse_args()

    aggregate_and_evaluate(
        args.candidates_json,
        args.target_embeddings_f,
        args.ensemble,
        args.private,
        args.eps,
    )


if __name__ == "__main__":
    main()
