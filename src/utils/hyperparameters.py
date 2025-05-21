import os
import itertools
from datasets import load_dataset
from ngram_model import NGramModel
import pickle

# Computes the character-level accuracy of predictions.
def evaluate_accuracy(model, dev_data):
    total = 0
    correct = 0
    for sample in dev_data:
        text = sample['normalized']
        for i in range(1, len(text)):
            context = text[:i]
            true_char = text[i]
            preds = model.predict_next_chars(context, top_k=3)
            if true_char in preds:
                correct += 1
            total += 1
    return correct / total if total > 0 else 0

# Loads the dataset, defines the grid of hyperparameters, trains and evaluates the model for each combination
def hyperparameter_tune(dataset_file='output/mldd_dataset.csv'):
    print("Loading dataset for tuning...")

    if not os.path.isfile(dataset_file):
        raise FileNotFoundError(f"{dataset_file} not found. Run 'train' mode first.")

    dataset = load_dataset("csv", data_files=dataset_file, encoding="utf-8")
    train_dataset, dev_dataset = dataset["train"].train_test_split(test_size=0.95).values()

    normalized_train = NGramModel.load_training_data(train_dataset)
    normalized_dev = NGramModel.load_dev_data(dev_dataset)
    normalized_dev = normalized_dev[:100]  # Only first 100 examples

    # Save once to pickle
    with open("output/normalized.pkl", "wb") as f:
        pickle.dump((normalized_train, normalized_dev), f)

    # Load later
    with open("output/normalized.pkl", "rb") as f:
        normalized_train, normalized_dev = pickle.load(f)

    param_grid = {
        "max_ngram_size": [2, 3, 4, 5, 6, 7, 8],
    }

    results = []

    print("Starting hyperparameter tuning...")
    for max_ngram in itertools.product(
        param_grid["max_ngram_size"],
    ):
        print(f"\nTraining with max_ngram={max_ngram}")

        model = NGramModel(max_grams=max_ngram)
        model.run_train(normalized_train, work_dir=None)
        acc = evaluate_accuracy(model, normalized_dev)
        results.append({
            "max_ngram": max_ngram,
            "accuracy": acc
        })
        print(f"Accuracy: {acc:.4f}")

    print("\nTop Configurations:")
    for res in sorted(results, key=lambda x: x["accuracy"], reverse=True)[:5]:
        print(res)

    return results
