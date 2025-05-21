from ngram_model import NGramModel
from utils.normalize import normalize_v2

def load_dev_dataset_from_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [normalize_v2(line.strip()) for line in f if line.strip()]

def main():
    # Load dev dataset
    dev_path = 'data/en_2.txt'  # update this path as needed
    dev_data = load_dev_dataset_from_txt(dev_path)

    # Initialize model (assume already trained and saved)
    model = NGramModel.load('work')  # path to saved model

    # Tune interpolation weights
    print("Tuning interpolation weights...")
    best_lambdas, best_perplexity = model.tune_lambdas([{ 'normalized': s } for s in dev_data], step=0.5)
    print(f"Best lambdas: {best_lambdas}")
    print(f"Best perplexity: {best_perplexity:.4f}")

    # Save updated model
    model.save('work')

if __name__ == "__main__":
    main()