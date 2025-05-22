#!/usr/bin/env python
import os
import random
import string
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datasets import load_dataset
import re
import unicodedata
import nltk
from nltk.corpus import stopwords
import subprocess
import torch
import torch.nn as nn

from ngram_model import NGramModel
from utils.hyperparameters import hyperparameter_tune
from char_lstm import CharLSTMWrapper

# Set UTF-8 encoding for standard output
import sys
sys.stdout.reconfigure(encoding='utf-8')

class MyModel(nn.Module):
    pass

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test', 'tune'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        # Ensure necessary NLTK data files are downloaded
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

        # Check if the mldd_dataset.csv file exists
        dataset_file = 'output/mldd_dataset.csv'
        if not os.path.isfile(dataset_file):
            print(f"{dataset_file} not found. Running script to generate it.")
            subprocess.run(['python', 'src/utils/combine_dataset_files.py'], check=True)

        # Load dataset
        dataset = load_dataset("csv", data_files=dataset_file, encoding="utf-8")

        # Split train/dev
        train_dataset, dev_dataset = dataset["train"].train_test_split(test_size=0.95).values()

        print("Preparing CharLSTM training...")
        # Extract conversations field as list of strings for training
        train_texts = [item['conversations'] for item in train_dataset]

        # Initialize model wrapper WITHOUT building vocab upfront
        model = CharLSTMWrapper()

        # Train model, it will handle normalization and vocab building internally
        print("Training CharLSTM...")
        model.train_model(train_texts, epochs=5)

        print("Saving CharLSTM model...")
        os.makedirs(args.work_dir, exist_ok=True)
        model.save(args.work_dir)
    elif args.mode == 'test':
        # Use with plain example/input.txt file
        # Example: python src/myprogram.py test --work_dir work --test_data example/input.txt --test_output pred.txt
        print('Loading CharLSTM model')
        from char_lstm import CharLSTMWrapper
        model = CharLSTMWrapper.load(args.work_dir)

        print('Loading test data from {}'.format(args.test_data))
        with open(args.test_data, encoding='utf-8') as f:
            test_lines = [line.strip() for line in f if line.strip()]

        print('Making predictions')
        predictions = []
        for line in test_lines:
            context = line[-20:]  # last 20 characters as context (adjust as needed)
            next_char = model.predict_next(context, top_k=1)[0]
            predictions.append(next_char)

        print('Writing predictions to {}'.format(args.test_output))
        with open(args.test_output, 'w', encoding='utf-8') as f:
            for p in predictions:
                f.write(p + '\n')

        assert len(predictions) == len(test_lines), f'Expected {len(test_lines)} predictions but got {len(predictions)}'

    elif args.mode == 'tune':
        print("Running hyperparameter tuning for NGramModel...")
        results = hyperparameter_tune()
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
