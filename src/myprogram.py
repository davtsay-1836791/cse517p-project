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

# Set UTF-8 encoding for standard output
import sys
sys.stdout.reconfigure(encoding='utf-8')

class MyModel(nn.Module):
    pass

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
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
            print(f"{dataset_file} not found. Running the necessary script to generate it.")
            # Run the script from src/utils/combine_dataset_files.py to combine the dataset splits
            script_path = 'src/utils/combine_dataset_files.py'
            subprocess.run(['python', script_path], check=True)
        else:
            print(f"{dataset_file} found. Proceeding with loading the dataset.")

        # Load the dataset
        dataset = load_dataset("csv", data_files="output/mldd_dataset.csv", encoding="utf-8")

        # Split the dataset into train and validation sets (90% train, 10% validation)
        train_dataset, dev_dataset = dataset["train"].train_test_split(test_size=0.9).values()

        # print part of the train dataset
        print("Train dataset sample:")

        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instantiating model')
        model = NGramModel()
        print('Normalizing training data...')
        normalized_train_data = NGramModel.load_training_data(train_dataset)  # Corrected method call

        print('Training')
        model.run_train(normalized_train_data, args.work_dir)  # Train with normalized train data
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        # print('Loading model')
        # model = NGramModel.load(args.work_dir)

        ### THE PRED STEP normalizes the data as context already
        # print('Normalizing test data...')
        # normalized_dev_data = NGramModel.load_dev_data(dev_dataset)  # Normalize dev data

        # print('Making predictions')
        # pred = model.run_pred(normalized_dev_data)
        # print('Writing predictions to {}'.format(args.test_output))
        # assert len(pred) == len(normalized_dev_data), 'Expected {} predictions but got {}'.format(len(normalized_dev_data), len(pred))
        # model.write_pred(pred, args.test_output)


        # use with plain example/input.txt file
        # python src/myprogram.py test --work_dir work --test_data example/input.txt --test_output pred.txt
        print('Loading model')
        model = NGramModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = NGramModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
