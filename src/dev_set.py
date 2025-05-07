import os
import random

def create_dev_set(min_len=3, max_len=12, num_examples=100000,
                   input_file=None, input_out=None, answer_out=None):

    base_dir = os.path.dirname(os.path.dirname(__file__))  # go up from src/
    input_file = input_file or os.path.join(base_dir, "data", "en.txt")
    input_out = input_out or os.path.join(base_dir, "data", "dev_input.txt")
    answer_out = answer_out or os.path.join(base_dir, "data", "dev_answer.txt")

    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read().replace('\n', ' ').strip()

    max_possible_start = len(text) - max_len - 1
    if max_possible_start <= 0:
        raise ValueError("Error")

    inputs, answers = [], []
    used_starts = set()

    while len(inputs) < num_examples:
        start = random.randint(0, max_possible_start)
        if start in used_starts:
            continue
        used_starts.add(start)

        context_len = random.randint(min_len, max_len)
        context = text[start:start + context_len]
        next_char = text[start + context_len]

        if context.strip() and next_char.strip():
            inputs.append(context)
            answers.append(next_char)

    with open(input_out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(inputs))
    with open(answer_out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(answers))

if __name__ == "__main__":
    create_dev_set(min_len=0, max_len=50)