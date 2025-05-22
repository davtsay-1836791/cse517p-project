import os
import re
import pickle
import math
from itertools import product
from collections import defaultdict, Counter
from utils.normalize import normalize_v2
from utils.constants import MAX_NGRAM_SIZE, MAX_UNIGRAM_FALLBACK_SIZE, MAX_TOP_K

class NGramModel:

    def __init__(self, max_grams=MAX_NGRAM_SIZE, lambdas=None):
        self.max_grams = max_grams
        self.models = {}
        self.context_totals = {}
        self.top_unigrams = []
        self.vocab = set()
        self.SPECIAL_TOKENS = {"<s>", "</s>"}
        self.lambdas = lambdas if lambdas else [0.05, 0.10, 0.15, 0.25, 0.45]  # Recommended for 5-gram char model

    @staticmethod
    def normalize_conversations(conversation_str_list):
        normalized = []
        pattern = re.compile(r"'value'\s*:\s*'(.*?)'", re.DOTALL)
        for row in conversation_str_list:
            matches = pattern.findall(row)
            for text in matches:
                norm = normalize_v2(text)
                normalized.append({"normalized": norm})
        return normalized

    @classmethod
    def load_training_data(cls, train_dataset=None):
        if train_dataset is None:
            return []
        try:
            train_conversations = train_dataset['conversations']
        except Exception as e:
            print(f"Error parsing conversations field: {e}")
            raise
        raw_normalized = cls.normalize_conversations(train_conversations)
        data = ''.join([f"<s>{conv['normalized']}</s>" for conv in raw_normalized])
        return data

    @classmethod
    def load_dev_data(cls, dev_dataset=None):
        if dev_dataset is None:
            return []
        try:
            dev_conversations = dev_dataset['conversations']
        except Exception as e:
            print(f"Error parsing conversations field: {e}")
            raise
        return cls.normalize_conversations(dev_conversations)

    @classmethod
    def load_test_data(cls, fname):
        with open(fname, encoding='utf-8') as f:
            for line in f:
                yield normalize_v2(line.strip())

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt', encoding='utf-8') as f:
            for p in preds:
                f.write(f'{p}\n')

    def run_train(self, data, work_dir):
        self.vocab = set(data)
        self.vocab.update(['<sos>', '<eos>'])

        for n in range(1, self.max_grams + 1):
            self.models[n] = defaultdict(Counter)
            for i in range(len(data) - n):
                context = data[i:i + n - 1] if n > 1 else ''
                next_char = data[i + n - 1]
                self.models[n][context][next_char] += 1

            self.context_totals[n] = {
                context: sum(counter.values())
                for context, counter in self.models[n].items()
            }

        all_chars = Counter(data)
        self.top_unigrams = [char for char, _ in all_chars.most_common(MAX_UNIGRAM_FALLBACK_SIZE)]
        print("Top unigrams: ", self.top_unigrams)

    def run_pred(self, data):
        preds = []
        for context in data:
            preds.append(self.predict_next_chars(context))
            print("Context: ", context)
            print("Predicted: ", preds[-1])
        return preds

    def predict_next_chars(self, context, top_k=MAX_TOP_K, k=0.0001):
        candidates = []
        char_scores = defaultdict(float)
        seen = set()
        max_token_len = max(len(tok) for tok in self.SPECIAL_TOKENS)
        V = len(self.vocab)
        scores = defaultdict(float)

        for n in range(1, self.max_grams + 1):
            weight = self.lambdas[n - 1]
            ctx = context[-(n - 1):] if n > 1 else ''
            model = self.models.get(n, {})
            dist = model.get(ctx, {})
            total = self.context_totals[n].get(ctx, 0)

            for char in self.vocab:
                prob = (dist.get(char, 0) + k) / (total + k * V)
                prob = min(prob, 0.99)  # cap to prevent overconfidence
                scores[char] += weight * prob

        sorted_chars = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for char, _ in sorted_chars:
            simulated = context + char
            if any(simulated.endswith(token) for token in self.SPECIAL_TOKENS):
                continue
            if char not in seen:
                candidates.append(char)
                seen.add(char)
            if len(candidates) >= top_k:
                break

        if not candidates:
            return ''.join(self.top_unigrams[:top_k])

        return ''.join(candidates[:top_k])


    def get_interpolated_char_probs(self, context):
        V = len(self.vocab)
        k = 0.001
        scores = defaultdict(float)

        active_weights = []
        probs_by_n = []

        for n in range(1, self.max_grams + 1):
            ctx = context[-(n - 1):] if n > 1 else ''
            model = self.models.get(n, {})
            dist = model.get(ctx, {})
            total = self.context_totals[n].get(ctx, 0)

            if total == 0:
                continue  # skip this n-gram if no match

            level_probs = {}
            for char in self.vocab:
                prob = (dist.get(char, 0) + k) / (total + k * V)
                prob = min(prob, 0.99)
                level_probs[char] = prob

            probs_by_n.append(level_probs)
            active_weights.append(self.lambdas[n - 1])

        if not active_weights or sum(active_weights) == 0:
            return {char: 1.0 / V for char in self.vocab}

        norm = sum(active_weights)
        normed_weights = [w / norm for w in active_weights]

        for weight, level_probs in zip(normed_weights, probs_by_n):
            for char, p in level_probs.items():
                scores[char] += weight * p

        return scores


    def perplexity(self, dev_data):
        total_log_prob = 0
        total_chars = 0

        for sentence in dev_data:
            context = ""
            for i in range(len(sentence)):
                true_char = sentence[i]
                probs = self.get_interpolated_char_probs(context)
                prob = max(probs.get(true_char, 1e-12), 1e-12)  # log-safe
                total_log_prob += -math.log(prob)
                context += true_char
                total_chars += 1

        return math.exp(total_log_prob / total_chars)

    def evaluate_lambdas(self, dev_data, lambdas):
        """
        Evaluate the given lambda weights and return perplexity.
        """
        assert abs(sum(lambdas) - 1.0) < 1e-6, "Lambdas must sum to 1.0"
        self.lambdas = lambdas
        print("Evaluating lambdas:", lambdas)
        perplexity = self.perplexity([d['normalized'] for d in dev_data])
        print(f"Perplexity for lambdas {lambdas}: {perplexity:.4f}")
        return perplexity

    def tune_lambdas(self, dev_data, step=0.1):
        best_perplexity = float('inf')
        best_lambdas = None
        search_space = [i * step for i in range(int(1 / step) + 1)]

        for combo in product(search_space, repeat=self.max_grams):
            if abs(sum(combo) - 1.0) > 1e-6:
                continue
            self.lambdas = list(combo)
            print("Working for combo: +", str(self.lambdas))
            perp = self.perplexity([d['normalized'] for d in dev_data])
            print("Perplexity for combo: +", str(perp))
            if perp < best_perplexity:
                best_perplexity = perp
                best_lambdas = list(combo)
                print(f"New best perplexity: {perp:.4f} with lambdas {combo}")

        self.lambdas = best_lambdas
        return best_lambdas, best_perplexity

    def save(self, work_dir):
        with open(os.path.join(work_dir, 'model.sda'), 'wb') as f:
            pickle.dump({
                "max_n": self.max_grams,
                "models": self.models,
                "context_totals": self.context_totals,
                "top_unigrams": self.top_unigrams,
                "vocab": list(self.vocab),
                "lambdas": self.lambdas
            }, f)

    @classmethod
    def load(cls, work_dir):
        with open(os.path.join(work_dir, 'model.sda'), 'rb') as f:
            obj = pickle.load(f)
        model = cls(max_grams=obj['max_n'], lambdas=obj.get('lambdas'))
        model.models = obj['models']
        model.context_totals = obj['context_totals']
        model.top_unigrams = obj['top_unigrams']
        model.vocab = set(obj['vocab'])
        return model
