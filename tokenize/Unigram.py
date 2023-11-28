from transformers import AutoTokenizer
from collections import defaultdict
from math import log
import copy

corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]


# init pre tokenize function
xlnet_tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
pre_tokenize_function = xlnet_tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str

# pre tokenize
pre_tokenized_corpus = [pre_tokenize_function(text) for text in corpus]


word2count = defaultdict(int)
for split_text in pre_tokenized_corpus:
    for word, _ in split_text:
        word2count[word] += 1


char2count = defaultdict(int)
sub_word2count = defaultdict(int)
for word, count in word2count.items():
    for i in range(len(word)):
        char2count[word[i]] += count
        for j in range(i + 2, len(word) + 1):
            sub_word2count[word[i:j]] += count
sorted_sub_words = sorted(sub_word2count.items(), key=lambda x: x[1], reverse=True)
# init a large vocab with 300
tokens = list(char2count.items()) + sorted_sub_words[: 300 - len(char2count)]

token2count = {token: count for token, count in tokens}
total_count = sum([count for token, count in token2count.items()])
model = {token: -log(count / total_count) for token, count in token2count.items()}



def _encode_word(word, model):
    best_segmentations = [{"start": 0, "score": 1}] + [{"start": None, "score": None} for _ in range(len(word))]
    for start_idx in range(len(word)):
        # This should be properly filled by the previous steps of the loop
        best_score_at_start = best_segmentations[start_idx]["score"]
        for end_idx in range(start_idx + 1, len(word) + 1):
            token = word[start_idx:end_idx]
            if token in model and best_score_at_start is not None:
                score = model[token] + best_score_at_start
                # If we have found a better segmentation (lower score) ending at end_idx
                if (
                        best_segmentations[end_idx]["score"] is None
                        or best_segmentations[end_idx]["score"] > score
                ):
                    best_segmentations[end_idx] = {"start": start_idx, "score": score}
    segmentation = best_segmentations[-1]
    if segmentation["score"] is None:
        # We did not find a tokenization of the word -> unknown
        return ["<unk>"], None
    score = segmentation["score"]
    start = segmentation["start"]
    end = len(word)
    tokens = []
    while start != 0:
        tokens.insert(0, word[start:end])
        next_start = best_segmentations[start]["start"]
        end = start
        start = next_start
    tokens.insert(0, word[start:end])
    return tokens, score


def _compute_loss(self, model, word2count):
    loss = 0
    for word, freq in word2count.items():
        _, word_loss = self._encode_word(word, model)
        loss += freq * word_loss
    return loss

def _compute_scores(self, model, word2count):
    scores = {}
    model_loss = self._compute_loss(model, word2count)
    for token, score in model.items():
        # We always keep tokens of length 1
        if len(token) == 1:
            continue
        model_without_token = copy.deepcopy(model)
        _ = model_without_token.pop(token)
        scores[token] = self._compute_loss(model_without_token, word2count) - model_loss
    return scores

scores = _compute_scores(model, word2count)

sorted_scores = sorted(scores.items(), key=lambda x: x[1])
# Remove percent_to_remove tokens with the lowest scores.
for i in range(int(len(model) * 0.1)):
    _ = token2count.pop(sorted_scores[i][0])

vocab_size = 50
percent_to_remove = 0.1
while len(model) > vocab_size:
    scores = _compute_scores(model, word2count)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1])
    # Remove percent_to_remove tokens with the lowest scores.
    for i in range(int(len(model) * percent_to_remove)):
        _ = token2count.pop(sorted_scores[i][0])
    total_count = sum([freq for token, freq in token2count.items()])
    model = {token: -log(count / total_count) for token, count in token2count.items()}


# 推理
def tokenize(self, text):
    words = [word for word, _ in self.pre_tokenize_str(text)]
    encoded_words = [self._encode_word(word, self.model)[0] for word in words]
    return sum(encoded_words, [])

tokenize("This is the Hugging Face course!")