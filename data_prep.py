import argparse
import string
import sys
import warnings
from collections import Counter
from typing import List

import mlflow
import nltk
import numpy as np
import pandas as pd
from datasets import load_dataset
from loguru import logger
from sklearn.model_selection import train_test_split

from utils import seed_everything

nltk.download("punkt_tab")

logger.remove()
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
warnings.filterwarnings("ignore")


class WordDataset:
    def __init__(self, sentences, word2ind):
        self.data = sentences
        self.unk_id = word2ind["<unk>"]
        self.bos_id = word2ind["<bos>"]
        self.eos_id = word2ind["<eos>"]
        self.pad_id = word2ind["<pad>"]
        self.word2ind = word2ind

    def __getitem__(self, idx: int) -> List[int]:
        tokenized_sentence = []
        tokenized_sentence.append(self.bos_id)
        tokenized_sentence.extend(
            [
                self.word2ind.get(word, self.unk_id)
                for word in nltk.word_tokenize(self.data[idx])
            ]
        )
        tokenized_sentence.append(self.eos_id)
        return tokenized_sentence

    def __len__(self) -> int:
        return len(self.data)


def clean_data(word_threshold=32, vocab_size=40000) -> list:
    dataset = load_dataset("imdb")
    logger.info("Movie reviews dataset downloaded")
    sentences = []

    # sentence split + cleaning. working with senlences, which len < word_threshold
    for text in dataset["train"]["text"]:
        sentences.extend(
            [
                x.lower().translate(str.maketrans("", "", string.punctuation))
                for x in nltk.tokenize.sent_tokenize(text, language="english")
                if len(x.split()) < word_threshold
            ]
        )
    logger.info(f"Dataset split into sentences. Sentences in total: {len(sentences)}")

    words = Counter()

    # counting the frequency of each word
    for sent in sentences:
        for word in sent.split(" "):
            words[word] += 1

    # making the vocab
    vocab = set(["<unk>", "<bos>", "<eos>", "<pad>"])
    vocab_size = 40000

    for word, _ in sorted(words.items(), key=lambda pair: pair[1], reverse=True):
        if len(vocab) < (vocab_size + 4):
            vocab.add(word)

    assert "<unk>" in vocab
    assert "<bos>" in vocab
    assert "<eos>" in vocab
    assert "<pad>" in vocab
    assert len(vocab) == vocab_size + 4

    logger.info(f"Vocabulary created. Its size: {len(vocab)}")

    word2ind = {char: i for i, char in enumerate(vocab)}
    ind2word = {i: char for char, i in word2ind.items()}

    return sentences, word2ind, ind2word


if __name__ == "__main__":
    seed_everything(42)
    run = mlflow.active_run()
    if run:
        print(f"Active run_id: {run.info.run_id}")
        mlflow.end_run()
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8081")
    mlflow.set_experiment("Movie_Review_Gen")
    with mlflow.start_run():
        parser = argparse.ArgumentParser()
        parser.add_argument("--test-size", default=0.3, type=float)
        parser.add_argument("--word_threshold", default=37, type=int)
        parser.add_argument("--vocab_size", default=40000, type=int)
        parser.add_argument("--batch_size", default=128, type=int)

        test_size = parser.parse_args().test_size
        word_threshold = parser.parse_args().word_threshold
        vocab_size = parser.parse_args().vocab_size
        batch_size = parser.parse_args().batch_size

        logger.info("Data processing started")
        sents, word2ind, ind2word = clean_data(
            word_threshold=word_threshold, vocab_size=vocab_size
        )
        train_sentences, test_sentences = train_test_split(
            sents, test_size=test_size, random_state=42
        )

        train_dataset = WordDataset(train_sentences, word2ind)
        test_dataset = WordDataset(test_sentences, word2ind)

        mlflow.log_metric("train_data_size", len(train_dataset))
        mlflow.log_metric("test_data_size", len(test_dataset))

        # saving train ds in mlflow
        np.save("train.npy", np.asarray(train_dataset, dtype="object"))
        mlflow.log_artifact("data/train.npy")
        # transfer train_dataset into mlflow.data type to log it as ds in the experimet
        train_dataset = mlflow.data.from_numpy(
            np.asarray(train_dataset, dtype="object"), name="train"
        )
        mlflow.log_input(train_dataset, context="train_dataset")

        # saving test ds in mlflow (the same as with train ds)
        np.save("test.npy", np.asarray(test_dataset, dtype="object"))
        mlflow.log_artifact("data/test.npy")
        test_dataset = mlflow.data.from_numpy(
            np.asarray(test_dataset, dtype="object"), name="test"
        )
        mlflow.log_input(test_dataset, context="test_dataset")

        # saving metadata
        metadata = {
            "test_size": test_size,
            "word_threshold": word_threshold,
            "vocab_size": vocab_size,
            "sentence_amount": len(sents),
            "batch_size": batch_size,
            "pad_id": word2ind["<pad>"],
        }
        pd.DataFrame.from_dict(metadata.items()).to_csv("data/metadata.csv")
        mlflow.log_artifact("data/metadata.csv")

        logger.info("Data preprocessing finished")
