import itertools
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import pipeline


def highest_scoring(scores: list[dict]) -> str:
    """Extracts highest scoring label from raw predicition scores."""
    values = [entry["score"] for entry in scores]
    highest_idx = np.argmax(values)
    return scores[highest_idx]["label"]


def batched(iterable, n):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := list(itertools.islice(it, n)):
        yield batch


def process_batched(
    classifier, sentences: Iterable[str], batch_size: int = 128
) -> list[list[dict]]:
    """Runs emotion classification in batches with a progress bar."""
    batches = list(batched(sentences, batch_size))
    res = []
    for batch in tqdm(batches, desc="Processing sentences in batches."):
        res.extend(classifier(batch))
    return res


def main():
    data = pd.read_csv("dat/Game_of_Thrones_Script.csv")
    # We don't want NA sentences
    data = data.dropna(subset=["Sentence"])
    classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True,
    )
    emotion_scores = process_batched(classifier, data["Sentence"])
    res_dir = Path("results")
    res_dir.mkdir(exist_ok=True)
    results = data.assign(scores=emotion_scores)
    results["emotion_label"] = results["scores"].map(highest_scoring)
    results = results.drop(columns=["Sentence"])
    # Saving as JSON instead of CSV so that the scores can be kept in their current format
    results.to_json(res_dir.joinpath("emotions.jsonl"), orient="records", lines=True)


if __name__ == "__main__":
    main()
