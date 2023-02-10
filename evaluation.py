from nltk import word_tokenize
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.translate import bleu_score
import numpy as np
from typing import List
import os
from pathlib import Path
from rouge import Rouge


nltk.download("wordnet")
nltk.download("punkt")

r = Rouge()


def rouge(references: List[str], hypothesis: str):
    return r.calc_score([hypothesis], references)


def meteor(references: List[str], hypothesis: str):
    references_tokens = [word_tokenize(reference) for reference in references]
    hypothesis_tokens = word_tokenize(hypothesis)
    return meteor_score(
        references_tokens,
        hypothesis_tokens,
    )


def bleu(references: List[str], hypothesis: str):
    splits_references = [reference.split() for reference in references]
    split_hypothesis = hypothesis.split()
    return bleu_score.sentence_bleu(splits_references, split_hypothesis)


def save_hypothesis(predction_path: str, hypothesis: List[str]):
    # save the hypothesis in a txt file
    file_path = Path(predction_path, "hypothesis.txt")
    with open(file_path, "w", encoding="utf-8") as file:
        for hypothesis_entry in hypothesis:
            file.write(hypothesis_entry)
            file.write("\n")


def load_references(references_path: List[str]):
    # load the references files given by the path and then split them line by line
    # result: [ [ref1, ref1, ...], [ref2, ref2, ...], ...]
    references = [open(path, "r").read().split("\n") for path in references_path]
    # tranform them to a list of format: [ [ref1, ref2, ...], [ref1, ref2, ...], ...]
    # this trick is explained in https://www.geeksforgeeks.org/python-unzip-a-list-of-tuples/
    references = list(zip(*references))

    return references


def load_hypothesis(prediction_path: str):

    # load the ids and predicted tokens results from the different gpus(rank)
    # and concatenate them

    # get all files in the prediction path
    all_files = os.listdir(prediction_path)

    if "hypothesis.txt" in all_files:
        hypothesis = (
            open(Path(prediction_path, "hypothesis.txt"), "r").read().split("\n")
        )
        return hypothesis


def eval_validation(references_path: List[str], prediction_path: str):
    references = load_references(references_path)
    hypothesis = load_hypothesis(prediction_path)

    save_hypothesis(prediction_path, hypothesis)

    meteor_results = []
    bleu_results = []
    rouge_results = []
    for hypo, reference in zip(hypothesis, references):
        bl = bleu(reference, hypo)
        me = meteor(reference, hypo)
        ro = rouge(reference, hypo)

        meteor_results.append(bl)
        bleu_results.append(me)
        rouge_results.append(ro)

    mean_bleu = np.mean(bleu_results)
    mean_meteor = np.mean(meteor_results)
    mean_rouge = np.mean(rouge_results)
    return mean_bleu, mean_meteor, mean_rouge
