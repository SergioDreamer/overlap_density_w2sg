import functools
import hashlib
from collections import Counter
from dataclasses import dataclass
from random import Random
from typing import Any, Callable, Literal, Union

from datasets import (
    Dataset as HfDataset,
    DatasetDict as HfDatasetDict,
    concatenate_datasets,
    load_dataset as hf_load_dataset,
)


@dataclass
class DatasetConfig:
    # split -> unshuffled dataset of items
    loader: Callable[[str], HfDataset]
    # formats items to have keys 'txt' and 'hard_label', takes a random.Random rng
    # (or for generative tasks, 'ctx' and 'target', and no 'hard_label' key)
    # deprecated OAI legacy:
    # optionally also adds the key 'choices', a pair of strings, indicating to use the
    # lm head
    formatter: Callable[[Any], Any]
    # "classify" or "generate"
    task: str = "classify"


# mapping from dataset name to load function and format function
_REGISTRY: dict[str, DatasetConfig] = {}


def register_dataset(name: str, config: DatasetConfig):
    _REGISTRY[name] = config


def balance(ds: HfDataset, seed: int):
    """Undersample balance to 50/50"""

    label_counts = Counter(ds["hard_label"])
    assert len(label_counts) == 2, f"Dataset must be binary {label_counts}"

    # undersample the majority class
    majority_label = max(label_counts, key=lambda k: label_counts[k])
    minority_label = 1 - majority_label
    minority_count = label_counts[minority_label]
    minority_ds = ds.filter(lambda ex: ex["hard_label"] == minority_label)
    majority_ds = (
        ds.filter(lambda ex: ex["hard_label"] == majority_label)
        .shuffle(seed=seed)
        .select(range(minority_count))
    )
    return concatenate_datasets([minority_ds, majority_ds]).shuffle(seed=seed)


def load_and_process_train_test(
    ds_name: str,
    split_sizes: dict[str, int],
    seed: int = 0,
    take_test_from_train: bool = False,
):
    n_test = split_sizes.get("test", 0)
    if take_test_from_train:
        # in this case we gather excess documents from the train set, and
        # at the end redistribute them to the test set
        split_sizes["train"] += n_test
        del split_sizes["test"]

    if ds_name not in _REGISTRY:
        raise ValueError(f"Unknown dataset {ds_name}, please register")
    cfg = _REGISTRY[ds_name]
    results = {}
    for split, n_docs in split_sizes.items():
        ds = cfg.loader(split).shuffle(seed=seed)
        ds = ds.map(functools.partial(cfg.formatter, rng=Random(seed)))  # type: ignore

        if cfg.task == "generate":
            ds = ds.filter(lambda ex: ex["ctx"] != "")  # remove empty texts
            ds = ds.filter(lambda ex: ex["target"] != "")
        else:
            ds = ds.filter(lambda ex: ex["txt"] != "")  # remove empty texts
            ds = balance(ds, seed)  # balance to 50/50

        try:
            ds = ds.select(range(n_docs))
        except IndexError:
            print(f"{ds_name} has < {n_docs} docs after balancing, using all {len(ds)}")

        if cfg.task == "generate":
            ds = ds.map(
                lambda ex: {
                    "id": hashlib.sha1(ex["ctx"].encode()).hexdigest()[:8],
                }
            )
        else:
            ds = ds.map(
                lambda ex: {
                    "id": hashlib.sha1(ex["txt"].encode()).hexdigest()[:8],
                    "soft_label": [
                        1 - float(ex["hard_label"]),
                        float(ex["hard_label"]),
                    ],
                }
            )
        results[split] = ds

    if take_test_from_train:
        # take the first n_test examples from the training set as the test set
        results["test"] = results["train"].select(range(n_test))
        results["train"] = results["train"].select(range(n_test, len(results["train"])))
    return results 


def load_and_process_dataset(
    ds_name: str,
    n_train: int,
    n_val: int,
    n_test: int,
    n_predict: Union[Literal["train"], int],
    take_test_from_train: bool = False,
    seed=0,
) -> HfDatasetDict:
    """
    Returns a dict with keys 'train', 'val', 'test', and optionally 'predict', and dataset values
    Examples in 'test' split can never appear in 'train', 'val', or 'predict' on any run.
    """
    split_sizes = dict(train=n_train + n_val, test=n_test)
    if n_predict != "train":
        assert n_predict >= 0
        split_sizes["train"] += n_predict

    results = load_and_process_train_test(
        ds_name, split_sizes, seed, take_test_from_train
    )

    splits = dict(
        val=results["train"].select(range(n_val)),
        train=results["train"].select(range(n_val, len(results["train"]))),
        test=results["test"],
    )
    if n_predict == "train":
        # simply use the training set for predictions
        splits["predict"] = splits["train"]
    elif n_predict > 0:
        # take the requested *fraction* of examples from the training set
        subsplits = splits["train"].train_test_split(
            test_size=n_predict / (n_train + n_predict)
        )
        splits["train"], splits["predict"] = subsplits["train"], subsplits["test"]

    return HfDatasetDict(splits)


warned_about_choices = set()


def encode_choice(text, tokenizer):
    global warned_about_choices

    c_ids = tokenizer.encode(text, add_special_tokens=False)

    # some tokenizers split off the leading whitespace character
    if tokenizer.decode(c_ids[0]).strip() == "":
        c_ids = c_ids[1:]
        assert c_ids == tokenizer.encode(text.lstrip(), add_special_tokens=False)

    c_ids = tuple(c_ids)
    if len(c_ids) != 1 and c_ids not in warned_about_choices:
        assert c_ids[0] not in [
            c[0] for c in warned_about_choices
        ], "Choice shares first token with another choice"
        warned_about_choices.add(c_ids)
        print(
            f'Warning: Only the first token of multitoken choice "{text}" will be used'
        )
    return c_ids[0]


def hf_loader(*hf_name, split_names=None, n_test=None):
    """
    If `split_names` is provided, it maps from the requested
    split name to the actual name in the hugginface dataset.
    If `n_test` is provided, it will concatenate all splits together
    and then take a deterministic test set of size `n_test` from it.
    """

    # this thunk avoids loading datasets at import time
    def thunk(split):
        nonlocal split_names
        if n_test is not None:
            assert split_names is None
            ds = hf_load_dataset(*hf_name, trust_remote_code=True)
            if isinstance(ds, HfDatasetDict):
                ds = concatenate_datasets(ds.values())  # type: ignore
            assert isinstance(ds, HfDataset)
            # the seed is fixed so that all runs use the same test pool
            splits = ds.train_test_split(test_size=n_test, seed=0)

            return splits[split]

        if split_names is None:
            split_names = dict()

        return hf_load_dataset(*hf_name, split=split_names.get(split, split), trust_remote_code=True)

    return thunk


#########################
# Datasets Registration #
#########################


def format_ethics_utilitarianism(ex, rng):
    hard_label = int(rng.random() < 0.5)

    choices = [ex["baseline"], ex["less_pleasant"]]
    rng.shuffle(choices)

    correct = choices.index(ex["baseline"])
    response = correct if hard_label else 1 - correct

    txt = f"Which is more pleasant?\n1) {choices[0]}\n2) {choices[1]} A: {response + 1}"
    return dict(txt=txt, hard_label=hard_label)

register_dataset(
    "ethics-utilitarianism",
    DatasetConfig(
        loader=hf_loader("hendrycks/ethics", "utilitarianism"),  # type: ignore
        formatter=format_ethics_utilitarianism,  # type: ignore
    ),
)


def format_twitter_sentiment(ex, rng):
    return dict(txt=ex["text"], hard_label=ex["label"])

register_dataset(
    "twitter-sentiment",
    DatasetConfig(
        loader=hf_loader("EleutherAI/twitter-sentiment"),  # type: ignore
        formatter=format_twitter_sentiment,  # type: ignore
    ),
)


SCIQ_N_TEST = 3000

def format_sciq(ex, rng):
    hard_label = int(rng.random() < 0.5)
    if hard_label:
        ans = ex["correct_answer"]
    else:
        ans = rng.choice([ex["distractor1"], ex["distractor2"], ex["distractor3"]])

    txt = f"Q: {ex['question']} A: {ans}"
    return dict(txt=txt, hard_label=hard_label)

register_dataset(
    "sciq",
    DatasetConfig(
        loader=hf_loader("sciq", n_test=SCIQ_N_TEST),  # type: ignore
        formatter=format_sciq,  # type: ignore
    ),
)


def format_anthropic_hh(ex, rng):
    hard_label = int(rng.random() < 0.5)
    txt = ex["chosen"] if hard_label else ex["rejected"]
    return dict(txt=txt, hard_label=hard_label)

register_dataset(
    "anthropic_hh",
    DatasetConfig(
        loader=hf_loader("Anthropic/hh-rlhf"),  # type: ignore
        formatter=format_anthropic_hh,  # type: ignore
    ),
)


def format_cosmosqa(ex, rng):
    true_answer = ex["answer" + str(ex["label"])]
    if "None of the above choices ." in true_answer:
        hard_label = 0
    else:
        assert "None of the above choices" not in true_answer, true_answer
        hard_label = int(rng.random() < 0.5)
    if hard_label:
        answer = true_answer
    else:
        candidate_answers = [ex["answer" + str(i)] for i in range(4)]
        answer = rng.choice([x for x in candidate_answers if x != true_answer])
    txt = f"Context: {ex['context']}\nQuestion: {ex['question']}\nAnswer: {answer}"
    return dict(txt=txt, hard_label=hard_label)

register_dataset(
    "cosmos_qa",
    DatasetConfig(
        loader=hf_loader(
            "cosmos_qa", split_names=dict(test="validation")
        ),  # type: ignore
        formatter=format_cosmosqa,  # type: ignore
    ),
)


def format_boolq(ex, rng):
    hard_label = int(ex["answer"])
    txt = f"Passage: {ex['passage']}\nQuestion: {ex['question']}"
    return dict(txt=txt, hard_label=hard_label)

register_dataset(
    "boolq",
    DatasetConfig(
        loader=hf_loader("boolq", split_names=dict(test="validation")),  # type: ignore
        formatter=format_boolq,  # type: ignore
    ),
)


def format_amazon_polarity(ex, rng):
    return dict(txt=f"{ex['title']} {ex['content']}", hard_label=ex["label"])

register_dataset(
    "amazon_polarity",
    DatasetConfig(
        loader=hf_loader("amazon_polarity"),  # type: ignore
        formatter=format_amazon_polarity,  # type: ignore
    ),
)


VALID_DATASETS: list[str] = list(_REGISTRY.keys())


"""
from datasets import disable_caching
disable_caching()

from w2s.datasets import load_and_process_dataset, VALID_DATASETS
import numpy as np

ds_name = "boolq"
print(VALID_DATASETS)

ds = load_and_process_dataset(ds_name, split_sizes=dict(train=500, test=10))
train = list(ds['train'])
test = list(ds['test'])
print(test[0])
print(np.mean([x['hard_label'] for x in train]))
"""
