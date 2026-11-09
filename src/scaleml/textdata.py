"""Public text datasets from the Hugging Face Hub, sized to your budget.

Downloads are cached (``HF_HOME``; in Colab, set ``USE_DRIVE`` in the setup
cell to keep them across sessions). Every loader takes explicit sizes and a
seed, samples *stratified by label* where it matters, and returns plain
pandas DataFrames so the data can be inspected before any model sees it.

Licences: each dataset has its own terms on its Hub page. The ones used in
Part III are listed in ``part3-deep-learning-systems/README.md`` and in each
chapter that uses them, with links; read them before reuse.
"""

from __future__ import annotations

import pandas as pd

#: name -> (hub id, config, text column, label column, train split, eval split)
DATASETS = {
    "ag_news": ("fancyzhx/ag_news", None, "text", "label", "train", "test"),
    "imdb": ("stanfordnlp/imdb", None, "text", "label", "train", "test"),
    "sst2": ("stanfordnlp/sst2", None, "sentence", "label", "train", "validation"),
}


def load_text_classification(name: str, n_train: int, n_eval: int, seed: int = 0):
    """Return (train_df, eval_df, label_names) with columns ``text`` and ``label``."""
    from datasets import load_dataset

    hub_id, config, text_col, label_col, train_split, eval_split = DATASETS[name]
    ds = load_dataset(hub_id, config)
    names = ds[train_split].features[label_col].names

    def take(split: str, n: int) -> pd.DataFrame:
        df = ds[split].to_pandas()[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
        if n >= len(df):
            return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        frac = n / len(df)
        out = df.groupby("label", group_keys=False).sample(frac=frac, random_state=seed)
        return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return take(train_split, n_train), take(eval_split, n_eval), list(names)
