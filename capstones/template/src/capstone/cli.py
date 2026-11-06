"""Command line entry points used by the Makefile: data, train, evaluate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .data import load_raw, split, validate
from .evaluate import evaluate
from .features import FEATURES, build_features
from .models import MeanBaseline, Metadata, build_model, load_bundle, predict_frame, save_bundle


def cmd_data(config: dict) -> None:
    df = validate(load_raw(config))     # TODO: download() first, then load_raw()
    out = Path(config["data"]["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "clean.csv.gz", index=False)  # parquet is better; add pyarrow if you want it
    print(f"wrote {len(df):,} rows to {out}")


def cmd_train(config: dict) -> None:
    df = validate(load_raw(config))
    train, _ = split(df, config)        # the test split is NOT touched here
    X, y = build_features(train), train["target"]
    art = Path(config["paths"]["artifacts_dir"])
    for name, model in [("baseline", MeanBaseline()), ("current", build_model(config))]:
        model.fit(X, y)
        save_bundle(model, Metadata(name=config["project"]["name"], version=name,
                                    features=FEATURES, config=config), art / name)
    print(f"saved baseline and model to {art}")


def cmd_evaluate(config: dict) -> None:
    df = validate(load_raw(config))
    _, test = split(df, config)
    X, y = build_features(test), test["target"]
    art, reports = Path(config["paths"]["artifacts_dir"]), Path(config["paths"]["reports_dir"])
    results = {}
    for name in ("baseline", "current"):
        model, _ = load_bundle(art / name)
        results[name] = evaluate(y, predict_frame(model, X))
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="capstone")
    parser.add_argument("command", choices=["data", "train", "evaluate"])
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    {"data": cmd_data, "train": cmd_train, "evaluate": cmd_evaluate}[args.command](config)


if __name__ == "__main__":
    main()
