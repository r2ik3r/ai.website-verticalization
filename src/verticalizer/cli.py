# src/verticalizer/cli.py (replace file)

import argparse
import os
import json
from .utils.logging import getlogger
from .utils.seed import seedall
from .pipeline.io import readtable, writejsonl
from .pipeline.nodes import train_from_labeled, infer as infernodes, evaluate as evalnodes
from .models.persistence import savemodel
from .apps.crawler.cli import addcrawlerclisubparsers, handlecrawlerargs
from .apps.embedder.cli import addembedderclisubparsers, handleembedderargs
from .apps.trainer.cli import addtrainerclisubparsers, handletrainerargs
from .apps.infer.cli import addinferclisubparsers, handleinferargs
from .apps.evaluate.cli import addevaluateclisubparsers, handleevaluateargs

log = getlogger("verticalizer")

def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def app():
    parser = argparse.ArgumentParser(prog="verticalizer")
    sub = parser.add_subparsers(dest="cmd", required=True)
    # Wire subcommands
    addcrawlerclisubparsers(sub)
    addembedderclisubparsers(sub)
    addtrainerclisubparsers(sub)
    addinferclisubparsers(sub)
    addevaluateclisubparsers(sub)

    args = parser.parse_args()
    seedall(42)
    if args.cmd == "crawl":
        handlecrawlerargs(args)
    elif args.cmd == "embed":
        handleembedderargs(args)
    elif args.cmd == "train":
        handletrainerargs(args)
    elif args.cmd == "infer":
        handleinferargs(args)
    elif args.cmd == "eval":
        handleevaluateargs(args)
    else:
        parser.error("Unknown command")

if __name__ == "__main__":
    app()