"""
The main script that serves as the entry-point for all kinds of training experiments.
"""

import argparse
import dataclasses
import gc
import logging
import os

import torch
from das.data.data_args import DataArguments
from das.model_analyzer.analyzer_args import AnalyzerArguments
from das.model_analyzer.model_analyzer import ModelAnalyzer
from das.utils.arg_parser import DASArgumentParser
from das.utils.basic_args import BasicArguments
from das.utils.basic_utils import configure_logger, create_logger

# setup logging
logger = create_logger(__name__)

# define dataclasses to parse arguments from
ARG_DATA_CLASSES = [BasicArguments, DataArguments, AnalyzerArguments]

# torch hub bug fix https://github.com/pytorch/vision/issues/4156
torch.hub._validate_not_a_forked_repo = lambda a, b, c: True


def parse_args():
    """
    Parses script arguments.
    """

    arg_parser_main = argparse.ArgumentParser()
    arg_parser_main.add_argument("--cfg", required=False)

    # get config file path
    args, unknown = arg_parser_main.parse_known_args()

    # initialize the argument parsers
    arg_parser = DASArgumentParser(ARG_DATA_CLASSES)

    # parse arguments either based on a json file or directly
    if args.cfg is not None:
        if args.cfg.endswith(".json"):
            return arg_parser.parse_json_file(os.path.abspath(args.cfg))
        elif args.cfg.endswith(".yaml"):
            return arg_parser.parse_yaml_file(os.path.abspath(args.cfg), unknown)
    else:
        return arg_parser.parse_args_into_dataclasses()


def print_args(title, args):
    """
    Pretty prints the arguments.
    """
    args_message = f"\n{title}:\n"
    for (k, v) in dataclasses.asdict(args).items():
        args_message += f"\t{k}: {v}\n"
    print(args_message)


def empty_cache():
    torch.cuda.empty_cache()
    gc.collect()


def main():
    """
    Initializes the training of a model given dataset, and their configurations.
    """

    # empty cuda cache
    empty_cache()

    # parse arguments
    basic_args, data_args, analyzer_args = parse_args()

    # # print arguments for verbosity
    logger.info("Initializing the training script with the following arguments:")
    print_args("Basic arguments", basic_args)
    print_args("Dataset arguments", data_args)
    print_args("Analyzer arguments", analyzer_args)

    # configure pytorch-lightning logger
    pl_logger = logging.getLogger("pytorch_lightning")
    configure_logger(pl_logger)

    # intialize torch random seed
    torch.manual_seed(basic_args.seed)

    # initialize the analyzer
    analyzer = ModelAnalyzer(basic_args, data_args, analyzer_args)
    analyzer.run()


if __name__ == "__main__":
    main()
