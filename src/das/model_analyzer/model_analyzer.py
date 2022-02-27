"""
The main script that serves as the entry-point for all kinds of training experiments.
"""


import torch
from das.data.data_args import DataArguments
from das.model_analyzer.analysis_tasks.factory import AnalysisTaskFactory
from das.model_analyzer.analyzer_args import AnalyzerArguments
from das.utils.basic_args import BasicArguments
from das.utils.basic_utils import create_logger

# setup logging
logger = create_logger(__name__)

# define dataclasses to parse arguments from
ARG_DATA_CLASSES = [BasicArguments, DataArguments, AnalyzerArguments]

# torch hub bug fix https://github.com/pytorch/vision/issues/4156
torch.hub._validate_not_a_forked_repo = lambda a, b, c: True


class ModelAnalyzer:
    def __init__(
            self,
            basic_args: BasicArguments,
            data_args: DataArguments,
            analyzer_args: AnalyzerArguments) -> None:

        # initializer configuration arguments
        self.basic_args = basic_args
        self.data_args = data_args
        self.analyzer_args = analyzer_args
        self.datamodule = None

    def run(self):
        logger.info("Initializing model analyzer.")
        for model_args in self.analyzer_args.models:
            logger.info(
                f"Analyzing model [{model_args.model_name}] on "
                f"dataset [{self.data_args.dataset_name}]")

            for analysis_task_args in self.analyzer_args.analysis_tasks:
                analysis_task_class = AnalysisTaskFactory.get_task(
                    analysis_task_args.task_name)
                analysis_task = \
                    analysis_task_class(
                        self.basic_args,
                        self.data_args,
                        model_args,
                        self.analyzer_args,
                        analysis_task_args)
                analysis_task.run()
