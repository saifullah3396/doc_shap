"""
The main script that serves as the entry-point for all kinds of training experiments.
"""

import sys
from pathlib import Path

import pytorch_lightning as pl
from das.data.data_args import DataArguments
from das.data.data_modules.factory import DataModuleFactory
from das.model_analyzer.analyzer_args import AnalysisTaskArguments, AnalyzerArguments
from das.model_analyzer.utils import DataCacher
from das.models.model_args import ModelArguments, ModelFactory
from das.utils.basic_args import BasicArguments
from das.utils.basic_utils import create_logger

logger = create_logger(__name__)


class AnalysisTask:
    def __init__(
        self,
        task_output_name,
        basic_args: BasicArguments,
        data_args: DataArguments,
        model_args: ModelArguments,
        analyzer_args: AnalyzerArguments,
        analysis_task_args: AnalysisTaskArguments,
    ) -> None:

        self.task_output_name = task_output_name
        self.basic_args = basic_args
        self.data_args = data_args
        self.model_args = model_args
        self.analyzer_args = analyzer_args
        self.analysis_task_args = analysis_task_args

        # setup datamodule
        self.datamodule = self.setup_datamodule()

        # setup the model
        self.model = self.setup_model()

        # setup analyser output dir
        self.output_dir = self.setup_output_dir()

        # setup data caching
        self.data_cachers = {}
        for cacher_type in ["pickle", "json"]:
            self.data_cachers[cacher_type] = self.setup_data_cacher(cacher_type)

    def setup_datamodule(self):
        # initialize data-handling module, set collate_fns later
        datamodule = DataModuleFactory.create_datamodule(
            self.basic_args, self.data_args
        )

        # prepare the modules
        datamodule.prepare_data()
        datamodule.setup()

        self.num_labels = datamodule.num_labels
        self.labels = datamodule.labels

        # set datamdule
        return datamodule

    def setup_output_dir(self):
        output_dir = (
            Path(self.analyzer_args.analyzer_output_dir) / self.data_args.dataset_name
        )

        if self.analyzer_args.output_data_subdir != "":
            output_dir = output_dir / self.analyzer_args.output_data_subdir

        output_dir = output_dir / self.model.model_name

        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        return output_dir

    def setup_data_cacher(self, cacher_type):
        return DataCacher(self.output_dir, cacher_type=cacher_type)

    def setup_model(self):
        # get model class
        model_class = ModelFactory.get_model_class(
            self.model_args.model_name, self.model_args.model_task
        )

        # intialize the lightning module for training
        model = model_class(
            self.basic_args,
            self.model_args,
            training_args=None,
            data_args=self.data_args,
            datamodule=self.datamodule,
        )

        # if model checkpoint is present, use it to load the weights
        if self.model_args.model_checkpoint_file is None:
            # intialize the model for training
            model = model_class(
                self.basic_args,
                self.model_args,
                training_args=None,
                data_args=self.data_args,
                datamodule=self.datamodule,
            )
        else:
            if not self.model_args.model_checkpoint_file.startswith("http"):
                model_checkpoint = Path(self.model_args.model_checkpoint_file)
                if not model_checkpoint.exists():
                    logger.error(
                        f"Checkpoint not found, cannot load weights from {model_checkpoint}."
                    )
                    sys.exit(1)
            else:
                model_checkpoint = self.model_args.model_checkpoint_file
            logger.info(f"Loading model from model checkpoint: {model_checkpoint}")

            # load model weights from checkpoint
            model = model_class.load_from_checkpoint(
                model_checkpoint,
                strict=True,
                basic_args=self.basic_args,
                model_args=self.model_args,
                training_args=None,
                data_args=self.data_args,
                datamodule=self.datamodule,
            )

        # set model device
        model = model.cuda()

        # put model in evaluation mode
        model.eval()

        return model

    def test_model(self):
        # get data collator required for the model
        self.datamodule.collate_fns = self.model.get_data_collators(
            self.data_args, None
        )

        # initialize the training
        trainer = pl.Trainer(
            gpus=self.basic_args.n_gpu,
            num_nodes=self.basic_args.n_nodes,
        )

        # get test results
        return trainer.test(self.model, datamodule=self.datamodule, verbose=False)
