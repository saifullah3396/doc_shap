"""
The main script that serves as the entry-point for all kinds of training experiments.
"""


import matplotlib.pyplot as plt
import torch
import torchmetrics
from das.data.data_args import DataArguments
from das.model_analyzer.analysis_tasks.base import AnalysisTask
from das.model_analyzer.analyzer_args import AnalysisTaskArguments, AnalyzerArguments
from das.model_analyzer.utils import annotate_heatmap, heatmap
from das.models.model_args import ModelArguments
from das.utils.basic_args import BasicArguments
from das.utils.basic_utils import create_logger
from das.utils.metrics import TrueLabelConfidence

# setup logging
logger = create_logger(__name__)


class GenerateMetricsTask(AnalysisTask):
    def __init__(
        self,
        basic_args: BasicArguments,
        data_args: DataArguments,
        model_args: ModelArguments,
        analyzer_args: AnalyzerArguments,
        analysis_task_args: AnalysisTaskArguments,
    ) -> None:
        super().__init__(
            "model_metrics",
            basic_args,
            data_args,
            model_args,
            analyzer_args,
            analysis_task_args,
        )

    def accuracy_setup(self):
        self.model.test_metrics["acc"] = torchmetrics.Accuracy(
            num_classes=self.num_labels
        )

    def confusion_matrix_setup(self):
        self.model.test_metrics["confusion_matrix"] = torchmetrics.ConfusionMatrix(
            num_classes=self.num_labels
        )

    def true_label_confidence_setup(self):
        self.model.test_metrics["true_label_confidence"] = TrueLabelConfidence()

    def get_accuracy_results(self, test_results):
        return test_results["tacc"]

    def get_confusion_matrix_results(self, test_results):
        # get confusion matrix and normalize it
        conf_mat = self.model.test_metrics["confusion_matrix"].compute()
        conf_mat = conf_mat / conf_mat.sum(dim=1).unsqueeze(1)

        # get accuracy and missclassification
        accuracy = None
        misclass = None
        if "tacc" in test_results:
            accuracy = test_results["tacc"]
            misclass = 1 - accuracy

        # generate confusion matrix figure
        fig = plt.figure(figsize=(10, 10))
        plt.ylabel("True label")
        if accuracy != None:
            plt.xlabel(
                "Predicted label\naccuracy={:0.4f}; misclass={:0.4f}".format(
                    accuracy, misclass
                )
            )
        plt.title(
            f"Confusion Matrix for model {self.model.model_name} on Dataset "
            "{self.data_args.dataset_name}",
            y=-0.2,
        )

        # generate confusion matrix heatmap
        ax = fig.gca()
        im, _ = heatmap(
            conf_mat,
            self.labels,
            self.labels,
            ax=ax,
            cmap="Blues",
            cbarlabel="Frequency",
        )
        _ = annotate_heatmap(im, valfmt="{x:.2f}")

        # save the figure
        plt.tight_layout()
        conf_mat_img = self.output_dir / "confusion_matrix.jpg"
        plt.savefig(conf_mat_img, dpi=300)

        return conf_mat

    def get_true_label_confidence_results(self, test_results):
        return self.model.test_metrics["true_label_confidence"].compute()

    def run(self):
        # load data only from pickle format
        results = self.data_cachers["pickle"].load_data_from_cache(
            self.task_output_name
        )
        results = {} if results == None else results

        # remove all metrics
        self.model.test_metrics = torch.nn.ModuleDict()

        # run metric setup before test
        all_metrics_computed = True
        for metric in self.analysis_task_args.metrics:
            try:
                if metric not in results:
                    all_metrics_computed = False
                    getattr(self, f"{metric}_setup")()
            except AttributeError:
                pass

        if not all_metrics_computed:
            # test model on the dataset
            test_results = self.test_model()

        # after tests
        logger.info("Results:")
        for metric in self.analysis_task_args.metrics:
            try:
                if metric not in results:
                    results[metric] = getattr(self, f"get_{metric}_results")(
                        test_results[-1]
                    )
                if metric == "confusion_matrix":
                    continue
                print(f"{metric}:\n\t{results[metric]}")
            except AttributeError:
                pass

        for data_cacher in self.data_cachers.values():
            # save in readable json format as well as pickle format
            data_cacher.save_data_to_cache(results, self.task_output_name)
