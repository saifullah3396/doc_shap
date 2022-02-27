"""
The main script that serves as the entry-point for all kinds of training experiments.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from das.data.data_args import DataArguments
from das.model_analyzer.analysis_tasks.base import AnalysisTask
from das.model_analyzer.analyzer_args import AnalysisTaskArguments, AnalyzerArguments
from das.models.model_args import ModelArguments
from das.utils.basic_args import BasicArguments
from das.utils.basic_utils import create_logger
from numpy.core.numeric import ones_like
from shap.plots import colors

# setup logging
logger = create_logger(__name__)


class GenerateShapVisualizationsTask(AnalysisTask):
    def __init__(
        self,
        basic_args: BasicArguments,
        data_args: DataArguments,
        model_args: ModelArguments,
        analyzer_args: AnalyzerArguments,
        analysis_task_args: AnalysisTaskArguments,
    ) -> None:
        super().__init__(
            "shap_visualizations",
            basic_args,
            data_args,
            model_args,
            analyzer_args,
            analysis_task_args,
        )

    def setup_output_dir(self):
        return super().setup_output_dir().parent

    def save_visualization(self, sample, output_path):
        # read image
        image = cv2.imread(sample["image_file_path"])
        shap_values = sample["shap_values"]
        indices = sample["indices"]

        # resizing image
        if not self.analysis_task_args.resize_shap:
            dsize = (
                sample["shap_values"][0].shape[3],
                sample["shap_values"][0].shape[2],
            )
            image = cv2.resize(image, dsize=dsize)

        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True)

        # save the original image
        plt.imsave(output_path, image)

        for idx, sv in enumerate(shap_values):
            abs_vals = np.stack([np.abs(sv)], 0).flatten()
            max_val = np.nanpercentile(abs_vals, 99.9)
            label = self.labels[indices[idx]]
            if self.analysis_task_args.resize_shap:
                dsize = (image.shape[1], image.shape[0])
                sv = cv2.resize(sv, dsize=dsize)
            fig = plt.figure(facecolor="w", frameon=False)
            ax = fig.gca()
            ax.set_axis_off()
            ax.imshow(ones_like(image) * 255, cmap=plt.get_cmap("gray"))
            ax.imshow(
                image,
                cmap=plt.get_cmap("gray"),
                alpha=0.5,
                extent=(-1, sv.shape[1], sv.shape[0], -1),
            )
            ax.imshow(
                sv,
                cmap=colors.red_transparent_blue,
                vmin=-max_val,
                vmax=max_val,
                interpolation="nearest",
                alpha=0.5,
            )
            fig.savefig(
                str(output_path)[:-4] + f"_{label}.png",
                dpi=600,
                bbox_inches="tight",
                pad_inches=0,
                transparent=False,
            )
            fig.clear()
            plt.close(fig)

    def get_shap_data_paths(self):
        # get shap values folder path
        shap_data_paths = []
        folder_path = Path(self.output_dir / "shap_values")
        for file in folder_path.glob("**/*"):
            if not file.name.endswith(".pickle"):
                continue
            shap_data_paths.append(file)
        return shap_data_paths

    def run(self):
        self.datamodule = self.setup_datamodule()
        shap_data_paths = self.get_shap_data_paths()

        per_class_samples = {}
        for data_path in tqdm.tqdm(shap_data_paths):
            output_path = Path(
                Path(str(data_path).replace("shap_values", "shap_visualizations"))
            ).with_suffix(".png")
            label = output_path.parents[2].name
            if "wrong" in str(data_path):
                continue
            model_name = output_path.with_suffix("").name
            if model_name not in per_class_samples:
                per_class_samples[model_name] = {}
            if label not in per_class_samples[model_name]:
                per_class_samples[model_name][label] = 0
            if per_class_samples[model_name][label] >= 50:
                continue
            output_path = output_path.parent
            output_path = Path(str(output_path).replace("correct", model_name))
            output_path = Path(str(output_path) + ".png")
            if not output_path.exists():
                sample = self.data_cachers["pickle"].load_data(data_path)
                self.save_visualization(sample, output_path)
            per_class_samples[model_name][label] += 1
