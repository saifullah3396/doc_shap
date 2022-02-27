"""
The main script that serves as the entry-point for all kinds of training experiments.
"""

import copy
import dataclasses
import pickle
import uuid
from pathlib import Path

import captum
import cv2
import matplotlib
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
from matplotlib import image
from numpy.lib.arraysetops import isin
from PIL import Image
from shap.plots import colors
from torch.functional import Tensor
from torchvision.transforms.functional import resize

# setup logging
logger = create_logger(__name__)

matplotlib.rcParams.update({"font.size": 5})


class FeaturePerturbationAnalysisTask(AnalysisTask):
    def __init__(
        self,
        basic_args: BasicArguments,
        data_args: DataArguments,
        model_args: ModelArguments,
        analyzer_args: AnalyzerArguments,
        analysis_task_args: AnalysisTaskArguments,
    ) -> None:
        super().__init__(
            "feature_perturbation_analysis",
            basic_args,
            data_args,
            model_args,
            analyzer_args,
            analysis_task_args,
        )

    def get_image(self, sample):
        # load image, resize it to feature attribution map size and
        # convert it to tensor
        sv = sample["shap_values"][0]
        image_tensor = np.array(Image.open(sample["image_file_path"]))
        image_tensor = torch.tensor(image_tensor)
        if len(image_tensor.shape) == 2:
            image_tensor = torch.unsqueeze(image_tensor, 0)
        image_tensor = resize(image_tensor, (sv.shape[1], sv.shape[0]))
        return image_tensor

    def load_data(self, path):
        if not Path(path).exists():
            logger.error(f"File [{path}] does not exist.")
            exit(1)

        data = None
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data

    def generate_confidence_curve(self, data):
        total = []
        for sample in data["samples"]:
            if sample is None:
                continue
            perturbation_confg_scores = sample["true_class_confidence_scores"]
            per_sample = []
            for n, conf in enumerate(perturbation_confg_scores[1:]):
                per_sample.append(conf)
            total.append(per_sample)
        return torch.tensor(total).mean(dim=0)

    def generate_aopc(self, data, label=None):
        aopc_curve_total = []
        for sample in data["samples"]:
            if sample is None:
                continue
            if label is not None and sample["label"] != label:
                continue
            perturbation_confg_scores = sample["true_class_confidence_scores"]
            aopc_curve_per_sample = []
            cum_value = 0.0
            for n, conf in enumerate(perturbation_confg_scores[:40]):
                if n == 0:
                    aopc_curve_per_sample.append(0)
                    continue
                cum_value += perturbation_confg_scores[0] - conf
                aopc_curve_per_sample.append(cum_value / (n + 1))
            aopc_curve_total.append(aopc_curve_per_sample)
        return torch.tensor(aopc_curve_total).mean(dim=0)

    def generate_abpc(self, lerf, morf):
        abpc_curve_total = []
        for idx in range(len(lerf["samples"])):
            lerf_sample = lerf["samples"][idx]
            morf_sample = morf["samples"][idx]
            if lerf_sample is None or morf_sample is None:
                continue

            # get scores
            lerf_sample_scores = lerf_sample["true_class_confidence_scores"]
            morf_sample_scores = morf_sample["true_class_confidence_scores"]

            aopc_curve_per_sample = []
            cum_value = 0.0
            for kdx in range(len(lerf_sample_scores)):
                cum_value += lerf_sample_scores[kdx] - morf_sample_scores[kdx]
                aopc_curve_per_sample.append(cum_value / (kdx + 1))
            abpc_curve_total.append(aopc_curve_per_sample)
        return torch.tensor(abpc_curve_total).mean(dim=0)

    def generate_metrics(self, morf):
        less_than_1_percent = []
        less_than_2point5_percent = []
        less_than_5_percent = []
        above_5_percent = []
        for idx in range(len(morf["samples"])):
            morf_sample = morf["samples"][idx]
            if morf_sample is None:
                continue

            if morf_sample["percent_perturbation"] < 0.01:
                less_than_1_percent.append(idx)
                print(
                    morf_sample["image_file_path"], morf_sample["percent_perturbation"]
                )
            elif morf_sample["percent_perturbation"] < 0.025:
                less_than_2point5_percent.append(idx)
            elif morf_sample["percent_perturbation"] < 0.05:
                less_than_5_percent.append(idx)
            else:
                above_5_percent.append(idx)
        print(
            "Number of samples with less than 1 percent perturbation: ",
            len(less_than_1_percent),
        )
        print(
            "Number of samples with less than 2.5 percent perturbation: ",
            len(less_than_2point5_percent),
        )
        print(
            "Number of samples with less than 5 percent perturbation: ",
            len(less_than_5_percent),
        )
        print(
            "Number of samples with above 5 percent perturbation: ",
            len(above_5_percent),
        )
        exit(1)

    def run(self):
        self.datamodule = self.setup_datamodule()

        lerf_data = self.load_data(self.analysis_task_args.least_relevant_first_data)
        morf_data = self.load_data(self.analysis_task_args.most_relevant_first_data)

        self.generate_metrics(morf_data)

        lerf_conf = self.generate_confidence_curve(lerf_data)
        morf_conf = self.generate_confidence_curve(morf_data)

        if len(self.analysis_task_args.random_data) > 0:
            mean_random_conf_curve = []
            for data_path in self.analysis_task_args.random_data:
                data = self.load_data(data_path)
                if data is not None:
                    conf_curve = self.generate_confidence_curve(data)
                    mean_random_conf_curve.append(conf_curve)
            mean_random_conf_curve = torch.stack(mean_random_conf_curve).mean(dim=0)

            plt.plot(mean_random_conf_curve)
        plt.plot(lerf_conf)
        plt.plot(morf_conf)
        plt.show()

        # for label in range(16):
        label = None
        lerf_aopc = self.generate_aopc(lerf_data, label=label)
        morf_aopc = self.generate_aopc(morf_data, label=label)

        mean_random_aopc_curve = None
        if len(self.analysis_task_args.random_data) > 0:
            mean_random_aopc_curve = []
            for data_path in self.analysis_task_args.random_data:
                data = self.load_data(data_path)
                if data is not None:
                    random_aopc = self.generate_aopc(data, label=label)
                    mean_random_aopc_curve.append(random_aopc)
            mean_random_aopc_curve = torch.stack(mean_random_aopc_curve).mean(dim=0)

        if mean_random_aopc_curve is not None:
            plt.plot(mean_random_aopc_curve - mean_random_aopc_curve)
            plt.plot(lerf_aopc - mean_random_aopc_curve)
            plt.plot(morf_aopc - mean_random_aopc_curve)
        else:
            plt.plot(lerf_aopc)
            plt.plot(morf_aopc)
        plt.show()

        abpc = self.generate_abpc(lerf_data, morf_data)

        plt.plot(abpc)
        plt.show()

    def save_visualization(self, sample, perturbation_results, output_path):
        (
            perturbed_image,
            perturbation_outputs,
            percentage_perturbation,
            missclassified,
            pred,
            _,
        ) = perturbation_results

        pred_class = pred.argmax(dim=1)
        true_class_score = pred[:, sample["label"]].item()
        pred_class_score = pred[:, pred_class.item()].item()

        # get top shap value
        sv = sample["shap_values"][0]
        dsize = (sv.shape[1], sv.shape[0])

        # read image
        image = cv2.imread(sample["image_file_path"])
        if not self.analysis_task_args.resize_perturbation:
            dsize = (sv.shape[1], sv.shape[0])
            image = cv2.resize(image, dsize=dsize)
            perturbed_image = perturbed_image.permute(1, 2, 0).cpu().numpy()
        else:
            dsize = (image.shape[1], image.shape[0])
            sv = cv2.resize(sv, dsize=dsize)
            perturbed_image = copy.copy(image)
            for k, v in perturbation_outputs.items():
                perturbation_outputs[k] = cv2.resize(
                    v.cpu().numpy().astype(np.uint8), dsize=dsize
                )
                if k == "black_perturbation_mask":
                    perturbed_image[perturbation_outputs[k] == 1, ...] = 255
                elif k == "white_perturbation_mask":
                    perturbed_image[perturbation_outputs[k] == 1, ...] = 0

        # generate grid config
        elements_per_row = 3
        total_elements = 4

        # generate figure and grid
        fig = plt.figure()
        gs = plt.GridSpec(total_elements // elements_per_row + 1, elements_per_row)
        gs.update(
            top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.25, wspace=0.25
        )
        ax = fig.add_subplot(gs[0])
        # ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(
            f"True: {self.labels[sample['label']]}\n"
            f"Predicted: {self.labels[sample['pred']]}"
        )
        ax.imshow(image, cmap="gray")
        ax.set_aspect("auto")

        # get confidences for each predicted label
        confidence_scores = F.softmax(torch.tensor(sample["logits"]), dim=-1).tolist()[
            0
        ]

        only_get_true_shap_value = True
        if only_get_true_shap_value:
            label_idx = sample["label"]
            label = self.labels[label_idx]
        elif "indices" in sample:
            label_idx = sample["indices"][0]
            label = self.labels[label_idx]
        else:
            label_idx = 0
            label = self.labels[label_idx]

        abs_vals = np.stack([np.abs(sv)], 0).flatten()
        max_val = np.nanpercentile(abs_vals, 99.9)

        ax = fig.add_subplot(gs[1])
        # ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        xlabel = f"Predicted: {label}\n"
        xlabel += f"Predicted class score: {confidence_scores[label_idx]}\n"
        ax.set_xlabel(xlabel)

        ax.imshow(
            image,
            cmap=plt.get_cmap("gray"),
            alpha=0.15,
            extent=(-1, sv.shape[1], sv.shape[0], -1),
        )
        ax.imshow(
            sv,
            cmap=colors.red_transparent_blue,
            vmin=-max_val,
            vmax=max_val,
            interpolation="nearest",
        )
        ax.set_aspect("auto")

        ax = fig.add_subplot(gs[2])
        # ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        xlabel = f"Predicted: {self.labels[pred_class]}\n"
        xlabel += f"Predicted class score: {pred_class_score}\n"
        xlabel += f"True class score: {true_class_score}\n"
        xlabel += f"Percentage perturbation: {percentage_perturbation}\n"
        ax.set_xlabel(xlabel)
        ax.imshow(perturbed_image, cmap="gray")
        ax.set_aspect("auto")

        ax = fig.add_subplot(gs[3])
        # ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"Perturbed image masked")
        ax.imshow(perturbed_image, cmap="gray")
        if "black_perturbation_mask" in perturbation_outputs:
            plt.imshow(
                perturbation_outputs["black_perturbation_mask"],
                "Blues",
                interpolation="none",
                alpha=0.5,
            )
        if "white_perturbation_mask" in perturbation_outputs:
            plt.imshow(
                perturbation_outputs["white_perturbation_mask"],
                "Reds",
                interpolation="none",
                alpha=0.5,
            )
        ax.set_aspect("auto")

        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True)

        print(output_path)
        plt.show()
        # plt.savefig(f'{output_path}', format='png', dpi=300)
        # fig.clear()
        # plt.close(fig)
