"""
The main script that serves as the entry-point for all kinds of training experiments.
"""

import copy
import dataclasses
import random
import uuid
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import captum
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from captum._utils.common import _expand_additional_forward_args, _reduce_list
from captum._utils.typing import TargetType
from captum.robust import MinParamPerturbation
from captum.robust._core.metrics.min_param_perturbation import MinParamPerturbation
from captum.robust._core.perturbation import Perturbation
from cv2 import data
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


def drange(
    min_val: Union[int, float], max_val: Union[int, float], step_val: Union[int, float]
) -> Generator[Union[int, float], None, None]:
    curr = min_val
    while curr < max_val:
        yield curr
        curr += step_val


class CustomMinParamPerturbation(MinParamPerturbation):
    def _linear_search(
        self,
        inputs: Any,
        preproc_input: Any,
        attack_kwargs: Optional[Dict[str, Any]],
        additional_forward_args: Any,
        expanded_additional_args: Any,
        correct_fn_kwargs: Optional[Dict[str, Any]],
        target: TargetType,
        perturbations_per_eval: int,
    ) -> Tuple[Any, Optional[Union[int, float]]]:
        input_list = []
        attack_inp_list = []
        output_list = []
        param_list = []
        model_outputs_list = []

        last_param = None
        last_output = None
        last_attack_input = None
        for param in drange(self.arg_min, self.arg_max, self.arg_step):
            for _ in range(self.num_attempts):
                preproc_attacked_inp, attacked_inp = self._apply_attack(
                    inputs, preproc_input, attack_kwargs, param
                )
                input_list.append(preproc_attacked_inp)
                param_list.append(param)
                if bool(attack_kwargs["perturbation_output"]):
                    output_list.append(copy.copy(attack_kwargs["perturbation_output"]))
                else:
                    output_list.append(attack_kwargs["perturbation_output"])
                attack_inp_list.append(attacked_inp)

                if len(input_list) == perturbations_per_eval:
                    model_out, successful_ind = self._evaluate_batch(
                        input_list,
                        expanded_additional_args,
                        correct_fn_kwargs,
                        target,
                    )
                    if len(model_outputs_list) == 0:
                        model_outputs_list.append(model_out)
                    if successful_ind is not None:
                        return (
                            attack_inp_list[successful_ind],
                            output_list[successful_ind],
                            param_list[successful_ind],
                            model_outputs_list,
                            successful_ind,
                            True,
                        )

                    last_param = param_list[-1]
                    last_output = output_list[-1]
                    last_attack_input = attack_inp_list[-1]

                    input_list = []
                    param_list = []
                    output_list = []
                    attack_inp_list = []
        if len(input_list) > 0:
            final_add_args = _expand_additional_forward_args(
                additional_forward_args, len(input_list)
            )
            model_out, successful_ind = self._evaluate_batch(
                input_list,
                final_add_args,
                correct_fn_kwargs,
                target,
            )
            if len(model_outputs_list) == 0:
                model_outputs_list.append(model_out)
            if successful_ind is not None:
                return (
                    attack_inp_list[successful_ind],
                    output_list[successful_ind],
                    param_list[successful_ind],
                    model_outputs_list,
                    successful_ind,
                    True,
                )

            last_param = param_list[-1]
            last_output = output_list[-1]
            last_attack_input = attack_inp_list[-1]

        return (
            last_attack_input,
            last_output,
            last_param,
            model_outputs_list,
            None,
            False,
        )

    def _evaluate_batch(
        self,
        input_list: List,
        additional_forward_args: Any,
        correct_fn_kwargs: Optional[Dict[str, Any]],
        target: TargetType,
    ) -> Optional[int]:
        if additional_forward_args is None:
            additional_forward_args = ()

        all_kwargs = {}
        if target is not None:
            all_kwargs["target"] = target
        if correct_fn_kwargs is not None:
            all_kwargs.update(correct_fn_kwargs)

        if len(input_list) == 1:
            model_out = self.forward_func(input_list[0], *additional_forward_args)
            out_metric = self.correct_fn(model_out, **all_kwargs)
            return 0 if not out_metric else None
        else:
            batched_inps = _reduce_list(input_list)
            model_out = []
            batch_size = 32
            for i in range(4):
                model_out.append(
                    self.forward_func(
                        batched_inps[batch_size * i : (i + 1) * batch_size].cuda(),
                        *additional_forward_args,
                    )
                )
            model_out = torch.cat(model_out)
            current_count = 0
            for i in range(len(input_list)):
                batch_size = (
                    input_list[i].shape[0]
                    if isinstance(input_list[i], Tensor)
                    else input_list[i][0].shape[0]
                )
                out_metric = self.correct_fn(
                    model_out[current_count : current_count + batch_size], **all_kwargs
                )
                if not out_metric:
                    return model_out, i
                current_count += batch_size
            return model_out, None


@dataclasses.dataclass
class PerturbationResults:
    perturbed_image: np.ndarray
    perturbation_outputs: dict
    percent_perturbation: float
    missclassified: bool
    pred: int
    successful_index: int
    true_class_confidence_scores: list


def white_pixel_dropout(
    image,
    dropout_pixels,
    feature_importance_grid=None,
    feature_importance_grid_size=4,
    image_black_and_white_regions=None,
    black_and_white_threshold=200,
    importance_order="descending",
    return_perturbation_output=False,
    perturbation_output={},
):
    if dropout_pixels == 0:
        perturbation_output["white_perturbation_mask"] = (
            torch.ones_like(image).squeeze().bool()
        )
        return image

    # see if bins to drop are greater than total number of bins that can be dropped
    if feature_importance_grid.nelement() < dropout_pixels:
        dropout_pixels = feature_importance_grid.nelement()

    # generate mask over which features to get
    feature_mask = torch.logical_and(
        # only take those features that positively contribute
        feature_importance_grid > 0,
        image_black_and_white_regions > black_and_white_threshold,
    )

    step_y = feature_importance_grid_size
    step_x = feature_importance_grid_size

    # get maximum value by sorting
    image = image.clone()
    top_values = torch.zeros_like(feature_importance_grid).bool()
    if importance_order == "descending":
        top_drop_pixels = torch.topk(
            feature_importance_grid[feature_mask], dropout_pixels
        )
        top_value_indices = feature_mask.nonzero()[top_drop_pixels.indices]
        top_values[top_value_indices[:, 0], top_value_indices[:, 1]] = True
    elif importance_order == "ascending":
        top_drop_pixels = torch.topk(
            feature_importance_grid[feature_mask], dropout_pixels, largest=False
        )
        top_value_indices = feature_mask.nonzero()[top_drop_pixels.indices]
        top_values[top_value_indices[:, 0], top_value_indices[:, 1]] = True
    elif importance_order == "random":
        top_drop_pixels = torch.randperm(
            feature_importance_grid[feature_mask].nelement()
        )[:dropout_pixels]
        top_value_indices = feature_mask.nonzero()[top_drop_pixels]
        top_values[top_value_indices[:, 0], top_value_indices[:, 1]] = True
    else:
        logger.error(f"Unsupported importance_order [{importance_order}] given.")
        exit(1)
    top_values = top_values.repeat_interleave(step_y, dim=0).repeat_interleave(
        step_x, dim=1
    )
    perturbation_output["white_perturbation_mask"] = top_values.cpu()
    image[0][top_values] = 0
    return image


def black_pixel_dropout(
    image,
    dropout_pixels,
    feature_importance_grid=None,
    feature_mask=None,
    feature_importance_grid_size=4,
    image_black_and_white_regions=None,
    black_and_white_threshold=200,
    importance_order="descending",
    return_perturbation_output=False,
    perturbation_output={},
    rand_indices=None,
):
    if dropout_pixels == 0:
        if return_perturbation_output:
            perturbation_output["black_perturbation_mask"] = (
                torch.zeros_like(image).squeeze().bool()
            )
        return image

    # get maximum value by sorting
    image = image.clone()

    top_values = torch.zeros_like(feature_importance_grid).bool()
    if importance_order == "descending":
        # see if bins to drop are greater than total number of bins that can be dropped
        masked_feature_importances = feature_importance_grid[feature_mask]
        if masked_feature_importances.nelement() < dropout_pixels:
            dropout_pixels = masked_feature_importances.nelement()

        top_drop_pixels = torch.topk(masked_feature_importances, dropout_pixels)
        top_value_indices = feature_mask.nonzero()[top_drop_pixels.indices]
        top_values[top_value_indices[:, 0], top_value_indices[:, 1]] = True
    elif importance_order == "ascending":
        # see if bins to drop are greater than total number of bins that can be dropped
        masked_feature_importances = feature_importance_grid[feature_mask]
        if masked_feature_importances.nelement() < dropout_pixels:
            dropout_pixels = masked_feature_importances.nelement()

        top_drop_pixels = torch.topk(
            masked_feature_importances, dropout_pixels, largest=False
        )
        top_value_indices = feature_mask.nonzero()[top_drop_pixels.indices]
        top_values[top_value_indices[:, 0], top_value_indices[:, 1]] = True
    elif importance_order == "random":
        # top_values_flat[rand_indices[:dropout_pixels*step_x*step_y]] = True
        # print(feature_importance_grid)
        top_value_indices = feature_mask.nonzero()[rand_indices[:dropout_pixels]]
        top_values[top_value_indices[:, 0], top_value_indices[:, 1]] = True
    else:
        logger.error(f"Unsupported importance_order [{importance_order}] given.")
        exit(1)

    # if importance_order != 'random':
    top_values = top_values.repeat_interleave(
        feature_importance_grid_size, dim=0
    ).repeat_interleave(feature_importance_grid_size, dim=1)
    if return_perturbation_output:
        perturbation_output["black_perturbation_mask"] = top_values
    image[0][top_values] = 255
    return image


def black_white_pixel_dropout(
    image,
    dropout_pixels,
    feature_importance_grid=None,
    feature_mask=None,
    feature_importance_grid_size=4,
    image_black_and_white_regions=None,
    black_and_white_threshold=200,
    importance_order="descending",
    return_perturbation_output=False,
    perturbation_output={},
    rand_indices=None,
):
    if dropout_pixels == 0:
        if return_perturbation_output:
            perturbation_output["white_perturbation_mask"] = (
                torch.zeros_like(image).squeeze().bool()
            )
            perturbation_output["black_perturbation_mask"] = (
                torch.zeros_like(image).squeeze().bool()
            )
        return image

    # get grid element size
    step_y = feature_importance_grid_size
    step_x = feature_importance_grid_size

    image = image.clone()

    top_values = torch.zeros_like(feature_importance_grid).bool()
    if importance_order == "descending":

        # see if bins to drop are greater than total number of bins that can be dropped
        masked_feature_importances = feature_importance_grid[feature_mask]
        if masked_feature_importances.nelement() < dropout_pixels:
            dropout_pixels = masked_feature_importances.nelement()

        top_drop_pixels = torch.topk(masked_feature_importances, dropout_pixels)
        top_value_indices = feature_mask.nonzero()[top_drop_pixels.indices]
        top_values[top_value_indices[:, 0], top_value_indices[:, 1]] = True
    elif importance_order == "ascending":
        # only take those features that positively contribute
        feature_mask = feature_importance_grid > 0

        # see if bins to drop are greater than total number of bins that can be dropped
        masked_feature_importances = feature_importance_grid[feature_mask]
        if masked_feature_importances.nelement() < dropout_pixels:
            dropout_pixels = masked_feature_importances.nelement()

        top_drop_pixels = torch.topk(
            masked_feature_importances, dropout_pixels, largest=False
        )
        top_value_indices = feature_mask.nonzero()[top_drop_pixels.indices]
        top_values[top_value_indices[:, 0], top_value_indices[:, 1]] = True
    elif importance_order == "random":
        top_value_indices = feature_mask.nonzero()[rand_indices[:dropout_pixels]]
        top_values[top_value_indices[:, 0], top_value_indices[:, 1]] = True
    else:
        logger.error(f"Unsupported importance_order [{importance_order}] given.")
        exit(1)

    # if importance_order != 'random':
    top_values = top_values.repeat_interleave(step_y, dim=0).repeat_interleave(
        step_x, dim=1
    )
    white_pixels_mask = torch.logical_and(
        image_black_and_white_regions > black_and_white_threshold, top_values
    )
    black_pixels_mask = torch.logical_and(
        image_black_and_white_regions < black_and_white_threshold, top_values
    )
    if return_perturbation_output:
        perturbation_output["white_perturbation_mask"] = white_pixels_mask
        perturbation_output["black_perturbation_mask"] = black_pixels_mask
    image[0][white_pixels_mask] = 0
    image[0][black_pixels_mask] = 255
    return image


ATTACKS = {
    "white_pixel_dropout": white_pixel_dropout,
    "black_pixel_dropout": black_pixel_dropout,
    "black_white_pixel_dropout": black_white_pixel_dropout,
}


class FeaturePerturbationTask(AnalysisTask):
    def __init__(
        self,
        basic_args: BasicArguments,
        data_args: DataArguments,
        model_args: ModelArguments,
        analyzer_args: AnalyzerArguments,
        analysis_task_args: AnalysisTaskArguments,
    ) -> None:
        super().__init__(
            "feature_perturbation",
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
        image_tensor = torch.tensor(image_tensor).cuda()
        if len(image_tensor.shape) == 2:
            image_tensor = torch.unsqueeze(image_tensor, 0)
        image_tensor = resize(image_tensor, (sv.shape[1], sv.shape[0]))
        return image_tensor

    def get_feature_importance_grid(self, sample):
        sv = torch.tensor(sample["shap_values"][0])
        kx = self.analysis_task_args.feature_importance_grid_size
        ky = self.analysis_task_args.feature_importance_grid_size
        feature_importance_grid = sv.unfold(0, ky, kx).unfold(1, ky, kx).sum(dim=(2, 3))
        return feature_importance_grid.cuda()

    def get_black_and_white_regions_mask(self, image_tensor):
        black_and_white_threshold = self.analysis_task_args.black_and_white_threshold
        kx = self.analysis_task_args.feature_importance_grid_size
        ky = self.analysis_task_args.feature_importance_grid_size
        black_and_white_regions_fast = (
            (
                image_tensor[0].unfold(0, ky, kx).unfold(1, ky, kx)
                < black_and_white_threshold
            )
            .any(dim=2)
            .any(dim=2)
        )
        black_and_white_regions_fast = black_and_white_regions_fast.repeat_interleave(
            ky, dim=0
        ).repeat_interleave(kx, dim=1)
        return ((~black_and_white_regions_fast).long() * 255).cuda()

    def preproc_fn(self, image):
        sample = {"image": image}
        for t in self.datamodule.test_dataset.transforms:
            sample = t(sample)
        image = sample["image"]
        return torch.unsqueeze(image, 0)

    def correct_fn(self, model_out, target):
        target_tensor = torch.tensor(target)
        return all(model_out.argmax(dim=1) == target_tensor)

    def generate_perturbation(self, sample, output_visualizations_path=None):
        label_idx = sample["label"]
        image_tensor = self.get_image(sample)
        feature_importance_grid = self.get_feature_importance_grid(sample)
        image_black_and_white_regions = self.get_black_and_white_regions_mask(
            image_tensor
        )

        # total pixel bins
        total_pixel_bins = (
            feature_importance_grid.shape[0] * feature_importance_grid.shape[1]
        )

        # total_perturbation
        total_perturbation = int(
            total_pixel_bins * self.analysis_task_args.max_perturbation_percentage
        )
        attack_type = self.analysis_task_args.attack_type
        attack_config = self.analysis_task_args.attack_config
        arg_max = (
            total_perturbation
            if attack_config.arg_max is None
            else attack_config.arg_max
        )
        min_pert_attr = CustomMinParamPerturbation(
            forward_func=self.model_wrapper,
            attack=ATTACKS[attack_type],
            arg_name=attack_config.arg_name,
            mode=attack_config.mode,
            arg_min=attack_config.arg_min,
            arg_max=arg_max,
            arg_step=attack_config.arg_step,
            preproc_fn=self.preproc_fn,
            apply_before_preproc=True,
            correct_fn=self.correct_fn,
        )

        # generate mask over which features to get
        step_y = self.analysis_task_args.feature_importance_grid_size
        step_x = self.analysis_task_args.feature_importance_grid_size

        if attack_type == "black_pixel_dropout":
            # generate mask over which features to get
            feature_mask = torch.logical_and(
                # only take those features that positively contribute
                feature_importance_grid > 0,
                image_black_and_white_regions[::step_y, ::step_x]
                < self.analysis_task_args.black_and_white_threshold,
            )
        else:
            feature_mask = feature_importance_grid > 0

        attack_kwargs = {
            "feature_importance_grid": feature_importance_grid,
            "feature_mask": feature_mask,
            "feature_importance_grid_size": self.analysis_task_args.feature_importance_grid_size,
            "image_black_and_white_regions": image_black_and_white_regions,
            "black_and_white_threshold": self.analysis_task_args.black_and_white_threshold,
            "importance_order": self.analysis_task_args.importance_order,
            "return_perturbation_output": self.analysis_task_args.save_perturbations,
            "perturbation_output": {},
        }

        if self.analysis_task_args.importance_order == "random":
            np.random.seed(self.analysis_task_args.random_seed)
            if attack_type == "black_pixel_dropout":
                rand_indices = np.random.choice(
                    feature_importance_grid[feature_mask].nelement(),
                    size=feature_importance_grid[feature_mask].nelement(),
                    replace=False,
                )
            elif attack_type == "black_white_pixel_dropout":
                white_feature_mask = torch.logical_and(
                    # only take those features that positively contribute
                    feature_importance_grid > 0,
                    image_black_and_white_regions[::step_y, ::step_x]
                    > self.analysis_task_args.black_and_white_threshold,
                )
                black_feature_mask = torch.logical_and(
                    # only take those features that positively contribute
                    feature_importance_grid > 0,
                    image_black_and_white_regions[::step_y, ::step_x]
                    < self.analysis_task_args.black_and_white_threshold,
                )
                rand_indices_white = white_feature_mask.nonzero()
                rand_indices_black = black_feature_mask.nonzero()

                # top_pixels = torch.topk(
                #     feature_importance_grid[feature_mask], arg_max, largest=True
                # ).indices
                # top_black_and_white = image_black_and_white_regions[::step_y, ::step_x][
                #     feature_mask
                # ][top_pixels]
                # n_top_black = len(top_black_and_white[top_black_and_white < 125])
                # n_top_white = len(top_black_and_white[top_black_and_white > 125])
                # ratio_black = n_top_black / (n_top_black + n_top_white)

                # set lesser weight to white for random perturbation since otherwise we
                # would be adding a lot of artifacts instead of actually removing information
                black_prob_total = 0.75
                white_prob_total = 1 - 0.75
                black_prob = black_prob_total / rand_indices_black.shape[0]
                white_prob = white_prob_total / rand_indices_white.shape[0]
                probs = torch.zeros_like(feature_importance_grid)
                probs[black_feature_mask] = black_prob
                probs[white_feature_mask] = white_prob
                rand_indices = np.random.choice(
                    feature_importance_grid[feature_mask].nelement(),
                    size=(arg_max // attack_config.arg_step, attack_config.arg_step),
                    replace=False,
                    p=probs[feature_mask].cpu(),
                ).flatten()

            attack_kwargs["rand_indices"] = rand_indices

        (
            perturbed_image,
            perturbation_outputs,
            dropped,
            model_outputs_list,
            successful_index,
            missclassified,
        ) = min_pert_attr.evaluate(
            image_tensor,
            target=label_idx,
            perturbations_per_eval=128,
            attack_kwargs=attack_kwargs,
        )

        # get the target output confidence scores until change
        model_outputs_list = torch.cat(model_outputs_list)
        true_class_confidence_scores = model_outputs_list[:, label_idx]

        # evaluate last time
        pred = self.model_wrapper(image=self.preproc_fn(perturbed_image))

        # get perturbation_outputs
        return PerturbationResults(
            perturbed_image=perturbed_image,
            perturbation_outputs=perturbation_outputs,
            percent_perturbation=dropped / total_pixel_bins,
            missclassified=missclassified,
            successful_index=successful_index,
            pred=pred,
            true_class_confidence_scores=true_class_confidence_scores,
        )

    def get_shap_data_paths(self):
        # get shap values folder path
        shap_data_paths = []
        folder_path = Path(self.output_dir.parent / "shap_values")
        for file in folder_path.glob("**/*"):
            if not file.name.endswith(f"{self.model.model_name}.pickle"):
                continue
            shap_data_paths.append(file)
        return shap_data_paths

    def run(self):
        logger.info("Running the analysis task [{self.analysis_task_args.task_name}]")

        # intialize torch random seed
        torch.manual_seed(self.analysis_task_args.random_seed)
        with torch.no_grad():
            self.datamodule = self.setup_datamodule()
            shap_data_paths = self.get_shap_data_paths()

            class ModelWrapper(torch.nn.Module):
                def __init__(self, model, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.model = model
                    self.softmax = torch.nn.Softmax(dim=1)

                def forward(self, *args, **kwargs):
                    return self.softmax(self.model(*args, **kwargs).logits)

            # wrap the model around the original model
            self.model_wrapper = ModelWrapper(self.model)
            self.model_wrapper = self.model_wrapper.cuda()
            self.model_wrapper.eval()

            # get results with matching config
            cached_result = None
            checkpoint = 0
            results_root_path = Path(self.output_dir / self.task_output_name)
            cached_result_paths = {}
            for file in results_root_path.glob("**/*"):
                if not file.name.endswith(".pickle"):
                    continue
                dir_name = str(file.parent.name)
                if dir_name not in cached_result_paths:
                    cached_result_paths[dir_name] = 0

                results_split = file.with_suffix("").name.split("_")
                if len(results_split) == 2:
                    checkpoint_n = int(results_split[-1])
                    cached_result_paths[dir_name] = (
                        cached_result_paths[dir_name]
                        if cached_result_paths[dir_name] > checkpoint_n
                        else checkpoint_n
                    )

            cached_result_paths = [
                results_root_path / k / f"results_{v}.pickle"
                for k, v in cached_result_paths.items()
            ]
            for result_path in cached_result_paths:
                # save in readable json format as well as pickle format
                cached_result_loaded = self.data_cachers["pickle"].load_data(
                    result_path
                )
                current_args = dataclasses.asdict(self.analysis_task_args)
                cached_args = cached_result_loaded["config"]
                same_config = True
                ignored_keys = [
                    "n_vis_per_class",
                    "save_visualizations",
                    "resize_perturbation",
                    "save_perturbations",
                    "shuffle_data",
                    "max_data_per_label",
                    "random_seed",
                ]
                for k, v in current_args.items():
                    if k in ignored_keys:
                        continue
                    if k not in cached_args:
                        same_config = False
                        continue

                    if isinstance(current_args[k], dict):
                        x = current_args[k]
                        y = cached_args[k]
                        diff_items = {k: x[k] for k in x if k in y and x[k] != y[k]}
                        if len(diff_items) > 0:
                            same_config = False
                    else:
                        if current_args[k] != cached_args[k]:
                            same_config = False

                if same_config:
                    logger.info(
                        f"Cached checkpoint for the same configuration found: {str(result_path)}: {cached_result_loaded['id']}"
                    )
                    checkpoint = int(result_path.with_suffix("").name.split("_")[-1])
                    cached_result = cached_result_loaded

            if cached_result is None:
                cached_result = {}
                cached_result["id"] = uuid.uuid4()
                cached_result["config"] = dataclasses.asdict(self.analysis_task_args)
                cached_result["samples"] = [None for _ in range(len(shap_data_paths))]

            if self.analysis_task_args.shuffle_data:
                random.Random(0).shuffle(shap_data_paths)

            data_per_label = {}
            if self.analysis_task_args.max_data_per_label > 0:
                for label in range(self.num_labels):
                    data_per_label[label] = 0

            pbar = tqdm.tqdm(shap_data_paths, mininterval=5.0, miniters=0)
            for idx, data_path in enumerate(pbar):
                try:
                    # only get the correct samples
                    save_result = False
                    if "correct/" in str(data_path):
                        perturbation_results = None
                        sample = None
                        max_samples_reached = False
                        if self.analysis_task_args.max_data_per_label > 0:
                            sample = self.data_cachers["pickle"].load_data(data_path)
                            if (
                                data_per_label[sample["label"]]
                                >= self.analysis_task_args.max_data_per_label
                            ):
                                max_samples_reached = True

                        perturbation_saved = True
                        if self.analysis_task_args.save_perturbations:
                            results_id = cached_result["id"]
                            data_post_string = Path(
                                str(data_path).split("shap_values/")[1]
                            )
                            output_path = (
                                self.output_dir
                                / f"feature_perturbation/{results_id}"
                                / data_post_string.parent
                            )
                            for k in [
                                "black_perturbation_mask",
                                "white_perturbation_mask",
                            ]:
                                k_output_path = Path(f"{str(output_path)}_{k}.png")
                                if not k_output_path.exists():
                                    perturbation_saved = False

                        if not perturbation_saved or (
                            not max_samples_reached
                            and cached_result["samples"][idx] is None
                        ):
                            if sample is None:
                                sample = self.data_cachers["pickle"].load_data(
                                    data_path
                                )
                            if perturbation_results is None:
                                perturbation_results = self.generate_perturbation(
                                    sample
                                )
                            cached_sample = {}
                            for k, v in sample.items():
                                if k in ["image_file_path", "label", "logits", "pred"]:
                                    cached_sample[k] = v
                            perturbed_class = perturbation_results.pred.argmax(dim=1)
                            perturbed_class_score = perturbation_results.pred[
                                :, perturbed_class.item()
                            ].item()
                            cached_sample[
                                "missclassified"
                            ] = perturbation_results.missclassified
                            cached_sample["perturbed_class"] = perturbed_class
                            cached_sample[
                                "perturbed_class_score"
                            ] = perturbed_class_score
                            cached_sample[
                                "percent_perturbation"
                            ] = perturbation_results.percent_perturbation
                            cached_sample[
                                "true_class_confidence_scores"
                            ] = perturbation_results.true_class_confidence_scores
                            cached_sample[
                                "successful_index"
                            ] = perturbation_results.successful_index
                            for k, v in cached_sample.items():
                                if isinstance(v, Tensor):
                                    cached_sample[k] = v.tolist()
                            cached_result["samples"][idx] = cached_sample
                            if self.analysis_task_args.max_data_per_label > 0:
                                data_per_label[sample["label"]] += 1
                            save_result = True

                            if self.analysis_task_args.save_perturbations:
                                results_id = cached_result["id"]
                                data_post_string = Path(
                                    str(data_path).split("shap_values/")[1]
                                )
                                output_path = (
                                    self.output_dir
                                    / f"feature_perturbation/{results_id}"
                                    / data_post_string.parent
                                )
                                for (
                                    k,
                                    v,
                                ) in perturbation_results.perturbation_outputs.items():
                                    k_output_path = Path(f"{str(output_path)}_{k}.png")
                                    if k_output_path.exists():
                                        continue
                                    if not k_output_path.parent.exists():
                                        k_output_path.parent.mkdir(parents=True)
                                    plt.imsave(
                                        k_output_path, v.cpu().numpy(), cmap="gray"
                                    )

                            if self.analysis_task_args.save_visualizations:
                                results_id = cached_result["id"]
                                data_post_string = Path(
                                    str(data_path).split("shap_values/")[1]
                                )
                                output_path = (
                                    self.output_dir
                                    / f"feature_perturbation/{results_id}"
                                    / data_post_string.parent
                                )
                                self.save_visualization(
                                    sample, perturbation_results, output_path
                                )
                        else:
                            if self.analysis_task_args.max_data_per_label > 0:
                                if cached_result["samples"][idx] is not None:
                                    data_per_label[sample["label"]] += 1

                    if idx % 100 == 0 and save_result:
                        results_id = cached_result["id"]
                        output_cache_path = (
                            results_root_path / f"{results_id}/results_{checkpoint}"
                        )
                        self.data_cachers["pickle"].save_data(
                            cached_result, output_cache_path, add_suffix=True
                        )
                        checkpoint += 1
                except Exception as e:
                    logger.exception(f"Exception raised at {idx}: {e}")
                    exit(1)

            results_id = cached_result["id"]
            output_cache_path = results_root_path / f"{results_id}/results_{checkpoint}"
            self.data_cachers["pickle"].save_data(
                cached_result, output_cache_path, add_suffix=True
            )
            self.data_cachers["json"].save_data(
                cached_result["config"], output_cache_path, add_suffix=True
            )

    def save_visualization(self, sample, perturbation_results, output_path):
        perturbed_image = perturbation_results.perturbed_image
        perturbation_outputs = perturbation_results.perturbation_outputs
        percent_perturbation = perturbation_results.percent_perturbation
        pred = perturbation_results.pred

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
        confidence_scores = F.softmax(torch.tensor(sample["logits"]), dim=-1).tolist()

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
        xlabel += f"Percentage perturbation: {percent_perturbation}\n"
        ax.set_xlabel(xlabel)
        ax.imshow(perturbed_image, cmap="gray")
        ax.set_aspect("auto")

        ax = fig.add_subplot(gs[3])
        # ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"Perturbed image masked")
        ax.imshow(image, cmap="gray", alpha=1.0)
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

        plt.show()
        plt.savefig(f"{output_path}.png", format="png", dpi=300)
