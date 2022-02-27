"""
Defines the dataclass for holding training related arguments.
"""

import json
import math
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from das.models.model_args import ModelArguments
from das.utils.basic_utils import create_logger

logger = create_logger(__name__)


@dataclass
class AnalysisTaskArguments:
    task_name: str = ""


@dataclass
class GenerateMetricsTaskArguments(AnalysisTaskArguments):
    task_name: str = "generate_metrics"
    metrics: List = field(default_factory=lambda: ["accuracy"])


@dataclass
class GenerateRobustnessMetricsTaskArguments(AnalysisTaskArguments):
    task_name: str = "generate_robustness_metrics"
    baseline_metrics_path: str = ""


@dataclass
class SimilarImagesClusteringTaskArguments(AnalysisTaskArguments):
    task_name: str = "similar_images_clustering"
    dim_reduction_method: str = "pca"
    dim_reduction_args: dict = field(default_factory=lambda: {"n_components": 128})
    generate_metrics: bool = True
    visualize_clusters: bool = False


@dataclass
class GenerateShapValuesTaskArguments(AnalysisTaskArguments):
    task_name: str = "generate_shap_values"
    analyze_complete_dataset: bool = False
    num_test_samples_per_class: int = 1
    shap_num_bg_samples: int = 100
    start_idx: int = 0
    end_idx: int = 999999
    bg_name: str = "shap_background"
    save_bg_to_cache: bool = True
    save_samples_to_cache: bool = True
    load_bg_from_cache: bool = True
    load_samples_from_cache: bool = True
    ranked_outputs: Optional[int] = None
    only_get_true_shap_value: bool = False
    only_get_pred_shap_value: bool = False
    get_true_and_pred_shap_value: bool = False


@dataclass
class GenerateShapVisualizationsTaskArguments(AnalysisTaskArguments):
    task_name: str = "generate_shap_visualizations"
    resize_shap: bool = True


@dataclass
class FeaturePerturbationAttackConfig:
    arg_name: str = "dropout_pixels"
    mode: str = "linear"
    arg_min: int = 0
    arg_max: Optional[int] = 0
    arg_step: int = 4


@dataclass
class FeaturePerturbationTaskArguments(AnalysisTaskArguments):
    task_name: str = "feature_perturbation"
    feature_importance_grid_size: int = 4
    black_and_white_threshold: int = 125
    importance_order: str = "descending"
    max_perturbation_percentage: float = 0.05
    attack_type: str = "black_white_pixel_dropout"
    attack_config: FeaturePerturbationAttackConfig = FeaturePerturbationAttackConfig()
    save_visualizations: bool = True
    save_perturbations: bool = True
    n_vis_per_class: int = 100
    resize_perturbation: bool = True
    shuffle_data: bool = False
    max_data_per_label: int = -1
    random_seed: int = 0


@dataclass
class FeaturePerturbationAnalysisTaskArguments(AnalysisTaskArguments):
    most_relevant_first_data: str = ""
    least_relevant_first_data: str = ""
    random_data: List = field(default_factory=lambda: [])
    task_name: str = "feature_perturbation_analysis"


SUPPORTED_MODEL_ARGUMENTS = {
    "generate_metrics": GenerateMetricsTaskArguments,
    "generate_robustness_metrics": GenerateRobustnessMetricsTaskArguments,
    "generate_shap_values": GenerateShapValuesTaskArguments,
    "generate_shap_visualizations": GenerateShapVisualizationsTaskArguments,
    "feature_perturbation": FeaturePerturbationTaskArguments,
    "similar_images_clustering": SimilarImagesClusteringTaskArguments,
    "feature_perturbation_analysis": FeaturePerturbationAnalysisTaskArguments,
}


class AnalysisTaskArgumentsFactory:
    @staticmethod
    def create_child_arguments(task_name: str):
        """
        Returns the analysis_task arguments class if present

        Args:
            task_name: The task name for which the configuration arguments are to
                be returned.
        """
        try:
            model_args_class = SUPPORTED_MODEL_ARGUMENTS.get(task_name, None)
            if model_args_class is None:
                raise ValueError(f"Analysis task {task_name} is not supported!")
            return model_args_class
        except Exception as exc:
            logger.exception(
                f"Exception raised while loading analysis task arguments "
                f"[{task_name}]: {exc}"
            )
            sys.exit(1)


@dataclass
class AnalyzerArguments:
    """
    Arguments related to the training loop.
    """

    cls_name = "analyzer_args"

    analyzer_output_dir: str
    analysis_tasks: List[AnalysisTaskArguments]
    models: Union[List[ModelArguments], ModelArguments]
    output_data_subdir: str = ""

    def __post_init__(self):
        pass
