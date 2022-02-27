import sys
from importlib import import_module

from das.utils.basic_utils import create_logger

logger = create_logger(__name__)


def import_task(task_name):
    tasks_base_path = "das.model_analyzer.analysis_tasks"
    return import_module(f"{tasks_base_path}.{task_name}")


SUPPORTED_TASKS = {
    "generate_metrics": lambda: import_task("generate_metrics").GenerateMetricsTask,
    "generate_robustness_metrics": lambda: import_task(
        "generate_robustness_metrics"
    ).GenerateRobustnessMetricsTask,
    "generate_shap_values": lambda: import_task(
        "generate_shap_values"
    ).GenerateShapValuesTask,
    "generate_shap_visualizations": lambda: import_task(
        "generate_shap_visualizations"
    ).GenerateShapVisualizationsTask,
    "feature_perturbation": lambda: import_task(
        "feature_perturbation"
    ).FeaturePerturbationTask,
    "similar_images_clustering": lambda: import_task(
        "similar_images_clustering"
    ).SimilarImagesClusteringTask,
    "feature_perturbation_analysis": lambda: import_task(
        "feature_perturbation_analysis"
    ).FeaturePerturbationAnalysisTask,
}


class AnalysisTaskFactory:
    @staticmethod
    def get_task(analysis_task_name: str):
        """
        Returns the model class if present

        Args:
            analysis_task: The analysis task
        """
        try:
            analysis_task = SUPPORTED_TASKS.get(analysis_task_name, None)
            if analysis_task is None:
                raise ValueError(
                    f"Analysis task {analysis_task_name} is not supported!"
                )
            return analysis_task()
        except Exception as exc:
            logger.exception(
                f"Exception raised while loading analysis_task "
                f"[{analysis_task_name}]: {exc}"
            )
            sys.exit(1)
