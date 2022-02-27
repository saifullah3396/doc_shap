"""
The main script that serves as the entry-point for all kinds of training experiments.
"""

import copy
import pickle

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torchmetrics
import tqdm
from das.data.augmentations.factory import AugmentationsEnum
from das.data.data_args import DataArguments
from das.data.data_modules.factory import DataModuleFactory
from das.data.datasets.datadings_writer import CustomFileWriter
from das.data.datasets.utils import DataKeysEnum
from das.model_analyzer.analysis_tasks.base import AnalysisTask
from das.model_analyzer.analyzer_args import AnalysisTaskArguments, AnalyzerArguments
from das.model_analyzer.utils import annotate_heatmap, heatmap
from das.models.model_args import ModelArguments
from das.utils.basic_args import BasicArguments
from das.utils.basic_utils import create_logger
from das.utils.metrics import TrueLabelConfidence
from datadings.reader.msgpack import MsgpackReader
from pytorch_lightning.core import datamodule
from torchmetrics.metric import Metric

# setup logging
logger = create_logger(__name__)


class GenerateRobustnessMetricsTask(AnalysisTask):
    def __init__(
        self,
        basic_args: BasicArguments,
        data_args: DataArguments,
        model_args: ModelArguments,
        analyzer_args: AnalyzerArguments,
        analysis_task_args: AnalysisTaskArguments,
    ) -> None:
        super().__init__(
            "model_robustness_metrics",
            basic_args,
            data_args,
            model_args,
            analyzer_args,
            analysis_task_args,
        )

    def run(self):
        # load data only from pickle format
        results = self.data_cachers["pickle"].load_data_from_cache(
            self.task_output_name
        )
        results = {} if results is None else results

        baseline_results = None
        if self.analysis_task_args.baseline_metrics_path != "":
            with open(self.analysis_task_args.baseline_metrics_path, "rb") as f:
                baseline_results = pickle.load(f)

        augmented_error_rates_baseline = None
        if baseline_results is not None:
            augmented_error_rates_baseline = baseline_results["augmented_error_rates"]
            clean_error_rate_baseline = baseline_results["clean_error_rate"]

        if (
            "base_corrects" not in results
            or not bool(results["base_corrects"])
            or "augmented_corrects" not in results
            or not bool(results["augmented_corrects"])
        ):
            test_results = self.test_model()
            results["base_corrects"] = test_results[0]
            results["augmented_corrects"] = test_results[1]
            self.data_cachers["pickle"].save_data_to_cache(
                results, self.task_output_name
            )

        # find clean error rate
        base_corrects = torch.tensor(results["base_corrects"])
        clean_error_rate = (1.0 - torch.sum(base_corrects) / len(base_corrects)).item()

        # find per augmentation, per severity error rates
        augmented_corrects = results["augmented_corrects"]
        augmented_error_rates = {}
        for k, v in augmented_corrects.items():
            if k not in augmented_error_rates:
                augmented_error_rates[k] = {}
            for kk, vv in v.items():
                # make sure all augmented images are present
                augmented_error_rates[k][kk] = (
                    1.0 - torch.sum(torch.tensor(vv)) / len(vv)
                ).tolist()

        results["clean_error_rate"] = clean_error_rate
        results["augmented_error_rates"] = augmented_error_rates

        logger.info(f"Clean Error Rate: {clean_error_rate}")
        mean_augmented_error_rates = {}
        for k, v in augmented_error_rates.items():
            mean_augmented_error_rates[k] = 0.0
            for kk, vv in v.items():
                mean_augmented_error_rates[k] += vv
            mean_augmented_error_rates[k] /= len(v.keys())

        logger.info("Mean Augmented Error Rates:")
        for idx, k in enumerate(list(AugmentationsEnum)):
            if idx in mean_augmented_error_rates:
                logger.info(f"{k}: {mean_augmented_error_rates[idx]}")

        results["mean_augmented_error_rates"] = mean_augmented_error_rates

        if augmented_error_rates_baseline is not None:
            per_aug_ce = {}
            per_aug_rel_ce = {}
            for a in range(len(AugmentationsEnum)):
                if a in augmented_error_rates:
                    per_sev_sum = 0.0
                    per_sev_sum_baseline = 0.0
                    per_sev_rel_sum = 0.0
                    per_sev_rel_sum_baseline = 0.0
                    for s in [1, 2, 3, 4, 5]:
                        if s in augmented_error_rates[a]:
                            per_sev_sum += augmented_error_rates[a][s]
                            per_sev_sum_baseline += augmented_error_rates_baseline[a][s]
                            per_sev_rel_sum += (
                                augmented_error_rates[a][s] - clean_error_rate
                            )
                            per_sev_rel_sum_baseline += (
                                augmented_error_rates_baseline[a][s]
                                - clean_error_rate_baseline
                            )

                    per_aug_ce[a] = per_sev_sum / per_sev_sum_baseline
                    per_aug_rel_ce[a] = per_sev_rel_sum / per_sev_rel_sum_baseline

                    # print(a, per_sev_sum, per_sev_sum_baseline)
                    # print(
                    #     a, per_sev_rel_sum, per_sev_rel_sum_baseline, per_aug_rel_ce[a]
                    # )
                    # print(augmented_error_rates[a], augmented_error_rates_baseline[a])

            mce = 0.0
            for v in per_aug_ce.values():
                mce += v
            mce /= len(per_aug_ce.values())
            results["per_aug_ce"] = per_aug_ce
            results["mce"] = mce

            rel_mce = 0.0
            for v in per_aug_rel_ce.values():
                rel_mce += v
            rel_mce /= len(per_aug_rel_ce.values())

            results["per_aug_rel_ce"] = per_aug_rel_ce
            results["rel_mce"] = rel_mce

            print(f"{self.model_args.model_name} Results: ")
            print("Clean Error Rate: ", clean_error_rate)
            print("Mean Corruption Error (MCE): ", mce)
            print("Relative Mean Corruption Error (MCE): ", rel_mce)

        self.data_cachers["pickle"].save_data_to_cache(results, self.task_output_name)

    def test_model(self):
        data_key_type_map = {
            DataKeysEnum.AUGMENTATION: torch.long,
            DataKeysEnum.SEVERITY: torch.long,
        }

        # get data collator required for the model
        self.datamodule.collate_fns = self.model.get_data_collators(
            self.data_args, None, data_key_type_map=data_key_type_map
        )

        cached_data_size = 0
        cache_preds_file = self.output_dir / "preds2.msgpack"
        if cache_preds_file.exists():
            data_reader = MsgpackReader(cache_preds_file)
            cached_data_size = len(data_reader)

        if cached_data_size < len(self.datamodule.test_dataloader()):
            with CustomFileWriter(cache_preds_file, overwrite=False) as writer:
                progress = enumerate(
                    tqdm.tqdm(
                        self.datamodule.test_dataloader(),
                        total=len(self.datamodule.test_dataloader()),
                    )
                )
                for index, batch in progress:
                    if index < cached_data_size:
                        continue
                    for k, v in batch.items():
                        batch[k] = v.to(self.model.device)
                    output = self.model(**batch)
                    pred_labels, target_labels = self.model.get_pred_target_labels(
                        output, batch
                    )
                    correct_preds = pred_labels == target_labels
                    writer.write(
                        {
                            "key": str(index),
                            "aug": [aug.item() for aug in batch["augmentation"]],
                            "severity": [
                                severity.item() for severity in batch["severity"]
                            ],
                            "pred": correct_preds.tolist(),
                        }
                    )

        base_corrects = []
        augmented_corrects = {}
        for didx in range(cached_data_size):
            data = data_reader.get(didx)
            for idx in range(len(data["pred"])):
                pred = data["pred"][idx]
                aug = data["aug"][idx]
                severity = data["severity"][idx]
                if aug == -1:
                    base_corrects.append(pred)
                else:
                    if aug not in augmented_corrects:
                        augmented_corrects[aug] = {}
                    if severity not in augmented_corrects[aug]:
                        augmented_corrects[aug][severity] = []
                    augmented_corrects[aug][severity].append(pred)

        for k, v in augmented_corrects.items():
            for kk, vv in v.items():
                # make sure all augmented images are present
                assert len(vv) == len(base_corrects)
        return base_corrects, augmented_corrects
