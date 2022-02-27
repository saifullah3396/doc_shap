"""
The main script that serves as the entry-point for all kinds of training experiments.
"""


from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import shap
import torch
import tqdm
from das.data.data_args import DataArguments
from das.data.datasets.utils import DataKeysEnum
from das.model_analyzer.analysis_tasks.base import AnalysisTask
from das.model_analyzer.analyzer_args import AnalysisTaskArguments, AnalyzerArguments
from das.models.model_args import ModelArguments
from das.utils.basic_args import BasicArguments
from das.utils.basic_utils import create_logger
from torch.utils.data import Subset

# setup logging
logger = create_logger(__name__)


class GenerateShapValuesTask(AnalysisTask):
    def __init__(
        self,
        basic_args: BasicArguments,
        data_args: DataArguments,
        model_args: ModelArguments,
        analyzer_args: AnalyzerArguments,
        analysis_task_args: AnalysisTaskArguments,
    ) -> None:
        super().__init__(
            "shap_values",
            basic_args,
            data_args,
            model_args,
            analyzer_args,
            analysis_task_args,
        )

    def setup_output_dir(self):
        return super().setup_output_dir().parent

    def generate_data_samples(self):
        if self.analysis_task_args.load_samples_from_cache:
            data = self.data_cachers["pickle"].load_data_from_cache("shap_data_samples")
            if data is not None:
                return data

        # get data collator required for the model
        self.datamodule.collate_fns = self.model.get_data_collators(
            self.data_args,
            None,
            data_key_type_map={
                DataKeysEnum.INDEX: torch.long,
                DataKeysEnum.IMAGE: torch.float,
                DataKeysEnum.IMAGE_FILE_PATH: list,
                DataKeysEnum.LABEL: torch.long,
            },
        )

        # here we temporarily remove transform from the dataset and save it
        test_dataloader = self.datamodule.test_dataloader()

        # get class labels
        num_labels = self.datamodule.num_labels

        data = []
        finish_sampling = False
        for batch in tqdm.tqdm(test_dataloader):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()
            output = self.model(**batch)
            pred = output.logits.argmax(dim=1)

            samples = [dict(zip(batch, t)) for t in zip(*batch.values())]
            for idx, sample in enumerate(samples):
                to_explain_data_labels = np.array([d["label"] for d in data])
                num_label_samples = len(
                    to_explain_data_labels[
                        to_explain_data_labels == sample["label"].cpu().item()
                    ]
                )

                if (
                    num_label_samples
                    < self.analysis_task_args.num_test_samples_per_class
                ):
                    sample["logits"] = output.logits[idx]
                    sample["pred"] = pred[idx]
                    for k, v in sample.items():
                        if isinstance(v, torch.Tensor):
                            sample[k] = v.cpu()
                    data.append(sample)

                if (
                    len(data)
                    == self.analysis_task_args.num_test_samples_per_class * num_labels
                ):
                    finish_sampling = True

            if finish_sampling:
                break

        if self.analysis_task_args.save_samples_to_cache:
            self.data_cachers["pickle"].save_data_to_cache(data, "shap_data_samples")
        return data

    def generate_background(self):
        if self.analysis_task_args.load_bg_from_cache:
            data = self.data_cachers["pickle"].load_data_from_cache(
                f"{self.analysis_task_args.bg_name}"
            )
            if data is not None:
                return data

        # here we temporarily remove transform from the train dataset
        # and replace it with test set transform
        dataset = self.datamodule.train_dataset
        if isinstance(dataset, Subset):
            dataset.dataset.transforms = self.datamodule.test_dataset.dataset.transforms
        else:
            dataset.transforms = self.datamodule.test_dataset.transforms

        logger.info("Generating shap background data from train dataset...")
        indices = np.arange(0, len(dataset))
        np.random.shuffle(indices)
        data = {}
        for label in range(len(self.labels)):
            data[label] = []

        for idx in tqdm.tqdm(indices):
            sample = dataset[idx]
            label = sample["label"]
            if len(data[label]) < self.analysis_task_args.shap_num_bg_samples:
                data[label].append(sample)

            finish = np.array(
                [
                    len(v) == self.analysis_task_args.shap_num_bg_samples
                    for v in data.values()
                ]
            )
            if finish.all():
                break
        if self.analysis_task_args.save_bg_to_cache:
            self.data_cachers["pickle"].save_data_to_cache(
                data, f"{self.analysis_task_args.bg_name}"
            )
        return data

    def run(self):
        self.datamodule = self.setup_datamodule()
        print(
            "self.analysis_task_args.analyze_complete_dataset",
            self.analysis_task_args.analyze_complete_dataset,
        )
        if not self.analysis_task_args.analyze_complete_dataset:
            self.data_samples = self.generate_data_samples()
        self.background = self.generate_background()

        # print out the data and background samples information
        bg = np.array([len(v) for v in self.background.values()])
        logger.info(f"Number of background samples: {bg.sum().squeeze()}")
        if not self.analysis_task_args.analyze_complete_dataset:
            logger.info(f"Number of test samples: {len(self.data_samples)}")

        for k, v in self.background.items():
            logger.info(f"Background samples per {self.labels[k]} = {len(v)}")

        # first test the model on the total test dataset
        # get model
        self.model = self.setup_model()
        for module in self.model.modules():
            if isinstance(module, torch.nn.ReLU):
                module.inplace = False

        # get data collator required for the model
        collate_fn = self.model.get_data_collators(self.data_args, None)["test"]

        # generate collated data
        self.background_collated = {}
        for k, v in self.background.items():
            self.background_collated[k] = collate_fn(v)
            for kk, vv in self.background_collated[k].items():
                self.background_collated[k][kk] = vv.cuda()

        if not self.analysis_task_args.analyze_complete_dataset:
            self.data_samples_collated = collate_fn(self.data_samples)
            for kk, vv in self.data_samples_collated.items():
                self.data_samples_collated[kk] = vv.cuda()

        self.initialize_explainer()
        self.explain()

    def initialize_explainer(self):
        background_sampled = {}
        for v in self.background_collated.values():
            for kk, vv in v.items():
                if kk not in background_sampled:
                    background_sampled[kk] = []
                background_sampled[kk].append(vv)

        for k, v in background_sampled.items():
            background_sampled[k] = torch.cat(v)

        class ModelWrapper(torch.nn.Module):
            def __init__(self, model, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.model = model
                self.softmax = torch.nn.Softmax(dim=1)

            def forward(self, *args, **kwargs):
                return self.softmax(self.model(*args, **kwargs).logits)

        # wrap the model around the original model
        model_wrapper = ModelWrapper(self.model).cuda()

        # make the explainer
        self.explainer = shap.DeepExplainer(
            model_wrapper,
            background_sampled["image"],
            grad_batch_size=self.model_args.grad_batch_size,
        )

    def explain(self):
        if not self.analysis_task_args.analyze_complete_dataset:
            for idx in tqdm.tqdm(range(self.data_samples_collated["image"].shape[0])):
                sample = self.data_samples[idx]
                if self.shap_values_output_path(sample).exists():
                    continue

                # self.analysis_task_args.ranked_outputs = None
                if self.analysis_task_args.only_get_true_shap_value:
                    shap_values = self.explainer.shap_values(
                        self.data_samples_collated["image"][idx].unsqueeze(dim=0),
                        output_idx=sample["label"],
                    )
                elif self.analysis_task_args.only_get_pred_shap_value:
                    shap_values = self.explainer.shap_values(
                        self.data_samples_collated["image"][idx].unsqueeze(dim=0),
                        output_idx=sample["pred"],
                    )
                elif self.analysis_task_args.get_true_and_pred_shap_value:
                    shap_values_true = self.explainer.shap_values(
                        self.data_samples_collated["image"][idx].unsqueeze(dim=0),
                        output_idx=sample["label"],
                    )
                    shap_values_pred = self.explainer.shap_values(
                        self.data_samples_collated["image"][idx].unsqueeze(dim=0),
                        output_idx=sample["pred"],
                    )
                    shap_values = shap_values_true + shap_values_pred
                    indices = []
                    indices.append(sample["label"])
                    indices.append(sample["pred"])
                    sample["indices"] = indices
                elif self.analysis_task_args.ranked_outputs is not None:
                    shap_values, indices = self.explainer.shap_values(
                        self.data_samples_collated["image"][idx].unsqueeze(dim=0),
                        ranked_outputs=self.analysis_task_args.ranked_outputs,
                        check_additivity=True,
                    )  # , nsamples=200)
                    sample["indices"] = indices.squeeze().tolist()
                else:
                    shap_values = self.explainer.shap_values(
                        self.data_samples_collated["image"][idx].unsqueeze(dim=0)
                    )
                sample["shap_values"] = shap_values

                self.save_shap_values(sample)
        else:
            # get data collator required for the model
            collate_fn = self.model.get_data_collators(self.data_args, None)["test"]
            self.data_args.per_device_eval_batch_size = 1

            for idx in tqdm.tqdm(range(len(self.datamodule.test_dataset))):
                if (
                    idx < self.analysis_task_args.start_idx
                    or idx > self.analysis_task_args.end_idx
                ):
                    continue
                sample = self.datamodule.test_dataset[idx]
                shap_results = {}
                shap_results["image_file_path"] = sample.pop("image_file_path")
                shap_results["label"] = sample["label"]
                sample_collated = collate_fn([sample])
                for kk, vv in sample_collated.items():
                    sample_collated[kk] = vv.cuda()
                output = self.model(**sample_collated)
                shap_results["logits"] = output.logits
                shap_results["pred"] = output.logits.argmax(dim=1)[0]

                if self.shap_values_output_path(shap_results).exists():
                    continue

                # self.analysis_task_args.ranked_outputs = None
                if self.analysis_task_args.only_get_true_shap_value:
                    shap_values = self.explainer.shap_values(
                        sample_collated["image"],
                        output_idx=shap_results["label"],
                        check_additivity=False,
                    )
                elif self.analysis_task_args.only_get_pred_shap_value:
                    shap_values = self.explainer.shap_values(
                        sample_collated["image"],
                        output_idx=shap_results["pred"],
                        check_additivity=False,
                    )
                elif self.analysis_task_args.get_true_and_pred_shap_value:
                    shap_values_true = self.explainer.shap_values(
                        sample_collated["image"],
                        output_idx=shap_results["label"],
                        check_additivity=False,
                    )
                    shap_values_pred = self.explainer.shap_values(
                        sample_collated["image"], output_idx=shap_results["pred"]
                    )
                    shap_values = shap_values_true + shap_values_pred
                    indices = []
                    indices.append(shap_results["label"])
                    indices.append(shap_results["pred"])
                    shap_results["indices"] = indices
                elif self.analysis_task_args.ranked_outputs is not None:
                    shap_values, indices = self.explainer.shap_values(
                        sample_collated["image"],
                        ranked_outputs=self.analysis_task_args.ranked_outputs,
                        check_additivity=True,
                    )  # , nsamples=200)
                    shap_results["indices"] = indices.squeeze().tolist()
                else:
                    shap_values = self.explainer.shap_values(sample_collated["image"])
                shap_results["shap_values"] = shap_values

                self.save_shap_values(shap_results)

    def shap_values_output_path(self, shap_results):
        correct_or_wrong_path = (
            "correct" if shap_results["pred"] == shap_results["label"] else "wrong"
        )
        output_dir = (
            self.output_dir
            / "shap_values"
            / self.labels[shap_results["label"]]
            / correct_or_wrong_path
        )

        output_dir = (
            output_dir
            / Path(shap_results["image_file_path"]).with_suffix("").name
            / str(self.model.model_name + ".pickle")
        )

        return output_dir

    def save_shap_values(self, shap_results):
        if "shap_values" not in shap_results:
            return

        output_path = self.shap_values_output_path(shap_results)
        saved_data = {}
        for k, v in shap_results.items():
            if k in ["image_file_path", "label", "pred", "indices", "logits"]:
                saved_data[k] = v
            elif k == "shap_values":
                # sum the shap values here over channels to take less space when saving
                saved_data[k] = [sv.sum(1).squeeze() for sv in v]

        # save to cache
        self.data_cachers["pickle"].save_data(saved_data, output_path)
