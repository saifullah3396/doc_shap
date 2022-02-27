"""
The main script that serves as the entry-point for all kinds of training experiments.
"""


import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
import tqdm
from das.data.data_args import DataArguments
from das.model_analyzer.analysis_tasks.base import AnalysisTask
from das.model_analyzer.analyzer_args import AnalysisTaskArguments, AnalyzerArguments
from das.model_analyzer.utils import annotate_heatmap, heatmap
from das.models.model_args import ModelArguments
from das.utils.basic_args import BasicArguments
from das.utils.basic_utils import create_logger
from das.utils.evaluation import evaluate_clustering
from das.utils.metrics import TrueLabelConfidence
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold._t_sne import TSNE

# setup logging
logger = create_logger(__name__)


class SimilarImagesClusteringTask(AnalysisTask):
    def __init__(
        self,
        basic_args: BasicArguments,
        data_args: DataArguments,
        model_args: ModelArguments,
        analyzer_args: AnalyzerArguments,
        analysis_task_args: AnalysisTaskArguments,
    ) -> None:
        super().__init__(
            "similar_images_clustering",
            basic_args,
            data_args,
            model_args,
            analyzer_args,
            analysis_task_args,
        )

    def generate_features(self):
        # get data collator required for the model
        self.datamodule.collate_fns = self.model.get_data_collators(
            self.data_args, None
        )

        # setup the model for feature extraction
        features_layer = self.model.features_layer()

        # set up features layer output hook
        def layer_output_hook(outputs={}):
            def hook(module, input, output):
                outputs["result"] = output

            return hook

        feature_layer_output = {}
        if features_layer is not None:
            features_layer.register_forward_hook(
                layer_output_hook(feature_layer_output)
            )

        # get predictions over the test set
        logger.info("Generating features from model...")
        features_list = []
        target_labels = []
        with torch.no_grad():
            for batch in tqdm.tqdm(self.datamodule.test_dataloader()):
                for kk, vv in batch.items():
                    batch[kk] = vv.cuda()
                self.model(**batch)
                features_list.append(
                    torch.flatten(feature_layer_output["result"], start_dim=1).cpu()
                )
                target_labels.append(batch["label"].cpu())
        features_list = torch.cat(features_list).numpy()
        target_labels = torch.cat(target_labels)

        # setup pca if required
        if self.analysis_task_args.dim_reduction_method == "pca":
            pca = PCA(
                **self.analysis_task_args.dim_reduction_args,
                random_state=self.basic_args.seed,
            )
            pca.fit(features_list)
            features_list = pca.transform(features_list)
        elif self.analysis_task_args.dim_reduction_method == "tsne":
            tsne = TSNE(**self.analysis_task_args.dim_reduction_args)
            features_list = tsne.fit_transform(features_list)

        return features_list, target_labels

    def cluster_features(self, features_list):
        kmeans = KMeans(
            n_clusters=self.num_labels, n_jobs=-1, random_state=self.basic_args.seed
        )
        kmeans.fit(features_list)
        pred_labels = kmeans.labels_

        # holds the cluster id and the images { id: [images] }
        clusters = {}
        for idx, cluster in enumerate(kmeans.labels_):
            if cluster not in clusters.keys():
                clusters[cluster] = []
                clusters[cluster].append(idx)
            else:
                clusters[cluster].append(idx)
        return clusters, pred_labels

    def generate_metrics(self, target_labels, pred_labels):
        return evaluate_clustering(
            target_labels,
            torch.from_numpy(pred_labels).to(target_labels.device),
            calc_acc=True,
        )

    def visualize_clusters(self, clusters):
        dataset = self.datamodule.test_dataset
        for k, v in clusters.items():
            fig = plt.figure(figsize=(25, 25))
            files = [dataset[idx]["image_file_path"] for idx in v]
            labels = [dataset[idx]["label"] for idx in v]
            if len(files) > 10:
                files = files[:10]
            for idx, file in enumerate(files):
                print(labels[idx])
                ax = fig.add_subplot(5, 5, idx + 1)
                ax.set_xlabel(f"{labels[idx]}")
                img = plt.imread(file)
                plt.imshow(img)
                plt.axis("off")
            plt.show()

    def visualize_tsne(self, features_list):
        logger.info("Visualizing TSNE features...")

        # scale and move the coordinates so they fit [0; 1] range
        def scale_to_01_range(x):
            # compute the distribution range
            value_range = np.max(x) - np.min(x)

            # move the distribution so that it starts from zero
            # by extracting the minimal value from all its values
            starts_from_zero = x - np.min(x)

            # make the distribution fit [0; 1] by dividing by its range
            return starts_from_zero / value_range

        # extract x and y coordinates representing the positions of the images on T-SNE plot
        tx = features_list[:, 0]
        ty = features_list[:, 1]

        tx = scale_to_01_range(tx)
        ty = scale_to_01_range(ty)

        # initialize a matplotlib plot
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # for every class, we'll add a scatter plot separately
        dataset = self.datamodule.test_dataset
        labels = [dataset[idx]["label"] for idx in range(features_list.shape[0])]
        for label in range(len(self.labels)):
            # find the samples of the current class in the data
            indices = [i for i, l in enumerate(labels) if l == label]

            # extract the coordinates of the points of this class only
            current_tx = np.take(tx, indices)
            current_ty = np.take(ty, indices)

            # add a scatter plot with the corresponding color and label
            ax.scatter(
                current_tx,
                current_ty,
                c=np.random.rand(
                    3,
                ),
                label=self.labels[label],
            )

        # build a legend using the labels we set previously
        ax.legend(loc="best")

        # finally, show the plot
        plt.show()

    def run(self):
        # load data only from pickle format
        clustering_results = self.data_cachers["pickle"].load_data_from_cache(
            self.task_output_name
        )
        clustering_results = {} if clustering_results is None else clustering_results

        method = self.analysis_task_args.dim_reduction_method
        if method not in clustering_results:
            clustering_results[method] = {}
            features_list, target_labels = self.generate_features()
            if method == "pca":
                clusters, pred_labels = self.cluster_features(
                    features_list=features_list
                )
                clustering_results[method]["clusters"] = clusters
                clustering_results[method]["pred_labels"] = pred_labels

                if self.analysis_task_args.generate_metrics:
                    (
                        nmi,
                        ami,
                        ari,
                        fscore,
                        adjacc,
                        match,
                        reordered_preds,
                    ) = self.generate_metrics(target_labels, pred_labels)
                    clustering_results[method]["nmi"] = nmi
                    clustering_results[method]["ami"] = ami
                    clustering_results[method]["ari"] = ari
                    clustering_results[method]["fscore"] = fscore
                    clustering_results[method]["adjacc"] = adjacc
                    clustering_results[method]["match"] = match
                    clustering_results[method]["reordered_preds"] = reordered_preds

                    logger.info("Results: ")
                    print("Method: ", method)
                    for k, v in clustering_results[method].items():
                        if k in ["clusters", "pred_labels", "match", "reordered_preds"]:
                            continue
                        print(f"{k} = {v}")

            # save in readable json format as well as pickle format
            clustering_results[method]["features_list"] = features_list
            self.data_cachers["pickle"].save_data_to_cache(
                clustering_results, self.task_output_name
            )

        if self.analysis_task_args.visualize_clusters:
            if method == "pca":
                self.visualize_clusters(clustering_results[method]["clusters"])
            elif method == "tsne":
                self.visualize_tsne(clustering_results[method]["features_list"])
