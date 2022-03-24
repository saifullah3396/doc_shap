# SHAP for Document Interpretability
This repository contains the evaluation code for the paper [The Reality of High Performing Deep Learning Models: A Case Study on Document Image Classification](https://www.techrxiv.org/articles/preprint/The_Reality_of_High_Performing_Deep_Learning_Models_A_Case_Study_on_Document_Image_Classification/19310537) by Saifullah, Stefan Agne, Andreas Dengel, and Sheraz Ahmed.

Requires Python 3+. For evaluation, please follow the steps below.

# Clone the repository and its submodules
```
git clone --recurse-submodules https://github.com/saifullah3396/doc_shap.git
```

# Requirements
Please install the requirements with pip as follows:
```
pip install -r requirements.txt
```

Set PYTHONPATH to match source the directory:
```
export PYTHONPATH=`pwd`/src
```

Create output directory for holding dataset, models, etc
```
export OUTPUT=</path/to/output/>
mkdir -p $OUTPUT
```

# Prepare the RVL-CDIP dataset:
Please download the [RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/) dataset and put it in the any directory `</path/to/rvlcdip_dataset>`.

# Generating SHAP Visualizations and Counterfactuals
To generate the SHAP values, visualizations and counterfactuals on RVL-CDIP dataset. Just run the following:
```
./scripts/analyze.sh --cfg ./cfg/alexnet.yaml data_args.dataset_dir </path/to/rvlcdip_dataset>
```
This will save the model results, confusion matrices, SHAP visualizations, and counterfactual visualizations and maps in the output/ directory. We only provide script for AlexNet and for a few samples just to demonstrate our approach.


# Citation
If you find this useful in your research, please consider citing our associated paper:
```
., Saifullah; Agne, Stefan; Dengel, Andreas; Ahmed, Sheraz (2022): The Reality of High Performing Deep Learning Models: A Case Study on Document Image Classification. TechRxiv. Preprint. https://doi.org/10.36227/techrxiv.19310537.v2
```

# License
This repository is released under the Apache 2.0 license as found in the LICENSE file.
