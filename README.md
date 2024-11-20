# CAALM
Repository for Combining Autoregressive and Autoencoder Language Models for Text Classification

## Introduction

This repository contains the code and data for the paper:

**"Combining Autoregressive and Autoencoder Language Models for Text Classification"**

Author: João Gonçalves


## Repository Structure

Folders:

- Corona/: Contains data and scripts related to the CoronaNet dataset experiments.
- Hate/: Contains data and scripts for the hate speech classification experiments.
- Military/: Contains data and scripts for the military stance detection experiments.
- Morality/: Contains data and scripts for the morality stance detection experiments.
  
Scripts and Files:

- analysis_script_updated.R: R script for analyzing baseline experimental results.
- nemo_analysis.R: R script for analyzing results with Nemo labels only.
- autoregressive_generation.py: Python script for generating intermediate texts using an autoregressive model.
- test_loop_BERT.py: Python script for training and evaluating BERT-based models.
- test_loop_BERT_NLI.py: Python script for training and evaluating BERT-NLI models.
- all_results.csv, nemo_results.csv, zero_shot_results.csv: CSV files containing experimental results.
- *.png: Plot images visualizing the results.

## Datasets
The following datasets are used in this project:

1. **CoronaNet Dataset**:
- Source: https://www.nature.com/articles/s41562-020-0909-7

2. **Hate Speech Dataset**:
- Source: https://ojs.aaai.org/index.php/ICWSM/article/view/14955

3. **Military and Traditional Morality Stance Detection Datasets**:
- Source: https://manifesto-project.wzb.eu/information/documents/corpus


## Usage
Generating intermediate texts can be done by running autoregressive_generation.py. Currently, the script is configured to take Mistral Nemo as the autoregressive model and use the hate speech detection dataset. Usage of different models, datasets, and classification instructions needs to be edited in the file directly.

The test_loop Python files replicate the analyses in the paper for baseline BERT models and NLI models. They can be adjusted to classify other datasets with CAALM generated intermediate texts.

## Funding

This research was funded by a VENI grant VI.Veni.221S.154 from the Dutch Research Council (NWO). The funding sources had no involvement in the study design; in the collection, analysis and interpretation of data; in the writing of the report; and in the decision to submit the article for publication.

## Acknowledgements

The scripts and testing approach in this repository draw significantly from https://github.com/MoritzLaurer/less-annotating-with-bert-nli.

Laurer, M., Van Atteveldt, W., Casas, A., & Welbers, K. (2023). Less Annotating, More Classifying: Addressing the Data Scarcity Issue of Supervised Machine Learning with Deep Transfer Learning and BERT-NLI. Political Analysis, 1–33. https://doi.org/10.1017/pan.2023.20

OpenAI's o1-preview model was used to speed up commenting and streamlining of the code in this repository to provide contextual information and make it more accessible. It was not used for the research paper.

## To do

1. Create demo file that makes the CAALM pipeline accessible for any user defined datasets, models and classification task.




