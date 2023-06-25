# Code and Dataset of the Paper: Efficient Non-Invasive FHB Estimation using RGB Images from a Novel Multi-Year, Multi-Rater Dataset


## Table of Contents

- [Dataset](#dataset)
  - [Download](#download)
  - [Data Overview](#data-overview)
- [Model](#model)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Citation](#citation)
## Dataset

### Download

The dataset can be downloaded using the following link: [Dataset](https://zenodo.org/record/8079099)

After the download extract the tar.gz file (Ubuntu) using the following command:

`tar -xf FHB_dataset.tar.gz`


### Data Overview

The dataset contains all images used to train and evaluate our model. We provide all training, validation, and testing splits as they were used in the paper.

The structure of the dataset looks as follows:

    .
    ├── 2020
    │   ├── Cam1
    │   │   ├── *.JPG
    │   ├── Cam2
    │   │   ├── *.JPG
    ├── 2021
    │   ├── ... (see structure of 2020)
    ├── 2022
    │   ├── ... (see structure of 2020)
    ├── expert_annotations
    │   ├── 2020
    │   │   ├── raterX_meta.txt
    │   │   ├── raterX_annotations.txt
    │   │   ├── raterX_annotations_*_stratified.txt
    │   │   ├── Cam1
    │   │   │   ├── raterX_meta.txt
    │   │   │   ├── raterX_annotations.txt
    │   │   │   ├── raterX_annotations_*_stratified.txt
    │   │   ├── Cam2
    │   │   │   ├── ... (see structure of expert annotations / 2020 / Cam1)
    │   ├── 2021
    │   │   ├── ... (see structure of expert annotations / 2020)
    │   ├── 2022
    │   │   ├── ... (see structure of expert annotations / 2020)
    │   ├── raterX_meta.txt
    │   ├── raterX_annotations.txt
    │   ├── raterX_annotations_*_stratified.txt


Especially in the image data of data year 2020 are images with a lot of black pixels. We never used them for training, evaluation and testing, so all annotation files with the *train, val, and test* tag are without these images.
Nevertheless, the overall annotation files (e.g. rater1_annotations.txt) includes these images.
You can use them to create your own splits or for any other purpose where you want to include the images with a lot of black pixels.


## Model

Before training and evaluation please use the packages listed in the *requirements.txt*.

### Training

Please use the config format displayed in *default_config.py* to start the training with a custom configuration.

Afterwards, manually change the code inside *src/train_model.py* to use the new config file (see section *__name__ == '__main__'*).

Please also adapt the location of the dataset inside the file before running the script.

To start the training go to the parent folder structure of the project and run

`python src/train_model.py`

The training process will automatically create folders to save the model configuration and weights (for evaluation purposes).

### Evaluation

To evaluate your model use the file *src/eval_model.py*.
Manually adapt the section *__name__ == '__main__'* of the file to run the evaluation with correct parameters.

The evaluation script automatically creates all results for the metrics described in the corresponding publication.

To start the evaluation go to the parent folder structure of the project and run

`python src/eval_model.py`


## Citation

If you use this project or dataset in your research, please consider citing it using the following BibTeX entry:

```bibtex
@article{
	RoessleFHBClassification2023,
	author = {Dominik Rößle and Lukas Prey and Ludwig Ramgraber and Anja Hanemann and Daniel Cremers and Patrick Ole Noack and Torsten Schön },
	title = {Efficient Non-Invasive FHB Estimation using RGB Images from a Novel Multi-Year, Multi-Rater Dataset},
	journal = {Plant Phenomics},
	year = {2023},
	doi = {10.34133/plantphenomics.0068},
	URL = {https://spj.science.org/doi/abs/10.34133/plantphenomics.0068}
}
