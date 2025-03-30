# A Clinical Benchmark of Public Self-Supervised Pathology Foundation Models

Repository of training recipes for the manuscript: "A Clinical Benchmark of Public Self-Supervised Pathology Foundation Models".
Manuscript link: [arxiv](https://arxiv.org/abs/2407.06508)

## Abstract
The use of self-supervised learning (SSL) to train pathology foundation models has increased substantially in the past few years. Notably, several models trained on large quantities of clinical data have been made publicly available in recent months. This will significantly enhance scientific research in computational pathology and help bridge the gap between research and clinical deployment. With the increase in availability of public foundation models of different sizes, trained using different algorithms on different datasets, it becomes important to establish a benchmark to compare the performance of such models on a variety of clinically relevant tasks spanning multiple organs and diseases. In this work, we present a collection of pathology datasets comprising clinical slides associated with clinically relevant endpoints including cancer diagnoses and a variety of biomarkers generated during standard hospital operation from three medical centers. We leverage these datasets to systematically assess the performance of public pathology foundation models and provide insights into best practices for training new foundation models and selecting appropriate pretrained models. To enable the community to evaluate their models on our clinical datasets, we make available an automated benchmarking pipeline for external use.

## Leaderboard

Average AUC (standard deviation) across 20 MCCV splits. Models are ordered in increasing rank left to right. Updated 11/26/2024.

### Detection Tasks

| Task            | H-optimus-0   | Prov-GigaPath   | SP85M         | UNI           | Virchow2      | SP22M         | Phikon-v2     | Virchow       | Phikon        | CTransPath    | tRes50        |
|:----------------|:--------------|:----------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
| MSHS Bladder    | 0.963 (0.017) | 0.958 (0.016)   | 0.961 (0.015) | 0.954 (0.018) | 0.964 (0.016) | 0.954 (0.016) | 0.960 (0.015) | 0.949 (0.022) | 0.950 (0.017) | 0.950 (0.023) | 0.938 (0.027) |
| MSHS Breast     | 0.981 (0.007) | 0.979 (0.009)   | 0.981 (0.007) | 0.981 (0.007) | 0.978 (0.008) | 0.980 (0.009) | 0.978 (0.008) | 0.978 (0.006) | 0.977 (0.007) | 0.969 (0.011) | 0.932 (0.013) |
| MSHS Colorectal | 0.974 (0.029) | 0.974 (0.026)   | 0.973 (0.024) | 0.970 (0.028) | 0.970 (0.027) | 0.972 (0.023) | 0.969 (0.027) | 0.970 (0.030) | 0.966 (0.028) | 0.963 (0.027) | 0.950 (0.022) |
| MSHS DCIS       | 0.992 (0.006) | 0.983 (0.017)   | 0.985 (0.018) | 0.992 (0.010) | 0.989 (0.014) | 0.985 (0.012) | 0.989 (0.010) | 0.984 (0.015) | 0.988 (0.012) | 0.974 (0.025) | 0.951 (0.034) |
| MSHS IBD        | 0.980 (0.008) | 0.980 (0.006)   | 0.975 (0.009) | 0.973 (0.008) | 0.973 (0.010) | 0.975 (0.007) | 0.961 (0.011) | 0.965 (0.010) | 0.967 (0.010) | 0.956 (0.009) | 0.939 (0.020) |
| MSHS Kidney     | 0.973 (0.009) | 0.970 (0.009)   | 0.972 (0.009) | 0.971 (0.009) | 0.971 (0.010) | 0.970 (0.010) | 0.967 (0.008) | 0.965 (0.011) | 0.965 (0.008) | 0.962 (0.009) | 0.952 (0.012) |
| MSHS Oral       | 0.991 (0.018) | 0.993 (0.013)   | 0.987 (0.018) | 0.992 (0.016) | 0.994 (0.013) | 0.989 (0.013) | 0.990 (0.013) | 0.992 (0.014) | 0.992 (0.012) | 0.987 (0.012) | 0.966 (0.024) |
| MSHS Prostate   | 0.991 (0.005) | 0.991 (0.006)   | 0.992 (0.006) | 0.992 (0.005) | 0.990 (0.007) | 0.990 (0.006) | 0.988 (0.006) | 0.992 (0.005) | 0.989 (0.008) | 0.983 (0.011) | 0.985 (0.007) |
| MSHS Thyroid    | 0.975 (0.013) | 0.974 (0.011)   | 0.975 (0.014) | 0.977 (0.010) | 0.968 (0.015) | 0.973 (0.012) | 0.970 (0.010) | 0.975 (0.012) | 0.972 (0.011) | 0.971 (0.013) | 0.968 (0.013) |
| Overall         | 0.980         | 0.978           | 0.978         | 0.978         | 0.978         | 0.976         | 0.975         | 0.975         | 0.974         | 0.968         | 0.954         |

### Biomarker Tasks

| Task              | H-optimus-0   | Prov-GigaPath   | UNI           | Phikon        | Virchow2      | Phikon-v2     | SP85M         | SP22M         | Virchow       | CTransPath    | tRes50        |
|:------------------|:--------------|:----------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
| MSHS BCa ER       | 0.973 (0.008) | 0.972 (0.008)   | 0.970 (0.009) | 0.966 (0.009) | 0.970 (0.007) | 0.964 (0.008) | 0.965 (0.010) | 0.967 (0.009) | 0.968 (0.008) | 0.949 (0.009) | 0.912 (0.022) |
| MSHS BCa HER2     | 0.860 (0.026) | 0.831 (0.027)   | 0.830 (0.024) | 0.812 (0.033) | 0.836 (0.023) | 0.807 (0.029) | 0.802 (0.023) | 0.814 (0.021) | 0.825 (0.026) | 0.803 (0.023) | 0.772 (0.039) |
| MSHS BCa PR       | 0.937 (0.010) | 0.925 (0.011)   | 0.929 (0.012) | 0.925 (0.013) | 0.912 (0.011) | 0.913 (0.014) | 0.926 (0.013) | 0.926 (0.011) | 0.918 (0.012) | 0.896 (0.013) | 0.838 (0.027) |
| MSHS BioMe HRD    | 0.695 (0.152) | 0.741 (0.097)   | 0.727 (0.129) | 0.711 (0.117) | 0.703 (0.092) | 0.813 (0.104) | 0.718 (0.099) | 0.685 (0.113) | 0.702 (0.099) | 0.672 (0.140) | 0.545 (0.138) |
| MSHS LUAD EGFR    | 0.829 (0.047) | 0.821 (0.039)   | 0.797 (0.041) | 0.763 (0.053) | 0.792 (0.054) | 0.754 (0.052) | 0.745 (0.043) | 0.725 (0.059) | 0.767 (0.052) | 0.718 (0.051) | 0.594 (0.057) |
| MSKCC LUAD ALK    | 0.821 (0.044) | 0.813 (0.042)   | 0.789 (0.045) | 0.782 (0.046) | 0.778 (0.053) | 0.761 (0.040) | 0.736 (0.053) | 0.742 (0.049) | 0.747 (0.041) | 0.732 (0.049) | 0.625 (0.062) |
| MSKCC LUAD EGFR   | 0.823 (0.030) | 0.812 (0.028)   | 0.795 (0.039) | 0.760 (0.039) | 0.766 (0.038) | 0.765 (0.036) | 0.755 (0.034) | 0.753 (0.030) | 0.755 (0.042) | 0.739 (0.037) | 0.649 (0.031) |
| MSKCC LUAD KRAS   | 0.711 (0.031) | 0.730 (0.030)   | 0.716 (0.031) | 0.683 (0.029) | 0.694 (0.037) | 0.671 (0.018) | 0.677 (0.027) | 0.667 (0.020) | 0.658 (0.020) | 0.633 (0.032) | 0.546 (0.062) |
| MSKCC LUAD STK11  | 0.874 (0.030) | 0.868 (0.036)   | 0.843 (0.034) | 0.824 (0.032) | 0.808 (0.048) | 0.803 (0.038) | 0.813 (0.054) | 0.811 (0.049) | 0.785 (0.059) | 0.799 (0.042) | 0.657 (0.061) |
| MSKCC LUAD TP53   | 0.732 (0.031) | 0.757 (0.028)   | 0.746 (0.032) | 0.743 (0.031) | 0.706 (0.027) | 0.728 (0.030) | 0.705 (0.032) | 0.703 (0.028) | 0.702 (0.024) | 0.708 (0.029) | 0.658 (0.043) |
| MSKCC NSCLC IO    | 0.596 (0.069) | 0.562 (0.060)   | 0.607 (0.053) | 0.573 (0.047) | 0.578 (0.055) | 0.541 (0.070) | 0.503 (0.039) | 0.525 (0.058) | 0.527 (0.060) | 0.556 (0.044) | 0.537 (0.077) |
| SUH Melanoma BRAF | 0.705 (0.053) | 0.651 (0.052)   | 0.698 (0.051) | 0.668 (0.057) | 0.671 (0.063) | 0.671 (0.053) | 0.653 (0.052) | 0.650 (0.043) | 0.607 (0.052) | 0.655 (0.090) | 0.567 (0.060) |
| SUH Melanoma NRAS | 0.650 (0.049) | 0.625 (0.068)   | 0.608 (0.060) | 0.635 (0.070) | 0.611 (0.053) | 0.608 (0.056) | 0.633 (0.069) | 0.609 (0.062) | 0.590 (0.061) | 0.575 (0.055) | 0.530 (0.058) |
| Overall           | 0.785         | 0.778           | 0.773         | 0.757         | 0.756         | 0.754         | 0.741         | 0.737         | 0.735         | 0.726         | 0.648         |


## Methods

### Clinically Relevant Downstream Tasks

#### Detection Tasks

| Task      | Origin | Disease           | Slides | Scanner           |
| --------- | ------ | -------------     | -----: | ----------------- |
| Detection | MSHS   | Breast Cancer     | 1,998  | Philips Ultrafast |
| Detection | MSHS   | Oral Cancer       |   279  | Philips Ultrafast |
| Detection | MSHS   | Bladder Cancer    |   448  | Philips Ultrafast |
| Detection | MSHS   | Kidney Cancer     | 1,000  | Philips Ultrafast |
| Detection | MSHS   | Thyroid Cancer    |   710  | Philips Ultrafast |
| Detection | MSHS   | DCIS              |   233  | Philips Ultrafast |
| Detection | MSHS   | Prostate Cancer   | 1,000  | Philips Ultrafast |
| Detection | MSHS   | Colorectal Cancer |   413  | Philips Ultrafast |
| Detection | MSHS   | IBD               | 1,448  | Philips Ultrafast |

#### Biomarker Tasks

| Task      | Origin | Biomarker     | Specimen      | Slides | Scanner           |
| --------- | ------ | ------------- | ------------- | -----: | ----------------- |
| Biomarker | MSHS   | IHC ER        | Breast Cancer | 2,000  | Philips Ultrafast |
| Biomarker | MSHS   | IHC PR        | Breast Cancer | 1,986  | Philips Ultrafast |
| Biomarker | MSHS   | IHC/FISH HER2 | Breast Cancer | 2,018  | Philips Ultrafast |
| Biomarker | MSHS   | BioMe HRD     | Breast        |   563  | Philips Ultrafast |
| Biomarker | SUH    | NGS BRAF      | Melanoma      |   283  | Nanozoomer S210   |
| Biomarker | SUH    | NGS NRAS      | Melanoma      |   283  | Nanozoomer S210   |
| Biomarker | MSHS   | NGS EGFR      | LUAD          |   294  | Philips Ultrafast |
| Biomarker | MSKCC  | NGS EGFR      | LUAD          | 1,000  | Aperio AT2        |
| Biomarker | MSKCC  | NGS ALK       | LUAD          |   999  | Aperio AT2        |
| Biomarker | MSKCC  | NGS STK11     | LUAD          |   998  | Aperio AT2        |
| Biomarker | MSKCC  | NGS KRAS      | LUAD          |   998  | Aperio AT2        |
| Biomarker | MSKCC  | NGS TP53      | LUAD          |   998  | Aperio AT2        |
| Outcome   | MSKCC  | ICI Response  | NSCLC         |   454  | Aperio AT2        |

MSHS: Mount Sinai Health System;
DCIS: Ductal Carcinoma In Situ;
IBD: Inflammatory Bowel Disease;
ER: Estrogen Receptor;
PR: Progesterone Receptor;
IHC: Immunohistochemistry;
FISH: Fluorescence In Situ Hybridization;
SUH: Sahlgrenska University Hospital;
MSKCC: Memorial Sloan Kettering Cancer Center;
LUAD: Lung Adenocarcinoma;
ICI: Immene Checkpoint Inhibitors;
NSCLC: Non-Small Cell Lung Cancer

### Public Pathology Foundation Models

| Model                                                               | Param. (M) | Algorithm | Training Data | Tiles (M) | Slides (K) |
| ------------------------------------------------------------------- | ---------: | --------- | ------------- | --------: | ---------: |
| [CTransPath](https://github.com/Xiyue-Wang/TransPath)               |         28 | SRCL      | TCGA, PAIP    |        16 |         32 |
| [Phikon]()                                                          |         86 | iBOT      | TCGA          |        43 |          6 |
| [UNI](https://huggingface.co/MahmoodLab/UNI)                        |        303 | DINOv2    | MGB           |       100 |        100 |
| [Virchow](https://huggingface.co/paige-ai/Virchow)                  |        631 | DINOv2    | MSKCC         |     2,000 |      1,488 |
| [SP22M](https://huggingface.co/MountSinaiCompPath/SP22M)            |         22 | DINO      | MSHS          |     1,600 |        423 |
| [SP85M](https://huggingface.co/MountSinaiCompPath/SP85M)            |         86 | DINO      | MSHS          |     1,600 |        423 |
| [Prov-GigaPath](https://huggingface.co/prov-gigapath/prov-gigapath) |      1,135 | DINOv2    | PHS           |     1,300 |        171 |
| [Virchow2](https://huggingface.co/paige-ai/Virchow2)                |        631 | DINOv2    | MSKCC         |     1,700 |      3,100 |
| [H-optimus-0](https://huggingface.co/bioptimus/H-optimus-0)         |      1,135 | DINOv2    | Proprietary   |      >100 |       >500 |
| [Phikon-v2](https://huggingface.co/owkin/phikon-v2)                 |        307 | DINOv2    | Multicenter   |       456 |         58 |

MGB: Mass General Brigham;
MSKCC: Memorial Sloan Kettering Cancer Center;
MSHS: Mount Sinai Health System;
PHS: Providence Health and Services


## Automated External Benchmarking
We provide a workflow to benchmark user submitted models. To submit a request follow the instructions below:
1. Submit [this form](https://forms.office.com/Pages/ResponsePage.aspx?id=YZ3odw9XsEO55GNPRi40uCeVZRc28JFPi1Agm1twtOFUMjVTRThNQVpRN1RNMldBMTNCUVZHVFFQSi4u) with the user's name and a valid email address. Optionally, a user can allow to record the results on our leaderboard by checking the relative checkbox and providing a model name.
2. The user will receive an email with a link to a secure OneDrive folder.
3. The user should upload to the provided OneDrive folder the following files:
   - Docker container: a Docker (or singularity) containerized environment including the model's weights. Note: currently there is a 250GB limit per file.
   - `inference.py` script: as script which can run in the container provided. It should accept as input a csv file listing the slides to run inference over. It should output a torch tensor of features per slide. We provide a sample [script](https://github.com/fuchs-lab-public/OPAL/blob/main/SSL_benchmarks/automated_external_benchmarking/inference.py) which can be modified accordingly. Additional instructions can be found [here](https://github.com/fuchs-lab-public/OPAL/tree/main/SSL_benchmarks/automated_external_benchmarking).
4. Within a 2 week timeframe, the user will receive via the provided email the results of the benchmarks as a csv file with the following columns:
   - Task
   - Task Type: Detection, Biomarker
   - Mean AUC
   - AUC Standard Deviation
5. After analysis, all data will be purged. If the user opted to save the results, they will be posted in the leaderboard.