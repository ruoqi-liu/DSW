# DSW

# Introduction

This repository contains code for reproducing the paper ["Estimating Individual Treatment Effects with Time-Varying Confounders"]().

In this paper, we study the problem of Estimating individual treatment effects with time-varying confounders (as illustrated by a causal graph in the figure below)

<img src="src/Fig1.png" width=40%>

The paper proposes Deep Sequential Weighting (DSW) for estimating ITE with time-varying confounders. DSW consists of three main components: representation learning module, balancing module and prediction module.

<img src="src/model4.png" width=80%>

To demonstrate the effectiveness of the framework, the original paper conducts comprehensive experiments on synthetic, semi-synthetic and real-world EMR datasets ([MIMIC-III](https://mimic.physionet.org/)). DSW outperforms state-of-the-art baselines in terms of PEHE and ATE.

# Requirement

Ubuntu16.04, python 3.6

Install [pytorch 1.4](https://pytorch.org/)

# Data Generation

We need to generate data files using the MIMIC-iii data. These files will be used to generate the semi-synthetic and real world datasets.

## ICU Stay Ids

Use this [toolkit](https://github.com/alistairewj/sepsis3-mimic/tree/v1.0.0) to generate the `sepsis3-df.csv` by running the `sepsis-3-get-data.ipynb`. This file will then be used to generate both the `icustay_ids.txt` and `icu_hadm_dict.json` that will be used to generate the semi-synthetic and real world datasets used in `./simulation/pre_mimic.py` and `./simulation/simulate_mimic.py`

To generate the `icustay_ids.txt` and `icu_hadm_dict.json` files, run [data_generation.py](./data_generator.py) using `python data_generation.py`

## Time Varying Covariates

## Static Demographic Data

# Data preprocessing

### Synthetic dataset

Simulate the all covariates, treatments and outcomes

```
cd simulation
python synthetic.py
```

### Semi-synthetic dataset

With a similar simulation process, we construct a semi-synthetic dataset based on a real-world dataset: [MIMIC-III](https://mimic.physionet.org/).

```
cd simulation
python synthetic_mimic.py
```

### MIMIC-III dataset

Obtain the patients data of two treatment-outcome pairs: (1) vasopressor-Meanbp; (2) ventilator-SpO2.

```
cd simulation
python pre_mimic.py
```

# Full list of covariates

We show the full list of static demographics and time-varying covariates of sepsis patients obtained from [MIMIC-III](https://mimic.physionet.org/).
| Category | Items | Type |
|--------------|---------------------------------------------------------|--------|
| Demographics | age | Cont. |
| | gender | Binary |
| | race (white, black, hispanic, other) | Binary |
| | metastatic cancer | Binary |
| | diabetes | Binary |
| | height | Cont. |
| | weight | Cont. |
| | bmi | Cont. |
| Vital signs | heart rate, systolic, mean and diastolic blood pressure | Cont. |
| | Respiratory rate, SpO2 | Cont. |
| | Temperatures | Cont. |
| Lab tests | sodium, chloride, magnesium | Cont. |
| | glucose, BUN, creatinine, urineoutput, GCS | Cont. |
| | white blood cells count, bands, C-Reactive protein | Cont. |
| | hemoglobin, hematocrit, aniongap | Cont. |
| | platelets count, PTT, PT, INR | Cont. |
| | bicarbonate, lactate | Cont. |

# DSW

#### Running example

```
python train_synthetic.py --observation_window 30 --epochs 64 --batch-size 128 --lr 1e-3
```

#### Outputs

- ITE estimation metrics: PEHE, ATE
- Factual prediction metric: RMSE
