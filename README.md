# DSW

# Full list of covariates
We show the full list of static demographics and time-varying covariates of sepsis patients obtained from [MIMIC-III](https://mimic.physionet.org/).
| Category     | Items                                                   | Type   |
|--------------|---------------------------------------------------------|--------|
| Demographics | age                                                     | Cont.  |
|              | gender                                                  | Binary |
|              | race (white, black, hispanic, other)                    | Binary |
|              | metastatic cancer                                       | Binary |
|              | diabetes                                                | Binary |
|              | height                                                  | Cont.  |
|              | weight                                                  | Cont.  |
|              | bmi                                                     | Cont.  |
| Vital signs  | heart rate, systolic, mean and diastolic blood pressure | Cont.  |
|              | Respiratory rate, SpO2                                  | Cont.  |
|              | Temperatures                                            | Cont.  |
| Lab tests    | sodium, chloride, magnesium                             | Cont.  |
|              | glucose, BUN, creatinine, urineoutput, GCS              | Cont.  |
|              | white blood cells count, bands, C-Reactive protein      | Cont.  |
|              | hemoglobin, hematocrit, aniongap                        | Cont.  |
|              | platelets count, PTT, PT, INR                           | Cont.  |
|              | bicarbonate, lactate                                    | Cont.  |

# Introduction
This repository contains source code for paper ["Estimating Individual Treatment Effects with Time-Varying Confounders"](). 

In this paper, we study the problem of Estimating individual treatment effects with time-varying confounders (as illustrated by a causal graph in the figure below)

<img src="src/Fig1.png" width=40%>

We propose Deep Sequential Weighting (DSW) for estimating ITE with time-varying confounders. DSW consists of three main components: representation learning module, balancing module and prediction module.

<img src="src/model4.png" width=80%>

To demonstrate the effectiveness of our framework, we conduct comprehensive experiments on synthetic, semi-synthetic and real-world EMR datasets ([MIMIC-III](https://mimic.physionet.org/)). DSW outperforms state-of-the-art baselines in terms of PEHE and ATE.

# Requirement
Ubuntu16.04, python 3.6

Install [pytorch 1.4](https://pytorch.org/)

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

# DSW
#### Running example 
```
python train_synthetic.py --observation_window 30 --epochs 64 --batch-size 128 --lr 1e-3
```

#### Outputs
- ITE estimation metrics: PEHE, ATE
- Factual prediction metric: RMSE

