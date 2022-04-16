# DSW

# Introduction
This repository contains the modified source code that is based on https://github.com/ruoqi-liu/DSW.

As part of the final project in CS 598 - Machine Learning in Health Care at the University of Illinois, we seek to reproduce the results published in the paper ["Estimating Individual Treatment Effects with Time-Varying Confounders"](https://arxiv.org/abs/2008.13620)


# Data preprocessing
### Synthetic dataset
We successfully simulated the all covariates, treatments and outcomes. The model was then trained on a modest laptop using "--observation_window 30 --epochs 64 --batch-size 128 --lr 1e-3" parameters to test the feasibility of the setup.
```
cd simulation
python synthetic.py
```

### Semi-synthetic dataset
The MIMIC III data had to be preprocessed using [Big Query](https://mimic.mit.edu/docs/gettingstarted/cloud/bigquery/) to format the input data to an expected format by the setup. 
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

