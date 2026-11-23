# CEEMDAN-LSTM PV Forecasting for Virtual Power Plants

Official code repository for the study *Ultra-Short-Term Power Output Prediction for Distributed Photovoltaic in Virtual Power Plants Using CEEMDAN-LSTM*.

## Repository structure

```
├── main.py                     # entry script orchestrating the full pipeline
├── config/
│   └── settings.py             # global hyperparameters and the seed configuration
├── src/
│   ├── preprocessing.py        # fault removal, interpolation, chronological partitioning
│   ├── decomposition.py        # CEEMDAN and frequency merging inside each partition
│   ├── model.py                # shared-weight multi-channel LSTM with quantile head
│   ├── data_generator.py       # input window and channel construction
│   └── utils.py                # metrics of equations 24-28 (RMSE, MAE, MAPE, PICP, PINAW)
├── results/
│   ├── per_step_rmse_curve.csv       # pooled and cumulative RMSE/MAE over the 48 forecast steps
│   ├── repeated_experiments_rmse.csv # RMSE of the 10 independent repeated experiments
│   ├── errors_proposed.npy           # forecast errors of the proposed model
│   ├── errors_baseline.npy           # forecast errors of the CEEMDAN-Informer baseline
│   ├── errors_informer.npy           # forecast errors of the Decomposition-Informer baseline
│   └── ramp_mask.npy                 # ramp-period indicator per window and forecast step
└── requirements.txt
```

## Installation

```
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Execution

```
python main.py
```

The random seeds of the 10 repeated experiments are configured in `config/settings.py` as `SEEDS = [2021, 2022, ..., 2030]`. All experiments iterate over this seed list; the seed currently active for a run is recorded in the output summary.

## Demonstration code and experiment pipeline

This repository additionally contains a lightweight demonstration implementation that runs on synthetic data on a central processing unit only. The demonstration implementation generates synthetic data on first execution, and the model components are implemented with pure numpy in order to run without GPU or CUDA dependencies. The reported experiments are reproduced by replacing the synthetic files in `data/` with the dataset available in the Dryad data repository at DOI:10.5061/dryad.m37pvmd99 and running the pipeline in the execution order described in the Experimental Setup and Evaluation Indicators section of the manuscript.

The hyperparameter values marked `[paper]` in `config/settings.py` correspond to the configuration used for the reported experiments; the lightweight values are used by the demonstration implementation only.

## Data

The raw rooftop photovoltaic dataset analyzed in this study is publicly available in the Dryad data repository at DOI:10.5061/dryad.m37pvmd99.
