# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a time series forecasting project that predicts nuclear energy production in the United States using Multi-Layer Perceptron (MLP) neural networks. The project uses hourly electricity consumption and production data from Kaggle.

## Environment Setup

The project uses Python 3.13 with a virtual environment:

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Dependencies:** pandas, numpy, statsmodels, matplotlib, scikit-learn, tqdm, kagglehub

## Working with the Notebook

The main analysis is in `IA_3_modelos.ipynb`. This is a Jupyter notebook containing:

**Data Pipeline:**
1. Downloads dataset from Kaggle using `kagglehub`
2. Loads CSV: `/kaggle/input/hourly-electricity-consumption-and-production/electricityConsumptionAndProductioction.csv`
3. Extracts 'Nuclear' column as univariate time series
4. Splits data chronologically:
   - Train: 2019-2021
   - Validation: 2022
   - Test: 2023+

**Preprocessing:**
- Normalizes data using `MinMaxScaler` fitted on training set only
- Creates sliding windows with 24-hour window size (based on ACF/PACF analysis)
- Transforms to supervised learning format (X: 24 timesteps, y: next value)

**Model Architecture:**
- Uses scikit-learn's `MLPRegressor`
- Grid search over hidden layers, activation functions, solvers, alpha, and learning rates
- Best model: (50, 50) hidden layers, logistic activation, LBFGS solver, alpha=0.01
- Training combines train+validation sets for final model

**Evaluation:**
- Metrics: MAPE, MSE, RMSE
- Learning curves using TimeSeriesSplit (5 splits)
- Visual comparisons of predictions vs actuals

## Key Functions

**`create_sliding_windows(series, window_size)`**: Transforms time series into supervised learning format by creating overlapping windows of size `window_size+1` (features + target).

**`acf_pacf(x, qtd_lag)`**: Plots autocorrelation and partial autocorrelation functions to determine optimal window size.

## Data Access Note

The notebook references a Kaggle dataset path at `/kaggle/input/...` which may need adjustment if running locally. The dataset is downloaded to `~/.cache/kagglehub/datasets/stefancomanita/hourly-electricity-consumption-and-production/`.
