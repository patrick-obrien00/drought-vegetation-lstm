# Modelling Dynamics of Enhanced Vegetation Index in Southern Africa

**Master's Thesis - Applied Data Science**  
Utrecht University, Department of Geosciences  
Author: Patrick O'Brien | July 2025

## 📖 Overview

This repository contains the code and analysis for my master's thesis, which develops a Long Short-Term Memory (LSTM) deep learning model to predict vegetation health (Enhanced Vegetation Index - EVI) across Southern Africa under varying climate conditions, with a focus on drought impacts and recovery dynamics.

## 🎯 Research Objectives

1. **Develop an LSTM model** capable of predicting EVI in Southern Africa using static environmental features and dynamic time-series climate data
2. **Perform sensitivity analysis** to identify the most influential predictors and assess the model's ability to capture vegetation stress during extreme conditions
3. **Generate counterfactual scenarios** by perturbing climate inputs during drought periods to explore vegetation responses to alternative weather conditions

## 🔬 Key Findings

- **Model Performance**: Achieved R² = 0.87 and RMSE = 0.36 on spatially-split test data
- **Feature Importance**: Precipitation emerged as the most influential predictor, especially during drought and recovery periods
- **Drought Detection**: Brier skill scores demonstrated strong performance in identifying periods of poor vegetation growth
- **Climate Sensitivity**: Counterfactual simulations revealed spatially heterogeneous vegetation responses to altered precipitation and temperature scenarios

## 📊 Data & Study Area

**Study Region**: Southern Africa (16°E to 34°E, 34°S to 22°S)  
**Spatial Resolution**: 0.1° grid (~11 km)  
**Temporal Coverage**: 2000-2022 (275 monthly timesteps)

### Datasets Used

| Variable | Source | Description |
|----------|--------|-------------|
| **EVI** | MODIS MOD13Q1 | Enhanced Vegetation Index - measure of vegetation greenness |
| **Precipitation** | ERA5 | Monthly precipitation (mm) |
| **Temperature** | ERA5 | Monthly mean temperature (K) |
| **SPEI** | Custom calculation | Standardized Precipitation Evapotranspiration Index (1 & 3-month) |
| **Elevation** | GMTED2010 | Terrain elevation (m) |
| **Water Table Depth** | Fan et al. 2013 | Depth to groundwater (m) |
| **Sand Fraction** | SoilGrids | Soil sand content (fraction) |
| **Land Cover** | MODIS MCD12Q1 | Land cover classification |

## 🧠 Methodology

### Model Architecture
- **Type**: Multi-layer LSTM (Recurrent Neural Network)
- **Layers**: 3 LSTM layers with dropout (0.40) for regularization
- **Hidden State Size**: 256
- **Training**: Spatially-split data (locations randomly assigned to train/validation/test sets)
- **Optimization**: Hyperparameters tuned using Optuna framework

### Analysis Components
1. **Feature Ablation**: Systematically removed each input to measure its predictive importance
2. **Brier Skill Scores**: Evaluated ability to predict "poor vegetation growth" events (EVI < -1)
3. **Counterfactual Simulations**: Four scenarios testing vegetation responses to altered climate:
   - Moderate Drought (−1σ precipitation, +1σ temperature)
   - Severe Drought (−1.5σ precipitation, +1.5σ temperature)  
   - Wet-Cool (+2σ precipitation, −2σ temperature)
   - Warm-Wet (+2σ precipitation, +2σ temperature)

## 📁 Repository Structure
```
/
├── data/                    # Data processing scripts and utilities
├── model/                  # LSTM model architecture and training code
├── main/                # Feature ablation, sensitivity analysis, counterfactuals
├── results_plots/                 # Output figures, maps, and performance metrics
├── scenarios/
├── Obrien_Patrick_thesis_4813898.pdf    # Full thesis document
└── README.md
```



## 🙏 Acknowledgments

- First examiner: Dr. ir. Niko Wanders
- Second examiner: Steye Verhoeve MSc
- Collaborator: Daan Van der Hoek
