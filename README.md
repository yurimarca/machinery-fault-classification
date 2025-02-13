# Machine Fault Classification with MAFAULDA

This repository hosts a project aimed at detecting and classifying mechanical faults using the [MAFAULDA dataset](https://www.kaggle.com/datasets/uysalserkan/fault-induction-motor-dataset/data).

## Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Requirements and Setup](#requirements-and-setup)
- [Data Acquisition](#data-acquisition)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)

## Overview
The objective is to build a time-series classification model to distinguish between different motor conditions, such as normal operation and imbalance. The project includes:
- Preprocessing of the raw sensor data
- Neural network model selection, training, and validation
- Performance evaluation using metrics like accuracy, F1-score, and AUC

## Repository Structure
```
├── notebooks/
│   └── EDA.ipynb     	# Jupyter Notebook for EDA
├── data/
│   └── ...              # Raw/processed data (ignored in version control)
├── environment.yml
├── README.md
└── ...
```

- **notebooks/**: Jupyter notebooks for data exploration, model building, etc.
- **data/**: Folder for raw and processed data (excluded from the repository).
- **environment.yml**: Python packages needed to replicate the conda environment.

## Requirements and Setup
1. **Python Version**: 3.8 or higher recommended
2. **Operating System**: Linux or Mac recommended
3. **Create conda environment and install dependencies**:
   ```bash
	conda env create --file environment.yml
	conda activate ml
   ```


## Exploratory Data Analysis (EDA)
- A first pass at data inspection is done via [ydata-profiling](https://github.com/ydataai/ydata-profiling).  
- Notebook: `notebooks/EDA.ipynb` generates summary statistics and simple visualizations.

