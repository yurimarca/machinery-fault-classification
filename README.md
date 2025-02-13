# Machine Fault Classification with MAFAULDA

This repository hosts a project aimed at detecting and classifying mechanical faults using the [MAFAULDA dataset](https://www02.smt.ufrj.br/~offshore/mfs/page_01.html). There is also a [Kaggle dataset](https://www.kaggle.com/datasets/uysalserkan/fault-induction-motor-dataset/data) available, but the previous weblink have the complete dataset with many categories of malfunctions.

## Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Requirements and Setup](#requirements-and-setup)
- [Data Acquisition](#data-acquisition)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [License](#license)
- [Contact](#contact)

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
├── .gitignore
├── requirements.txt
├── README.md
└── ...
```

- **notebooks/**: Jupyter notebooks for data exploration, model building, etc.
- **data/**: Folder for raw and processed data (excluded from the repository).
- **requirements.txt**: Python packages needed to replicate the environment.

## Requirements and Setup
1. **Python Version**: 3.8 or higher recommended
2. **Operating System**: Linux or Mac recommended
2. **Install Dependencies**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```


## Exploratory Data Analysis (EDA)
- A first pass at data inspection is done via [ydata-profiling](https://github.com/ydataai/ydata-profiling).  
- Notebook: `notebooks/EDA.ipynb` generates summary statistics and simple visualizations.

