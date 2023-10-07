# ChurnDetect

## Project Description
This repository contains the pipeline for a churn prediction project, it relies on the analysis of customer feature through exploratory data analysis and uses variaous machine learning models to predict customer attrition.
We rely on a dataset of bank customers to extract features (feature engineering) for our modeling. The project aims to identify key drivers behind credit card customer attrition and provide actionable insights for businesses to enhance their customer retention strategies.

## Files and data description

Here's a quick overview of the main components in the root directory:

- `churn_notebook.ipynb`: Jupyter notebook containing exploratory data analysis and initial model prototyping.
- `run_train.py`: Python script responsible for training the machine learning models and saving the trained models.
- `churn_library.py`: A library of functions and utilities used throughout the project.
- `data/`: Folder containing raw data files.
  - `bank_data.csv`: Main dataset with customer attributes and their churn status.
- `images/`: Folder with visualizations and plots generated during the EDA.
- `requirements.txt`: A list of Python dependencies required for this project.

## Running Files

To run the files, follow the steps below:

1. **Setting up the Environment**:
    - Ensure Python 3.8 or newer is installed.
    - Create a virtual environment: `python -m venv .venv`
    - Activate the virtual environment:
      - On Windows: `.venv\Scripts\activate`
      - On macOS/Linux: `source .venv/bin/activate`
    - Install the required packages: `pip install -r requirements.txt`

2. **Running the Training Script**:
    - Execute the `run_train.py` script: `python run_train.py`
    - This will train the models and save the trained versions. Various logs and performance metrics will be displayed in the console.

3. **Exploratory Data Analysis**:
    - Launch Jupyter Notebook: `jupyter notebook`
    - Open `churn_notebook.ipynb` and run the cells for a step-by-step analysis of the data.

After executing the scripts, you should see updated model files, performance metrics, and possibly new visualizations in the `images/` folder.