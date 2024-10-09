# The Unknown Project

## Overview

A project for monthly sales forecasting using PyTorch.

## Project Structure

\\\
the-unknown-project/
 .github/
    dependabot.yml
 Data/
    Monthly_sales/
       1.xlsx
       2.xlsx
       ...
    env/                      # Conda environment directory
    ...                       # Other data-related files
├── logs/
│   └── log_file.txt
├── src/
│   ├── data_processing.py
│   ├── evaluate.py
│   ├── model.py
│   ├── train.py
│    utils.py
 configs/
    config.yaml
 notebooks/
    exploratory_analysis.ipynb
 models/
    sales_forecast_model.pth
    scaler_X.pkl
    scaler_y.pkl
 results/
    actual_vs_predicted.png
 .gitignore
 LICENSE
 README.md
 environment.yml                # Conda environment configuration
\\\

## Installation

### Using Conda

1. **Navigate to the Data Directory:**

    `ash
    cd Q:\the-unknown-project\Data
    `

2. **Create the Conda Environment:**

    `ash
    conda create --prefix ./env python=3.9
    `

3. **Activate the Environment:**

    `ash
    conda activate ./env
    `

4. **Install Dependencies:**

    `ash
    conda install pytorch torchvision torchaudio cpuonly -c pytorch
    conda install pandas numpy scikit-learn matplotlib openpyxl pyyaml ipykernel
    pip install python-dotenv
    `

5. **Export the Environment:**

    `ash
    conda env export --prefix ./env > environment.yml
    `

### Running the Project

1. **Activate the Environment:**

    `ash
    conda activate ./env
    `

2. **Run Training Script:**

    `ash
    python ..\src\train.py
    `

3. **Run Evaluation Script:**

    `ash
    python ..\src\evaluate.py
    `

### Using Jupyter Notebooks

1. **Install Jupyter (If Not Already Installed):**

    `ash
    conda install jupyter
    `

2. **Launch Jupyter Notebook:**

    `ash
    jupyter notebook
    `

3. **Open xploratory_analysis.ipynb and select the "sales-forecasting" kernel.**

## License

[MIT License](LICENSE)
