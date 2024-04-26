This project demonstrates the implementation and evaluation of custom classifiers alongside a Random Forest classifier for a thyroid dataset. The custom classifiers include a Bernoulli Naive Bayes classifier and a K-Nearest Neighbors classifier.

## Dataset

The dataset used in this project is `Thyroid_Diff.csv`, which contains various features related to thyroid conditions. The target variable is whether the condition recurred (`Recurred`).

## Requirements

To run the code in this project, you'll need:

- Python 3.x
- Required libraries: numpy, pandas, scikit-learn, matplotlib, seaborn

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/Sasi-Praneeth-Reddy/predicting-thyroid-cancer-reccurence.git
    ```

2. Navigate to the project directory:

    ```bash
    cd CODE
    ```

3. Install the required libraries using pip:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Ensure you have the dataset `Thyroid_Diff.csv` placed in the `data/` directory.

2. Run the Python script `custom_classifiers.py` located in the `models/` directory:

    ```bash
    python models/custom_classifiers.py
    ```

3. The script will generate various evaluation metrics and plots, including confusion matrices, ROC curves, accuracy comparison, and AUC comparison.

4. The predictions made by each classifier will be saved in the `outputs/predictions.txt` file.

## Directory Structure

- **data/**: Contains the dataset file `Thyroid_Diff.csv`.
- **models/**: Contains the Python script `custom_classifiers.py` for implementing and evaluating custom classifiers.
- **outputs/**: Contains output files such as confusion matrices, ROC curves, and comparison plots.
- **tests/**: Directory for storing any test scripts or test data.

## Results

- The `outputs/` directory contains various evaluation plots and files, including confusion matrices, ROC curves, accuracy comparison, and AUC comparison.