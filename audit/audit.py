import pandas as pd
import numpy as np

# --- Configuration ---
FILEPATH = '../heart_disease_uci.csv'

# Column names based on the description provided in the prompt
# Note: Some UCI datasets use '?' for missing values, so we'll explicitly handle that.
COLUMNS_OF_INTEREST = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalch',
    'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num' # 'num' is the target class
]


def load_and_analyze_data(filepath):
    """
    Loads the dataset, calculates sample/feature counts, checks for missing
    values, and analyzes class balance and demographics.
    """
    print(f"--- Loading and Analyzing Dataset: {filepath} ---")
    try:
        # Load the CSV, replacing common non-standard missing indicators ('?')
        df = pd.read_csv(filepath, na_values='?')
        print("Successfully loaded the dataset.")
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}. Please ensure the file is in the correct directory.")
        return

    # 1. Feature Count and Sample Count
    print("\n--- 1. Sample and Feature Count (Shape) ---")
    n_samples = df.shape[0]
    n_features = df.shape[1]
    n_features_used = len(COLUMNS_OF_INTEREST)
    print(f"Total Samples (Rows): {n_samples}")
    print(f"Total Features (Columns) in file: {n_features}")
    print(f"Features of Interest used for ML: {n_features_used} ({COLUMNS_OF_INTEREST})")

    # Assess Sample vs. Feature Ratio
    if n_samples > n_features_used * 10:
        print(f"Sample-to-Feature Ratio: {n_samples / n_features_used:.1f}x. Seems adequate.")
    else:
        print("Warning: Sample size might be low relative to the number of features. Requires further investigation.")


    # 2. Missing Values Check
    print("\n--- 2. Missing Values Analysis ---")
    # Focus only on the columns relevant to the ML task
    missing_values = df[COLUMNS_OF_INTEREST].isnull().sum()
    total_missing = missing_values.sum()

    if total_missing > 0:
        print(f"Total missing values found across features: {total_missing}")
        print("Missing counts per feature:")
        print(missing_values[missing_values > 0].sort_values(ascending=False))
        print("\nConclusion: Missing values **do** need imputation for the features listed above.")
    else:
        print("No missing values found in the primary features.")
        print("Conclusion: No imputation is required based on '?' or standard NaN markers.")

    # 3. Class Balance Check (Target: 'num')
    print("\n--- 3. Class Balance Check ('num' column) ---")
    target_counts = df['num'].value_counts(dropna=False)
    print(target_counts.to_string())

    # Check for binary classification balance (assuming 0=No Disease, >0=Disease)
    # The 'num' column in the UCI dataset often has values 0, 1, 2, 3, 4 where >0 means disease.
    # We will check if it is binary (0 or 1) or multi-class and assess balance.
    if len(target_counts) > 2:
        print("\nTarget is Multi-Class (0, 1, 2, 3, 4). Data is likely UNBALANCED.")
    else:
        # Simple binary check for 0 vs 1
        ratio = target_counts.max() / target_counts.min() if target_counts.min() > 0 else np.inf
        if ratio > 2.0:
            print("\nTarget is Binary. The data is UNBALANCED (Ratio > 2:1).")
        else:
            print("\nTarget is Binary. The data appears BALANCED (Ratio <= 2:1).")

    # 4. Demographic Coverage (Age and Sex)
    print("\n--- 4. Demographic Coverage Check ('age' and 'sex') ---")
    print("\nAge Distribution (Descriptive Stats):")
    print(df['age'].describe().to_string())

    print("\nSex Distribution (Male/Female Counts):")
    # Assuming standard encoding (e.g., 1 for male, 0 for female, or strings)
    sex_mapping = {1.0: 'Male', 0.0: 'Female'} # Common UCI encoding
    sex_counts = df['sex'].value_counts(dropna=False)
    print(sex_counts.rename(index=sex_mapping).to_string())

    # Note on Demographic Fairness:
    print("\nConclusion on Demographics:")
    print("The numbers show the distribution, but assessing 'fairness' requires external knowledge of the target population and ethical review.")


# Execute the analysis
if __name__ == "__main__":
    load_and_analyze_data(FILEPATH)
