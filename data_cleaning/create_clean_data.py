import pandas as pd

# Configuration
INPUT_FILEPATH = '../data/heart_disease_uci.csv'
OUTPUT_FILEPATH = '../data/heart_disease_uci_cleaned.csv'

# Define feature types for appropriate imputation strategy
CONTINUOUS_FEATURES = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
CATEGORICAL_FEATURES = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
TARGET = 'num'


# Note: 'id' and 'dataset' will be dropped automatically if present
# 'id' is just an identifier and 'dataset' has zero variance (all Cleveland)

def clean_dataset(input_filepath, output_filepath):
    """
    Cleans the heart disease dataset by imputing missing values:
    - Continuous features: median imputation
    - Categorical features: mode imputation
    Saves the cleaned dataset to a new CSV file.
    """
    print(f"Loading dataset from: {input_filepath}")

    try:
        df = pd.read_csv(input_filepath)
        print(f"Dataset loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: File not found at {input_filepath}")
        return None

    # Report initial missing values
    print("\n--- Initial Missing Values Summary ---")
    total_missing_before = df.isnull().sum().sum()
    print(f"Total missing values: {total_missing_before}")
    missing_by_column = df.isnull().sum()
    if total_missing_before > 0:
        print("\nMissing values by column:")
        print(missing_by_column[missing_by_column > 0].sort_values(ascending=False))

    # Create a copy for cleaning
    df_cleaned = df.copy()

    # Drop columns with zero variance (e.g., 'dataset' if all values are the same)
    print("\n--- Checking for Zero-Variance Columns ---")
    columns_to_drop = []

    # Check 'dataset' column specifically
    if 'dataset' in df_cleaned.columns:
        print("'dataset' column displays where the data is from -- not relevant. Dropping it.")
        columns_to_drop.append('dataset')

    # Drop 'id' column as it's just an identifier
    if 'id' in df_cleaned.columns:
        print("'id' column is an identifier. Dropping it.")
        columns_to_drop.append('id')

    if columns_to_drop:
        df_cleaned = df_cleaned.drop(columns=columns_to_drop)
        print(f"Dropped columns: {columns_to_drop}")
        print(f"New shape: {df_cleaned.shape}")

    # Impute continuous features with median
    print("\n--- Imputing Continuous Features (Median) ---")
    for feature in CONTINUOUS_FEATURES:
        if feature in df_cleaned.columns:
            missing_count = df_cleaned[feature].isnull().sum()
            if missing_count > 0:
                median_value = df_cleaned[feature].median()
                df_cleaned[feature] = df_cleaned[feature].fillna(median_value)
                print(f"{feature}: Imputed {missing_count} values with median = {median_value:.2f}")

    # Impute categorical features with mode
    print("\n--- Imputing Categorical Features (Mode) ---")
    for feature in CATEGORICAL_FEATURES:
        if feature in df_cleaned.columns:
            missing_count = df_cleaned[feature].isnull().sum()
            if missing_count > 0:
                mode_value = df_cleaned[feature].mode()
                if len(mode_value) > 0:
                    mode_value = mode_value[0]
                    df_cleaned.loc[df_cleaned[feature].isnull(), feature] = mode_value
                    print(f"{feature}: Imputed {missing_count} values with mode = '{mode_value}'")
                else:
                    print(f"{feature}: Warning - Could not compute mode (all values missing)")

    # Special handling for boolean-like categorical features that might be stored as strings
    # Convert TRUE/FALSE strings to proper format if they exist
    boolean_features = ['fbs', 'exang']
    for feature in boolean_features:
        if feature in df_cleaned.columns:
            # Check if values are strings like 'TRUE'/'FALSE'
            if df_cleaned[feature].dtype == 'object':
                print(f"\nNote: '{feature}' contains string values, preserving as-is for imputation")

    # Handle target variable if it has missing values
    if TARGET in df_cleaned.columns:
        target_missing = df_cleaned[TARGET].isnull().sum()
        if target_missing > 0:
            print(f"\nWarning: Target variable '{TARGET}' has {target_missing} missing values.")
            print("Removing rows with missing target values...")
            df_cleaned = df_cleaned.dropna(subset=[TARGET])
            print(f"Removed {target_missing} rows. New shape: {df_cleaned.shape}")

    # Final missing values check
    print("\n--- Final Missing Values Summary ---")
    total_missing_after = df_cleaned.isnull().sum().sum()
    print(f"Total missing values after cleaning: {total_missing_after}")

    if total_missing_after > 0:
        print("\nRemaining missing values by column:")
        remaining_missing = df_cleaned.isnull().sum()
        print(remaining_missing[remaining_missing > 0])
    else:
        print("All missing values have been handled successfully.")

    # Save cleaned dataset
    print(f"\n--- Saving Cleaned Dataset ---")
    df_cleaned.to_csv(output_filepath, index=False)
    print(f"Cleaned dataset saved to: {output_filepath}")
    print(f"Final shape: {df_cleaned.shape}")

    assert df_cleaned.shape == (920, 14) # specified in doc

    return df_cleaned


if __name__ == "__main__":
    cleaned_data = clean_dataset(INPUT_FILEPATH, OUTPUT_FILEPATH)
    print("\nData cleaning complete!")