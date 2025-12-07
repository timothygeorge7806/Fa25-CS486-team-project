import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from math import sqrt
from collections import Counter

class RandomForestPipeline:
    """
    Orchestrates the entire Random Forest experiment pipeline for heart disease prediction.
    This class handles data preparation, one-hot encoding, hyperparameter tuning via manual
    k-fold cross-validation, final model training, evaluation, and plotting.
    """

    def __init__(self, orig_file_path, random_state=42, k_folds=3):
        self.orig_file_path = orig_file_path
        self.random_state = random_state
        self.train_file_path = None
        self.verif_file_path = None
        self.k_folds = k_folds
        self.categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
        self.target_col = "num"
        self.X = None

    def load_and_preprocess_data(self):
        """
        Loads the original dataset, preprocesses it (one-hot encoding), confirms class balance,
        and creates a separate verification database with one positive and one negative sample.

        Converts multi-class target (0-4) to binary classification (0 = no disease, 1 = disease).
        """
        df = pd.read_csv(self.orig_file_path)

        print(f"Original shape (All Origins): {df.shape}")

        # 1. Filter for only Cleveland samples
        if 'dataset' in df.columns:
            df = df[df['dataset'] == 'Cleveland']

            # 2. Drop the 'origin' column (it's now useless since every row is 'Cleveland')
            df = df.drop('dataset', axis=1)
            print("Filtered to Cleveland only.")
        else:
            print("WARNING: 'origin' column not found. Could not filter.")

        print(f"New shape (Cleveland Only): {df.shape}")

        # Convert multi-class target to binary: 0 = no disease, 1-4 = disease
        print(f"Original target distribution:")
        print(df[self.target_col].value_counts().sort_index())
        df[self.target_col] = (df[self.target_col] > 0).astype(int)
        print(f"\nBinary target distribution:")
        print(df[self.target_col].value_counts().sort_index())

        # Clean categorical columns
        for col in self.categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        # Detect and convert non-numeric columns to numeric (handles '?' values)
        non_numeric = [c for c in df.columns if df[c].dtype == 'object' and c not in self.categorical_cols]
        for c in non_numeric:
            df[c] = pd.to_numeric(df[c], errors='coerce')

        # Drop rows with NaN values after conversion
        df = df.dropna()

        # One-hot encode categorical features (drop_first=True to avoid multicollinearity)
        df_encoded = pd.get_dummies(df, columns=self.categorical_cols, drop_first=True)

        print(f"Original shape: {df.shape}")
        print(f"Encoded shape: {df_encoded.shape}")
        print(f"Number of features after encoding: {df_encoded.shape[1] - 1}")  # -1 for target

        # Check class balance
        y = df_encoded[self.target_col]
        self._confirm_balance(y)

        # Create verification and training databases
        class_0_samples = df_encoded[df_encoded[self.target_col] == 0]
        class_1_samples = df_encoded[df_encoded[self.target_col] == 1]

        verification_sample_0 = class_0_samples.sample(n=1, random_state=self.random_state)
        verification_sample_1 = class_1_samples.sample(n=1, random_state=self.random_state)

        verification_db = pd.concat([verification_sample_0, verification_sample_1])
        training_db = df_encoded.drop(verification_db.index)

        # Save to files
        os.makedirs("data", exist_ok=True)
        self.verif_file_path = "data/verification_db.csv"
        self.train_file_path = "data/training_db.csv"

        verification_db.to_csv(self.verif_file_path, index=False)
        training_db.to_csv(self.train_file_path, index=False)

        print(f"\nTraining DB shape: {training_db.shape}")
        print(f"Verification DB shape: {verification_db.shape}")

        return training_db, verification_db

    def _confirm_balance(self, y):
        """
        Confirms that the dataset is reasonably balanced (each class > 10% of total).
        """
        threshold = 0.1
        sample_size = len(y)

        print("\nClass Distribution:")
        for k, v in Counter(y).items():
            proportion = int(v) / sample_size
            assert proportion > threshold, f"Class {k} has proportion {proportion:.2%}, below threshold {threshold:.0%}"
            print(f"  Class {k}: {proportion:.2%} ({v} samples)")
        print("Data is balanced")

    @staticmethod
    def create_param_grid(n_features):
        """
        Defines the hyperparameter grid for the grid search.
        max_features options are: 0.5*sqrt(n), sqrt(n), 2*sqrt(n), capped at n_features.
        """
        sqrt_n = sqrt(n_features)
        max_features_options = [
            int(0.5 * sqrt_n),
            int(sqrt_n),
            min(int(2 * sqrt_n), n_features)  # Cap at n_features
        ]

        return {
            'n_estimators': [500, 1000],
            'max_features': max_features_options,
        }

    def k_fold_cross_validation(self, X, y, NTREE, MTRY, verbose=False):
        """
        Performs a manual k-fold cross-validation to evaluate model performance
        for a given set of hyperparameters. A new model is trained for each fold.
        """
        np.random.seed(self.random_state)
        indices = np.random.permutation(len(X))

        # Use iloc for DataFrame indexing
        X_shuffled = X.iloc[indices].reset_index(drop=True)
        y_shuffled = y.iloc[indices].reset_index(drop=True)

        k = self.k_folds
        fold_size = len(X_shuffled) // k

        folds = []
        for i in range(k):
            start = i * fold_size
            end = (i + 1) * fold_size if i != k - 1 else len(X_shuffled)
            folds.append((X_shuffled.iloc[start:end], y_shuffled.iloc[start:end]))

        accuracies = []
        errors = []
        confusion_matrices = []
        recalls = []
        precisions = []
        f1_scores = []
        specificities = []

        for i in range(k):
            X_test, y_test = folds[i]
            X_train = pd.concat([folds[j][0] for j in range(k) if j != i])
            y_train = pd.concat([folds[j][1] for j in range(k) if j != i])

            # Initialize a new model for each fold
            model = RandomForestClassifier(n_estimators=NTREE, max_features=MTRY, random_state=self.random_state)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            err = 1 - acc

            accuracies.append(acc)
            errors.append(err)
            confusion_matrices.append(confusion_matrix(y_test, y_pred))
            recalls.append(recall_score(y_test, y_pred, average='weighted'))
            precisions.append(precision_score(y_test, y_pred, average='weighted'))
            f1_scores.append(f1_score(y_test, y_pred, average='weighted'))

            # Specificity calculation (for binary: recall of negative class)
            cm = confusion_matrix(y_test, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            else:
                spec = recall_score(y_test, y_pred, pos_label=0, average='weighted')
            specificities.append(spec)

            if verbose:
                print(f"  Fold {i + 1}: acc={acc:.5f}, err={err:.5f}")

        # Sum confusion matrices from all folds
        final_confusion_matrix = np.sum(confusion_matrices, axis=0)

        return {
            'accuracy': np.mean(accuracies),
            'error': np.mean(errors),
            'recall': np.mean(recalls),
            'precision': np.mean(precisions),
            'f1_score': np.mean(f1_scores),
            'specificity': np.mean(specificities),
            'confusion_matrix': final_confusion_matrix
        }

    def run_grid_search(self):
        """
        Orchestrates the grid search to find the best hyperparameters.
        Returns the metrics for the best-performing model.
        """

        df = pd.read_csv(self.train_file_path)
        X = df.drop(self.target_col, axis=1)
        self.X = X
        y = df[self.target_col]
        n_features = X.shape[1]

        params = self.create_param_grid(n_features)
        print(f"\nGrid Search Parameters:")
        print(f"  n_estimators: {params['n_estimators']}")
        print(f"  max_features: {params['max_features']}")
        print(f"  Total combinations: {len(params['n_estimators']) * len(params['max_features'])}")

        all_results = []
        print(f"\nRunning {self.k_folds}-fold cross-validation for each combination...")

        for ntree in params['n_estimators']:
            for mtry in params['max_features']:
                print(f"\nTesting n_estimators={ntree}, max_features={mtry}")
                cv_metrics = self.k_fold_cross_validation(X, y, ntree, mtry, verbose=True)

                cv_metrics['n_estimators'] = ntree
                cv_metrics['max_features'] = mtry

                all_results.append(cv_metrics)

        results_df = pd.DataFrame(all_results)
        print("\n" + "=" * 80)
        print("Grid Search Results:")
        print("=" * 80)
        print(results_df[['n_estimators', 'max_features', 'accuracy', 'error', 'precision', 'recall', 'f1_score']])

        best_model_results = results_df.loc[results_df['accuracy'].idxmax()]
        print("\n" + "=" * 80)
        print("Best Model Found:")
        print("=" * 80)
        print(best_model_results)

        return best_model_results, X, y, df

    def train_final_model(self, X, y, best_ntree, best_mtry):
        """
        Trains the final model using the best hyperparameters on the full training set.
        """
        final_model = RandomForestClassifier(
            n_estimators=best_ntree,
            max_features=best_mtry,
            random_state=self.random_state,
            oob_score=True
        )

        final_model.fit(X, y)

        return final_model

    def run_time_predict(self, final_model):
        """
        Acts as the RF run time engine to predict the class of new samples
        from the verification database.
        """
        verification_df = pd.read_csv(self.verif_file_path)
        X_verif = verification_df.drop(self.target_col, axis=1)
        y_verif = verification_df[self.target_col]

        predictions = final_model.predict(X_verif)
        probabilities = final_model.predict_proba(X_verif)

        return predictions, probabilities, y_verif


class Plotter:
    """
    Handles all visualization tasks for the Random Forest pipeline.
    """

    @staticmethod
    def plot_confusion_matrix(confusion_matrix, title="Confusion Matrix (Cross-Validation)"):
        """
        Plots a confusion matrix as a heatmap.
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Class 0', 'Class 1'],
                    yticklabels=['Class 0', 'Class 1'])
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_feature_importances(model, feature_names, top_n=10, categorical_cols=None):
        """
        Plots feature importances with optional aggregation of one-hot encoded features.

        Shows two plots:
        1. Individual feature importances (all one-hot encoded features)
        2. Aggregated importances (one-hot features summed back to original categorical)
        """
        importances = pd.Series(model.feature_importances_, index=feature_names)

        # Plot 1: Individual features (top N)
        plt.figure(figsize=(12, 6))
        top_features = importances.sort_values(ascending=False).head(top_n)
        top_features.plot(kind='bar')
        plt.title(f'Top {top_n} Individual Feature Importances (Gini)')
        plt.ylabel('Importance')
        plt.xlabel('Feature')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        # Plot 2: Aggregated features (if categorical columns provided)
        if categorical_cols:
            aggregated_dict = {}

            for feature, importance in importances.items():
                found_original = False
                for cat_col in categorical_cols:
                    if feature.startswith(cat_col + '_'):
                        aggregated_dict[cat_col] = aggregated_dict.get(cat_col, 0) + importance
                        found_original = True
                        break

                if not found_original:
                    aggregated_dict[feature] = aggregated_dict.get(feature, 0) + importance

            aggregated_importances = pd.Series(aggregated_dict)

            plt.figure(figsize=(12, 6))
            top_aggregated = aggregated_importances.sort_values(ascending=False).head(top_n)
            top_aggregated.plot(kind='bar', color='coral')
            plt.title(f'Top {top_n} Aggregated Feature Importances (Gini)')
            plt.ylabel('Importance')
            plt.xlabel('Feature (Aggregated)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()

            print("\nTop 10 Aggregated Feature Importances:")
            print(aggregated_importances.sort_values(ascending=False).head(10))

    @staticmethod
    def plot_verification_results(predictions, probabilities, y_true):
        """
        Displays verification results in a formatted table.
        """
        results_df = pd.DataFrame({
            'True Label': y_true,
            'Predicted Label': predictions,
            'Probability Class 0': probabilities[:, 0],
            'Probability Class 1': probabilities[:, 1]
        })

        print("\n" + "=" * 80)
        print("Verification Results:")
        print("=" * 80)
        print(results_df.to_string(index=False))

        # Visual representation
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.axis('tight')
        ax.axis('off')

        table_data = []
        for idx, row in results_df.iterrows():
            table_data.append([
                f"{row['True Label']:.0f}",
                f"{row['Predicted Label']:.0f}",
                f"{row['Probability Class 0']:.4f}",
                f"{row['Probability Class 1']:.4f}"
            ])

        table = ax.table(cellText=table_data,
                         colLabels=['True Label', 'Predicted Label', 'Prob Class 0', 'Prob Class 1'],
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        plt.title('Verification Database Predictions', pad=20, fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()


def main():
    """
    Main execution block that orchestrates the entire pipeline.
    """
    DATA_FILE = "../data/heart_disease_uci_cleaned_w_dataset.csv"

    # Initialize pipeline and plotter
    pipeline = RandomForestPipeline(orig_file_path=DATA_FILE, random_state=42, k_folds=3)
    plotter = Plotter()

    print("=" * 80)
    print("HEART DISEASE PREDICTION - RANDOM FOREST PIPELINE")
    print("=" * 80)

    # Step 1: Load and preprocess data
    print("\nStep 1: Loading and preprocessing data...")
    training_db, verification_db = pipeline.load_and_preprocess_data()

    # Step 2: Run grid search to find best parameters
    print("\nStep 2: Running grid search...")
    best_results, X, y, df = pipeline.run_grid_search()

    # Step 3: Train final model with best parameters
    print("\nStep 3: Training final model with best parameters...")
    best_ntree = int(best_results['n_estimators'])
    best_mtry = int(best_results['max_features'])

    print(f"\nBest hyperparameters:")
    print(f"  n_estimators: {best_ntree}")
    print(f"  max_features: {best_mtry}")

    final_model = pipeline.train_final_model(X, y, best_ntree, best_mtry)

    # Step 4: Evaluate final model
    print("\nStep 4: Evaluating final model...")
    oob_score = final_model.oob_score_
    oob_error = 1 - oob_score
    print(f"\nOut-of-Bag (OOB) Score: {oob_score:.5f}")
    print(f"Out-of-Bag (OOB) Error: {oob_error:.5f}")

    # Additional CV using sklearn's built-in
    cv_scores = cross_val_score(final_model, X, y, cv=pipeline.k_folds)
    print(f"Built-in CV Accuracy (mean): {np.mean(cv_scores):.5f}")

    # Print final metrics from grid search
    print("\n" + "=" * 80)
    print("Final Model Performance Metrics (from Grid Search):")
    print("=" * 80)
    print(f"Accuracy:    {best_results['accuracy']:.5f}")
    print(f"Error:       {best_results['error']:.5f}")
    print(f"Precision:   {best_results['precision']:.5f}")
    print(f"Recall:      {best_results['recall']:.5f}")
    print(f"Specificity: {best_results['specificity']:.5f}")
    print(f"F1 Score:    {best_results['f1_score']:.5f}")

    # Step 5: Plot confusion matrix
    print("\nStep 5: Generating visualizations...")
    plotter.plot_confusion_matrix(best_results['confusion_matrix'])

    # Step 6: Plot feature importances
    feature_names = df.drop(pipeline.target_col, axis=1).columns
    plotter.plot_feature_importances(
        model=final_model,
        feature_names=feature_names,
        top_n=10,
        categorical_cols=pipeline.categorical_cols
    )

    # Step 7: Run verification predictions
    print("\nStep 6: Running verification predictions...")
    predictions, probabilities, y_verif = pipeline.run_time_predict(final_model)
    plotter.plot_verification_results(predictions, probabilities, y_verif)

    print("\n" + "=" * 80)
    print("Pipeline completed successfully!")
    print("=" * 80)

    # Return the final model for further use
    return final_model, pipeline.X


if __name__ == '__main__':
    final_trained_model, X = main()
    print(f"\nFinal model is accessible via the 'final_trained_model' variable")
    print(f"Model type: {type(final_trained_model)}")
    print(f"Number of trees: {final_trained_model.n_estimators}")
    import shap

    # Calculate values
    explainer = shap.TreeExplainer(final_trained_model)
    shap_values = explainer.shap_values(X)

    print(f"Original SHAP shape: {shap_values.shape}")

    # If it is a 3D array (Samples, Features, Classes), slice it
    if len(shap_values.shape) == 3:
        print("Detected 3D array. Slicing for Class 1 (Disease)...")
        # [:, :, 1] means: "All samples, All features, ONLY Class 1"
        shap_values_class1 = shap_values[:, :, 1]
        shap.summary_plot(shap_values_class1, X)

    # Fallback: If it's already 2D, just plot it
    else:
        print("Detected 2D array. Plotting directly...")
        shap.summary_plot(shap_values, X)