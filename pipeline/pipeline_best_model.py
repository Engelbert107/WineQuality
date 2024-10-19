import os
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
# from function_utils import get_balance_class

# from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE  # type: ignore
from imblearn.under_sampling import RandomUnderSampler  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore


##################################################
########## Data Loading and Preparation ##########
##################################################


# Change the current working directory to the script's directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Define the relative paths
relative_path_red = "../data/winequality-red.csv"
relative_path_white = "../data/winequality-white.csv"

# Check if the files exist
if os.path.exists(relative_path_red) and os.path.exists(relative_path_white):
    # Load the datasets
    df_red = pd.read_csv(relative_path_red, sep=";")
    df_white = pd.read_csv(relative_path_white, sep=";")

    # Rename columns
    df_red.rename(
        columns={
            "fixed acidity": "fixed_acidity",
            "volatile acidity": "volatile_acidity",
            "citric acid": "citric_acid",
            "residual sugar": "residual_sugar",
            "free sulfur dioxide": "free_sulfur_dioxide",
            "total sulfur dioxide": "total_sulfur_dioxide",
        },
        inplace=True,
    )

    df_white.rename(
        columns={
            "fixed acidity": "fixed_acidity",
            "volatile acidity": "volatile_acidity",
            "citric acid": "citric_acid",
            "residual sugar": "residual_sugar",
            "free sulfur dioxide": "free_sulfur_dioxide",
            "total sulfur dioxide": "total_sulfur_dioxide",
        },
        inplace=True,
    )

    # Add a column to each dataset indicating the wine type
    df_red["wine_type"] = "red"
    df_white["wine_type"] = "white"

    # Combine datasets
    df_wine = pd.concat([df_red, df_white], ignore_index=True)

    # Convert target variable to binary classes
    df_wine["quality"] = np.where(df_wine["quality"] > 5, 1, 0)

else:
    print("One or both files not found.")


###################################################
########## Features and Target Definition ##########
###################################################

########## Custom Function ##########
def get_balance_class(
    df,
    target_column,
    test_size=0.2,
    random_state=5,
    handle_imbalance=True,
    method="over",
):
    """
    Splits the dataset into training and testing sets, and either oversamples the minority class
    or undersamples the majority class in the training set.

    Parameters:
    - df: pandas DataFrame containing the data.
    - target_column: str, the name of the target variable.
    - test_size: float, the proportion of the dataset to include in the test split.
    - random_state: int, controls the shuffling applied to the data before applying the split.
    - imbalance: bool, whether to handle class imbalance.
    - method: str, "over" to oversample the minority class, "under" to undersample the majority class.

    Returns:
    - X_train_resampled, X_test, y_train_resampled, y_test: Split and balanced training and testing data.
    """
    # Split the data into features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if handle_imbalance:
        if method == "over":
            # Apply SMOTE to oversample the minority class in the training set
            smote = SMOTE(random_state=random_state)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        elif method == "under":
            # Apply RandomUnderSampler to undersample the majority class in the training set
            rus = RandomUnderSampler(random_state=random_state)
            X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)
        else:
            raise ValueError("Invalid method specified. Use 'over' or 'under'.")

    else:
        # If not handling imbalance, keep the original training data
        X_train_resampled, y_train_resampled = X_train, y_train

    return X_train_resampled, X_test, y_train_resampled, y_test

########## End ##########


# Apply Undersample
x_train, x_test, y_train, y_test = get_balance_class(
    df_wine,
    "quality",
    test_size=0.2,
    random_state=5,
    handle_imbalance=True,
    method="under",
)


########################################
########## Feature Extraction ##########
########################################

# Get numeric and categorical columns from the DataFrame
numeric_features = df_wine.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = df_wine.select_dtypes(
    include=["object", "category"]
).columns.tolist()

# Remove the target variable ('quality') from the feature lists
numeric_features = [col for col in numeric_features if col != "quality"]
categorical_features = [col for col in categorical_features if col != "quality"]


###################################
########## Preprocessing ##########
###################################

# Preprocessing for numeric and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ("num", RobustScaler(), numeric_features),
        (
            "cat",
            OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
            categorical_features,
        ),
    ]
)


######################################
########## Model Definition ##########
######################################

# Best hyperparameters from GridSearchCV
best_params = {
    "n_estimators": 100,
    "criterion": "entropy",
    "max_depth": 20,
    "min_samples_split": 2,
    "max_features": "sqrt",
}


# Define the final pipeline
final_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(**best_params, random_state=5)),
    ]
)


###############################################
########## Model Training and Saving ##########
###############################################

# Fit the pipeline on the training data
final_pipeline.fit(x_train, y_train)

# Save the final pipeline for deployment
joblib.dump(final_pipeline, "wine_quality_best_model_pipeline.pkl")
print()
print("The pipeline has been saved successfully!\n")
