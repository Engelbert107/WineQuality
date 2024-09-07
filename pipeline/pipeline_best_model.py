import joblib 
import pandas as pd 
import numpy as np
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import OneHotEncoder, RobustScaler 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.compose import ColumnTransformer 
from sklearn.model_selection import train_test_split 


##################################################
########## Data Loading and Preparation ##########
##################################################

# Load red and white wine datasets
df_red = pd.read_csv('C:/Users/engel/OneDrive/Documents/DataForML/wine-quality/data\winequality-red.csv', sep=';')
df_white = pd.read_csv('C:/Users/engel/OneDrive/Documents/DataForML/wine-quality/data/winequality-white.csv', sep=';')

# Rename the existing DataFrame (rather than creating a copy) 
df_red.rename(columns={'fixed acidity': 'fixed_acidity', 
                   'volatile acidity': 'volatile_acidity',
                   'citric acid': 'citric_acid',
                   'residual sugar': 'residual_sugar',
                   'free sulfur dioxide': 'free_sulfur_dioxide',
                   'total sulfur dioxide': 'total_sulfur_dioxide'}, inplace=True)

df_white.rename(columns={'fixed acidity': 'fixed_acidity', 
                   'volatile acidity': 'volatile_acidity',
                   'citric acid': 'citric_acid',
                   'residual sugar': 'residual_sugar',
                   'free sulfur dioxide': 'free_sulfur_dioxide',
                   'total sulfur dioxide': 'total_sulfur_dioxide'}, inplace=True)

# Add a column to each dataset indicating the wine type
df_red['wine_type'] = 'red'
df_white['wine_type'] = 'white'

# Combine datasets
df_wine = pd.concat([df_red, df_white], ignore_index=True)

# Convert target variable to binary classes
df_wine["quality"] = np.where(df_wine["quality"] > 5, 1, 0)


###################################################
########## Features and Target Definition ##########
###################################################

# Independent variables (features)
X = df_wine.drop(columns=['quality'])  # Drop the target column
y = df_wine['quality']  # Target variable

# Split into training and test set
x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2,
                                                    shuffle = True,
                                                    random_state = 5)


########################################
########## Feature Extraction ##########
########################################

# Get numeric and categorical columns from the DataFrame
numeric_features = df_wine.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = df_wine.select_dtypes(include=['object', 'category']).columns.tolist()

# Remove the target variable ('quality') from the feature lists
numeric_features = [col for col in numeric_features if col != 'quality']
categorical_features = [col for col in categorical_features if col != 'quality']


###################################
########## Preprocessing ##########
###################################

# Preprocessing for numeric and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), numeric_features),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
    ])


######################################
########## Model Definition ##########
######################################

# Best hyperparameters from GridSearchCV
best_params = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2
}

# Define the final pipeline
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(**best_params, random_state=5))
])


###############################################
########## Model Training and Saving ##########
###############################################

# Fit the pipeline on the training data
final_pipeline.fit(x_train, y_train)

# Save the final pipeline for deployment
joblib.dump(final_pipeline, 'wine_quality_best_model_pipeline.pkl')
print()
print('The pipeline has been saved successfully!\n')



