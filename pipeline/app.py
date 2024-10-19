import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify


app = Flask(__name__)


# Load the model pipeline
# pipeline = joblib.load('wine_quality_best_model_pipeline.pkl')

# Avoid changing the working directory
pipeline_file_path = os.path.join(os.path.dirname(__file__), 'wine_quality_best_model_pipeline.pkl')

if os.path.exists(pipeline_file_path):
    pipeline = joblib.load(pipeline_file_path)
else:
    print(f"File not found: {pipeline_file_path}")


@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Read CSV file into DataFrame
        df = pd.read_csv(file, delimiter=';')
        # print(df.head(3))  # Debugging: Print the first few rows

        # Ensure all expected columns are present
        expected_columns = set(['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'chlorides', 'wine_type'])
        present_columns = set(df.columns)
        
        missing_columns = expected_columns - present_columns
        if missing_columns:
            return jsonify({'error': f'Missing columns: {missing_columns}'})

        # Make predictions
        predictions = pipeline.predict(df)

        # Convert predictions to a list and return as JSON
        return jsonify(predictions.tolist())
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
