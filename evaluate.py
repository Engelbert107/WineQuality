import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

from metrics import plot_confusion_matrix
from pipeline.pipeline_best_model import x_test, y_test

# Load the saved pipeline
pipeline_best_model = joblib.load('pipeline/wine_quality_best_model_pipeline.pkl')

if __name__ == '__main__':
    
    # Apply the pipeline for prediction on test set
    y_pred = pipeline_best_model.predict(x_test)

    print("----"*14)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.3f}%")
    print("----"*14)
    print(f"Classification Report:\n{classification_report(y_test, y_pred, zero_division=0)}")

    # Plot confusion matrix
    plot_confusion_matrix(confusion_matrix(y_test, y_pred))
