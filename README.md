# Wine Quality Classification using Machine Learning Models
In this project, we aimed to classify wine quality using data from the [UCI Wine Quality dataset](https://archive.ics.uci.edu/dataset/186/wine+quality), focusing on predicting whether a wine's quality rating falls into a lower (3-5) or higher (6-9) category. The dataset included various physicochemical properties, which were used as features to train machine learning models. Our process involved data preprocessing, exploratory analysis, and model training with a focus on normalization techniques like RobustScaler to handle potential outliers.

We trained and evaluated six models: Logistic Regression, ``K-Nearest Neighbors, Random Forest, Gradient Boosting, LightGBM,`` and ``XGBoost``. Among them, ``Random Forest`` emerged as the best performer, achieving an ``accuracy of 84.38%`` and a ``ROC-AUC score of 0.912``, demonstrating strong performance in distinguishing between high and low wine quality. While some false positives occurred, the model maintained a good balance between precision and recall across the classes.

Finally, ``we deployed the Random Forest model using Flask``, enabling local predictions. This deployment provides a solid foundation for future implementation in a production environment. The project showcases how robust preprocessing and model selection can lead to effective predictions, offering potential business value in classifying wine quality.

## How to run scripts?
1. Download the project repository to explore the code and documentation.
2. Run the pipeline for the best model.
   ```bash
   python pipeline/pipeline_best_model.py
   ```
3. To evalute the best model on the test set.
   ```bash
   python evaluate.py
   ```
   ![](https://github.com/Engelbert107/WineQuality/blob/main/images/best_test_result.PNG)
   ![](https://github.com/Engelbert107/WineQuality/blob/main/images/cm_rf.png)

### Model comparison 
We compared all six models using the ROC Curve, AUC scores, and other performance metrics. Although K-Nearest Neighbors and Random Forest achieved similar results in terms of overall performance, ``Random Forest`` outperformed K-Nearest Neighbors in precision, recall, and confusion matrix analysis. Based on these evaluations, we concluded that Random Forest is the best model for our task.
![](https://github.com/Engelbert107/WineQuality/blob/main/images/roc_with_outliers.png)

## How to test the deployed model on unseen data?

1. Make sure you have already run the pipeline for the best model, then run the following.
   ```bash
   python pipeline/pipeline_best_model.py
   ```
2. Run 
   ```bash
   python pipeline/app.py
   ```
   or navigate through the folder.
   ![](https://github.com/Engelbert107/WineQuality/blob/main/images/run_app.PNG)
3. Run
   ```bash
   python pipeline/test_app.py
   ```
   or navigate through the folder.
   ![](https://github.com/Engelbert107/WineQuality/blob/main/images/run_test.PNG)

## Access this repository through the following link
- Access to the [data here](https://github.com/Engelbert107/WineQuality/tree/main/data)
- Access to the [notebook here](https://github.com/Engelbert107/WineQuality/tree/main/notebook)
- Access to different [images here](https://github.com/Engelbert107/WineQuality/tree/main/images)
- Access to the [pipeline and app here](https://github.com/Engelbert107/WineQuality/tree/main/pipeline) 
