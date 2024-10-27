# Wine Quality Classification using Machine Learning Models
In this project, we aimed to classify wine quality using data from the [UCI ML Repository](https://archive.ics.uci.edu/dataset/186/wine+quality), focusing on predicting whether a wine's quality rating falls into a lower category (3-5) or a higher category (6-9). The dataset included various physicochemical properties, which served as features for training machine learning models. Our process involved data preprocessing, exploratory analysis, and model training, with an emphasis on normalization techniques like RobustScaler to address potential outliers.

We trained and evaluated six models: ``Logistic Regression, K-Nearest Neighbors, Random Forest, Gradient Boosting, LightGBM,`` and ``XGBoost``. Among them, ``Random Forest`` emerged as the best performer, achieving an ``accuracy of 81%`` and a ``ROC-AUC score of 0.897``, demonstrating strong performance in distinguishing between high and low wine quality. While some false alarms occurred, the model maintained a good balance between precision and recall across the classes. [Lime](https://arxiv.org/abs/1705.07874) analysis reveals that for the first instance in the dataset, ``alcohol, density, free sulfur dioxide`` and ``citric acid`` positively contribute to high wine quality, while ``volatile acidity, sulfates, pH, chlorides, residual sugar``, and ``fixed acidicy`` are associated with lower wine quality.

Finally, ``we deployed the Random Forest model using Flask``, enabling local predictions. This deployment provides a solid foundation for future implementation in a production environment. The project showcases how robust preprocessing and model selection can lead to effective predictions, offering potential business value in classifying wine quality.

## How to run scripts?
1. Download the project repository to explore the code and documentation.
2. Install packages.
   ```bash
   pip install -r requirements.txt
   ```
4. Run the pipeline for the best model.
   ```bash
   python our_pipeline/pipeline_best_model.py
   ```
5. To evalute the best model on the test set.
   ```bash
   python evaluation.py
   ```
   ![](https://github.com/Engelbert107/WineQuality/blob/main/images/best_test_result.png)
   ![](https://github.com/Engelbert107/WineQuality/blob/main/images/cm_rf.png)

### Model comparison 
We evaluated six models using ROC curves, AUC scores, and various performance metrics. The ``Random Forest`` model consistently demonstrated the highest precision, recall, and ROC-AUC score, along with strong results in confusion matrix analysis. Based on these evaluations, we identified ``Random Forest`` as the best-performing model for our task. However, additional improvements are necessary to further reduce classification errors.

![](https://github.com/Engelbert107/WineQuality/blob/main/images/auc_roc_robust.png)

## How to test the deployed model on unseen data?

1. Make sure you have already run the pipeline for the best model, then run the following.
   ```bash
   python our_pipeline/pipeline_best_model.py
   ```
2. Run 
   ```bash
   python our_pipeline/app.py
   ```
   or navigate through the folder.
   ![](https://github.com/Engelbert107/WineQuality/blob/main/images/run_app.png)
3. Run
   ```bash
   python our_pipeline/test_app.py
   ```
   or navigate through the folder.
   ![](https://github.com/Engelbert107/WineQuality/blob/main/images/run_test.png)

## Access this repository through the following links:
- Access to the [data here](https://github.com/Engelbert107/WineQuality/tree/main/data)
- Access to the [notebook here](https://github.com/Engelbert107/WineQuality/tree/main/notebook)
- Access to different [images here](https://github.com/Engelbert107/WineQuality/tree/main/images)
- Access to the [pipeline and app here](https://github.com/Engelbert107/WineQuality/tree/main/pipeline) 
