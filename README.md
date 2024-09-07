# WineQuality
In this project, we aimed to classify wine quality using data from the UCI Wine Quality dataset, focusing on predicting whether a wine's quality rating falls into a lower (3-5) or higher (6-9) category. The dataset included various physicochemical properties, which were used as features to train machine learning models. Our process involved data preprocessing, exploratory analysis, and model training with a focus on normalization techniques like RobustScaler to handle potential outliers.

We trained and evaluated six models: Logistic Regression, K-Nearest Neighbors, Random Forest, Gradient Boosting, LightGBM, and XGBoost. Among them, Random Forest emerged as the best performer, achieving an accuracy of 84.38% and a ROC-AUC score of 0.912, demonstrating strong performance in distinguishing between high and low wine quality. While some false positives occurred, the model maintained a good balance between precision and recall across the classes.

Finally, we deployed the Random Forest model using Flask, enabling local predictions. This deployment provides a solid foundation for future implementation in a production environment. The project showcases how robust preprocessing and model selection can lead to effective predictions, offering potential business value in classifying wine quality.
