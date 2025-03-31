### Project Description: Impact of Outliers and Null Values on Deep Learning Models and GLM of H2O AutoML: Analysis of the Completeness and Consistency of the Dataset

**Project Objective:**

This project, carried out as part of the Data Architectures course, aims to perform an in-depth analysis of the impact that outliers (anomalous values) and null values have on the performance of specific machine learning models and on fundamental dimensions of data quality, such as **completeness** and **consistency**.

**Dataset Used:**

The analysis was conducted on the "Customer Analytics" dataset (sourced from Kaggle). This dataset aggregates information related to customers and transactions of an e-commerce business, including features like: mode of shipment, cost of the product, customer care calls, customer rating, prior purchases, product importance, discount offered, and weight in grams. The specific modeling goal was a **binary classification** task: predicting whether the shipment arrived on time (`Reached on time` - 1) or late (0).

**Methodology Adopted:**

1.  **Preprocessing:** The dataset's categorical features were converted into numerical values using Scikit-Learn's `LabelEncoder` to make them processable by the models.
2.  **Artificial Anomaly Generation:** To study the impact in a controlled manner, several modified datasets were created from the original one:
    * Datasets with **null values** randomly inserted in increasing percentages (5%, 10%, 15%, 20%).
    * Datasets with **outliers** artificially introduced in increasing percentages (5%, 10%, 15%, 20%). Outliers were generated outside the interquartile range (IQR) of the numerical features.
    * Datasets with a **combination** of both null values and outliers in the same percentages.
3.  **Modeling with H2O AutoML:** The H2O AutoML platform was employed to automate the model training and evaluation process. Specifically, two types of models known for their potential sensitivity to anomalous data were selected and compared:
    * **Generalized Linear Model (GLM):** Used in its logistic regression form for binary classification.
    * **Deep Learning:** Implemented as a multi-layer feedforward neural network (FNN/MLP).
4.  **Evaluation:** The performance of the models trained on the different datasets (original, with nulls, with outliers, mixed) was systematically compared using standard classification metrics: AUC, Logloss, AUCPR, RMSE, and MSE.

**Technologies Used:**

* **Language:** Python
* **Core Libraries:**
    * `Pandas`: For data manipulation and analysis.
    * `NumPy`: For numerical operations.
    * `Scikit-Learn`: For preprocessing (`LabelEncoder`, `train_test_split`).
    * `H2O AutoML`: For automated training, selection, and evaluation of GLM and Deep Learning models.

**Key Results Obtained:**

* **Overall Impact:** It was confirmed that both the presence of outliers and null values **negatively** affect the performance of GLM and Deep Learning models, as evidenced by the deterioration of evaluation metrics (e.g., decrease in AUC and AUCPR, increase in Logloss and MSE).
* **Outliers vs. Null Values (Performance):** Comparative analysis showed that **outliers** have a **more pronounced and detrimental** impact on the performance of the examined GLM and Deep Learning models compared to null values, given the same percentage introduced. Outliers distort model estimates more significantly and affect metrics sensitive to extreme values, like MSE.
* **Outliers vs. Null Values (Data Quality):**
    * **Null values** primarily compromise the dataset's **completeness**, reducing the amount of usable information for training.
    * **Outliers** mainly undermine the dataset's **consistency**, introducing values that can violate defined schemas, logical constraints, or expected correlations between features.
* **Mixed Datasets:** Results on datasets containing both outliers and null values did not show substantial differences compared to those with only outliers, suggesting that the dominant effect on performance degradation is attributable to the outliers in this scenario.
