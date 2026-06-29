# Machine Learning Projects Portfolio

A collection of 6 end-to-end Machine Learning notebooks, each covering a different algorithm and real-world use case — from data cleaning and EDA to model building, tuning, and evaluation.

| # | Project | Algorithm | Problem Type |
|---|---|---|---|
| 1 | Insurance Charges Prediction | Linear Regression | Regression |
| 2 | Titanic Survival Prediction | Logistic Regression | Binary Classification |
| 3 | Loan Applicant Credit Risk Analysis | K-Nearest Neighbors (KNN) | Classification |
| 4 | Loan Default Status Classification | Naïve Bayes (GaussianNB) | Binary Classification |
| 5 | Cardiac Arrest Risk Grouping | K-Means & Hierarchical Clustering | Unsupervised Clustering |
| 6 | Hotel Booking Cancellation Prediction | Random Forest | Binary Classification |
| 7 | Airline Customer Satisfaction | Decision Tree | Binary Classification |

---

## 1. `Linear_Regression_Insurance_Prediction.ipynb`
**Goal:** Predict medical insurance `charges` based on personal attributes.

**Key Steps:**
- Handled missing values (`children` filled with 0, `bmi` filled with column mean)
- Visualized relationships: Age vs Children scatter plot, BMI vs Children bar plot
- Label-encoded categorical columns: `sex`, `smoker`, `region`
- Standardized all numeric features using `StandardScaler`
- Split data 80/20 and trained a `LinearRegression` model
- Evaluated using **MAE, MSE, RMSE, and R² score**, and inspected model coefficients/intercept

**What this demonstrates:** Regression workflow, feature scaling, and interpreting linear model coefficients for a continuous target.

---

## 2. `Logistic_Regression_Titanic_Survival.ipynb`
**Goal:** Predict passenger `Survived` status on the Titanic dataset.

**Key Steps:**
- Filled missing `Age` with median, `Embarked` with mode, dropped high-null `Cabin` column
- One-hot encoded `Sex` and `Embarked` (`drop_first=True`)
- Dropped non-predictive columns: `PassengerId`, `Name`, `Ticket`
- Trained a `LogisticRegression` classifier (80/20 split)
- Evaluated with **Accuracy, Precision, Recall, F1 Score, Confusion Matrix**
- Plotted **ROC Curve** and computed **AUC Score** using predicted probabilities

**What this demonstrates:** Classic binary classification pipeline with probability-based evaluation (ROC/AUC).

---

## 3. `_KNN_Loan_Applicant_Credit_Risk_Analysis.ipynb`
**Goal:** Predict `Total bounces past 12 months` for loan applicants (credit risk indicator) using KNN.

**Key Steps:**
- Removed duplicate records, checked for nulls
- Visualized Age vs Total Work Experience (scatter), and boxplots for `Age` and `Cibil score` to spot outliers
- Standardized features using `StandardScaler`
- Trained a `KNeighborsClassifier` (default k) and evaluated train/test scores + accuracy
- **Hyperparameter exploration:** looped k = 1 to 14, plotted training vs testing accuracy curves to identify the best k and check for over/under-fitting

**What this demonstrates:** Distance-based classification and how to tune the `k` hyperparameter visually using a train/test accuracy curve.

---

## 4. `_Naïve_Bayes_Loan_Status_Classification.ipynb`
**Goal:** Classify loan applicants' `Default Status` using Naïve Bayes.

**Key Steps:**
- Identified and filled null numeric columns with their mean; dropped any remaining nulls
- Visualized outliers via boxplots for `LIMIT_BAL` and `AGE`
- Label-encoded the target column `Default Status`
- Split data 80/20 and trained a `GaussianNB` classifier
- Evaluated with **Accuracy, Precision, Recall, Confusion Matrix, Classification Report**

**What this demonstrates:** Probabilistic classification using Naïve Bayes on financial/credit data.

---

## 5. `ML_Project_Clustering_Cardiac_Arrest_Predictio.ipynb`
**Goal:** Discover natural risk groupings among patients (unsupervised) related to cardiac arrest risk.

**Key Steps:**
- Dropped the labeled column `UnderRisk` to keep the analysis unsupervised
- **K-Means Clustering** with `n_clusters=3`; inspected cluster centers and cluster sizes
- **Hierarchical Clustering**: built a dendrogram (Ward linkage) to visualize natural groupings
- Applied `AgglomerativeClustering` (3 clusters) and compared with K-Means
- Evaluated cluster quality using **Silhouette Score**
- Looped through k = 2 to 11 to assess clustering performance across different cluster counts

**What this demonstrates:** Unsupervised pattern discovery, comparing centroid-based (K-Means) vs hierarchical clustering, and using silhouette score for cluster validation.

---

## 6. `Random_Forest_Hotel_Cancellation_Pre.ipynb`
**Goal:** Predict whether a hotel booking will be cancelled (`is_canceled`).

**Key Steps:**
- Filled missing categorical values with `'other'`, filled `agent` with mean, dropped remaining nulls
- Visualized total adults vs. children with a bar chart
- Label-encoded all categorical columns
- Dropped target-leakage column `reservation_status`; defined features/target
- Trained a `RandomForestClassifier` (80/20 split)
- Evaluated using **Classification Report, Precision, Recall, Confusion Matrix**

**What this demonstrates:** Ensemble tree-based classification on a real-world booking dataset, including care taken to remove a feature that would leak the target.

---

## 7. `_Decision_Tree_Airline_Customer_Satisfaction.ipynb`
**Goal:** Predict airline passenger `satisfaction` (satisfied vs neutral/dissatisfied).

**Key Steps:**
- Dropped redundant index columns, cleaned column names
- Mapped target to binary (0/1) and explored satisfaction trends by Gender, Age, Food & Drink rating
- Used boxplots to inspect `Flight_Distance` and `Checkin_service` for outliers
- Label/ordinal-encoded categorical columns (`Gender`, `Customer_Type`, `Type_of_Travel`, `Class`)
- Trained a baseline `DecisionTreeClassifier`, then a **tuned** version (`max_depth=5`, `min_samples_split=10`, `min_samples_leaf=5`) to reduce overfitting
- Evaluated with **Accuracy, Precision, Recall, F1 Score, Confusion Matrix, Classification Report**

**What this demonstrates:** Tree-based classification with explicit overfitting control via hyperparameter tuning, plus before/after comparison of train vs test scores.

---

## 🛠️ Common Tech Stack (across all projects)
- **Language:** Python 3.x
- **Core Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`
- **ML Library:** `scikit-learn` (preprocessing, model_selection, linear_model, tree, ensemble, neighbors, naive_bayes, cluster, metrics)
- Optional: `scipy` (for hierarchical clustering / dendrograms)

Install all dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

## 📌 General Workflow Pattern Used in Every Project
1. **Load Data** — read CSV/Excel/TXT file
2. **Explore & Clean** — check nulls, handle missing values, drop irrelevant/leaky columns
3. **Visualize** — scatter plots, boxplots, bar charts to understand distributions and outliers
4. **Encode** — Label Encoding / One-Hot Encoding for categorical features
5. **Scale** (where relevant) — `StandardScaler` for distance-based or coefficient-based models
6. **Split** — train/test split (commonly 80/20)
7. **Train Model** — algorithm-specific model from scikit-learn
8. **Evaluate** — accuracy/precision/recall/F1/confusion matrix (classification) or MAE/MSE/RMSE/R² (regression), or silhouette score (clustering)
9. **Tune (where applicable)** — hyperparameter adjustments to address overfitting (e.g., Decision Tree depth, KNN's k value)

## 📌 Notes
- None of the raw dataset files are included in this repository — each notebook references a local file path that should be updated to point to your own copy of the dataset before running.
- A few notebooks contain minor inconsistent variable casing (e.g., `X_test` vs `x_test`) inherited from the original source files — fix these to match your variable names if you re-run the notebooks as-is.
