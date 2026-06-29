🤖 Machine Learning Projects Portfolio

A collection of end-to-end ML notebooks — from raw data to evaluated models

📂 Project Index

🧠 Project⚙️ Algorithm🎯 Problem Type1Insurance Charges PredictionLinear Regression📈 Regression2Titanic Survival PredictionLogistic Regression🧮 Binary Classification3Loan Applicant Credit Risk AnalysisK-Nearest Neighbors🧮 Classification4Loan Default Status ClassificationNaïve Bayes🧮 Binary Classification5Cardiac Arrest Risk GroupingK-Means / Hierarchical Clustering🔍 Unsupervised Clustering6Hotel Booking Cancellation PredictionRandom Forest🧮 Binary Classification7Airline Customer SatisfactionDecision Tree🧮 Binary Classification

1️⃣ Linear_Regression_Insurance_Prediction.ipynb

🎯 Goal: Predict medical insurance charges based on personal attributes.

🔑 Key Steps

🧹 Handled missing values (children → filled with 0, bmi → filled with column mean) 📊 Visualized relationships: Age vs Children scatter plot, BMI vs Children bar plot 🔤 Label-encoded categorical columns: sex, smoker, region ⚖️ Standardized all numeric features using StandardScaler ✂️ Split data 80/20 and trained a LinearRegression model 📐 Evaluated using MAE · MSE · RMSE · R² Score, and inspected model coefficients/intercept

💡 Demonstrates: Regression workflow, feature scaling, and interpreting linear model coefficients for a continuous target.

2️⃣ Logistic_Regression_Titanic_Survival.ipynb

🎯 Goal: Predict passenger Survived status on the Titanic dataset.

🔑 Key Steps

🧹 Filled missing Age with median, Embarked with mode; dropped high-null Cabin column 🔤 One-hot encoded Sex and Embarked (drop_first=True) 🗑️ Dropped non-predictive columns: PassengerId, Name, Ticket 🤖 Trained a LogisticRegression classifier (80/20 split) 📐 Evaluated with Accuracy · Precision · Recall · F1 Score · Confusion Matrix 📉 Plotted ROC Curve and computed AUC Score using predicted probabilities

💡 Demonstrates: Classic binary classification pipeline with probability-based evaluation (ROC/AUC).

3️⃣ _KNN_Loan_Applicant_Credit_Risk_Analysis.ipynb

🎯 Goal: Predict Total bounces past 12 months for loan applicants (credit risk indicator).

🔑 Key Steps

🗑️ Removed duplicate records, checked for nulls 📊 Visualized Age vs Total Work Experience (scatter); boxplots for Age & Cibil score to spot outliers ⚖️ Standardized features using StandardScaler 🤖 Trained a KNeighborsClassifier (default k); evaluated train/test scores + accuracy 🔁 Hyperparameter exploration: looped k = 1 → 14, plotted training vs testing accuracy curves to identify the best k and check for over/under-fitting

💡 Demonstrates: Distance-based classification and visual tuning of the k hyperparameter via accuracy curves.

4️⃣ _Naïve_Bayes_Loan_Status_Classification.ipynb

🎯 Goal: Classify loan applicants' Default Status using Naïve Bayes.

🔑 Key Steps

🧹 Identified & filled null numeric columns with their mean; dropped remaining nulls 📊 Visualized outliers via boxplots for LIMIT_BAL and AGE 🔤 Label-encoded the target column Default Status ✂️ Split data 80/20 and trained a GaussianNB classifier 📐 Evaluated with Accuracy · Precision · Recall · Confusion Matrix · Classification Report

💡 Demonstrates: Probabilistic classification using Naïve Bayes on financial/credit data.

5️⃣ ML_Project_Clustering_Cardiac_Arrest_Predictio.ipynb

🎯 Goal: Discover natural risk groupings among patients (unsupervised) related to cardiac arrest risk.

🔑 Key Steps

🗑️ Dropped the labeled column UnderRisk to keep the analysis unsupervised 🌀 K-Means Clustering with n_clusters=3; inspected cluster centers & sizes 🌳 Hierarchical Clustering: built a dendrogram (Ward linkage) to visualize natural groupings 🌀 Applied AgglomerativeClustering (3 clusters) and compared with K-Means 📐 Evaluated cluster quality using Silhouette Score 🔁 Looped k = 2 → 11 to assess clustering performance across cluster counts

💡 Demonstrates: Unsupervised pattern discovery — comparing centroid-based (K-Means) vs hierarchical clustering, validated via silhouette score.

6️⃣ Random_Forest_Hotel_Cancellation_Pre.ipynb

🎯 Goal: Predict whether a hotel booking will be cancelled (is_canceled).

🔑 Key Steps

🧹 Filled missing categorical values with 'other', filled agent with mean, dropped remaining nulls 📊 Visualized total adults vs. children with a bar chart 🔤 Label-encoded all categorical columns ⚠️ Dropped target-leakage column reservation_status; defined features/target 🤖 Trained a RandomForestClassifier (80/20 split) 📐 Evaluated using Classification Report · Precision · Recall · Confusion Matrix

💡 Demonstrates: Ensemble tree-based classification, including careful removal of a feature that would leak the target.

7️⃣ _Decision_Tree_Airline_Customer_Satisfaction.ipynb

🎯 Goal: Predict airline passenger satisfaction (satisfied vs neutral/dissatisfied).

🔑 Key Steps

🗑️ Dropped redundant index columns, cleaned column names 🔢 Mapped target to binary (0/1); explored satisfaction trends by Gender, Age, Food & Drink rating 📊 Boxplots to inspect Flight_Distance and Checkin_service for outliers 🔤 Label/ordinal-encoded categorical columns (Gender, Customer_Type, Type_of_Travel, Class) 🤖 Trained a baseline DecisionTreeClassifier, then a tuned version (max_depth=5, min_samples_split=10, min_samples_leaf=5) to reduce overfitting 📐 Evaluated with Accuracy · Precision · Recall · F1 Score · Confusion Matrix · Classification Report

💡 Demonstrates: Tree-based classification with explicit overfitting control via hyperparameter tuning, plus before/after train vs test comparison.

🛠️ Common Tech Stack

CategoryToolsLanguagePython 3.xCore Librariespandas · numpy · matplotlib · seabornML Libraryscikit-learn (preprocessing, model_selection, linear_model, tree, ensemble, neighbors, naive_bayes, cluster, metrics)Optionalscipy (hierarchical clustering / dendrograms)
📦 Install all dependencies:
bashpip install pandas numpy matplotlib seaborn scikit-learn scipy

🔄 General Workflow Pattern (Used in Every Project)

📥 Load Data → Read CSV / Excel / TXT file
🧹 Explore & Clean → Check nulls, handle missing values, drop irrelevant/leaky columns
📊 Visualize → Scatter plots, boxplots, bar charts for distributions & outliers
🔤 Encode → Label Encoding / One-Hot Encoding for categorical features
⚖️ Scale (if needed) → StandardScaler for distance/coefficient-based models
✂️ Split → Train/test split (commonly 80/20)
🤖 Train Model → Algorithm-specific model from scikit-learn
📐 Evaluate → Accuracy / Precision / Recall / F1 / Confusion Matrix (or MAE / MSE / RMSE / R² for regression, or Silhouette Score for clustering)
🔧 Tune (if applicable) → Hyperparameter adjustments to address overfitting
📌 Notes

⚠️ Dataset files are not included. Each notebook references a local file path — update it to point to your own copy of the dataset before running.
