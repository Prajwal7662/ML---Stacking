Stacking (Stacked Generalization) in Machine Learning — Theory

Stacking, or stacked generalization, is an ensemble learning technique that combines predictions from multiple models (called base learners) using another model (called a meta-learner or level-1 model).
It is applied to both classification and regression tasks to improve predictive performance by reducing bias and variance.

🔹 Core Concept

Different machine learning algorithms have different strengths and weaknesses.
Stacking leverages this diversity — it trains several base models on the same data and then uses another model to learn the best way to combine their outputs.
This meta-model learns how to correct the mistakes of the base models, producing stronger final predictions.

🔹 Working Mechanism

Train Base Models (Level-0)
Several diverse algorithms (e.g., Decision Tree, Random Forest, SVM, Logistic Regression, etc.) are trained on the training data.

Generate Meta-Features
Each base model predicts on unseen (out-of-fold) data from cross-validation.
These predictions become new input features for the meta-model.

Train Meta-Model (Level-1)
A meta-learner (e.g., Logistic Regression, Ridge Regression, or LightGBM) is trained on the predictions from base models.

Final Prediction
On unseen test data, each base model generates predictions that are fed into the meta-model, which outputs the final prediction.

This process ensures that the meta-model learns patterns in how base model predictions relate to the true outcome.

🔹 Stacking for Classification
Concept

In classification, the goal is to combine outputs (often probabilities) of multiple classifiers to improve class prediction accuracy.
The meta-learner typically learns how to weight or combine the class probabilities produced by base classifiers.

Example Base Models

Logistic Regression

Decision Tree

Random Forest

Support Vector Machine (SVM)

Gradient Boosting Classifier

Meta-Model Choices

Logistic Regression (common choice)

Naïve Bayes

Gradient Boosting or Neural Network (for complex problems)

Characteristics

Base models produce class probabilities using predict_proba().

The meta-learner combines these probabilities to make a final class decision.

Stacking helps capture nonlinear relationships between model outputs.

Example Flow
Base Models: SVM, Random Forest, KNN  
Meta-Model: Logistic Regression  

Training:
- Train base models using K-fold CV.
- Collect out-of-fold probabilities.
- Train logistic regression on these probabilities.

Prediction:
- Generate base model probabilities for new data.
- Feed them to the meta-model to get final prediction.

Advantages in Classification

Reduces classification errors.

Combines models that perform differently on various feature subsets.

More stable than individual classifiers in noisy datasets.

🔹 Stacking for Regression
Concept

In regression, stacking combines predictions from multiple regressors to improve the accuracy of continuous value predictions.
The meta-learner learns how to weight and adjust the outputs of base regressors.

Example Base Models

Linear Regression

Decision Tree Regressor

Random Forest Regressor

Gradient Boosting Regressor

SVR (Support Vector Regressor)

Meta-Model Choices

Ridge Regression

Lasso Regression

Linear Regression

LightGBM or XGBoost (for nonlinear combinations)

Characteristics

Base models output continuous predictions.

The meta-learner learns a linear or nonlinear combination of these outputs.

Stacking reduces both bias (underfitting) and variance (overfitting).

Example Flow
Base Models: Random Forest, Gradient Boosting, Linear Regression  
Meta-Model: Ridge Regression  

Training:
- Train base regressors using K-fold CV.
- Collect out-of-fold predictions.
- Train Ridge model on these predictions.

Prediction:
- Generate base regressor outputs for new data.
- Feed them into Ridge model for final prediction.

Advantages in Regression

Reduces prediction errors and improves R².

Smooths out individual model fluctuations.

Handles nonlinear relationships between features and targets effectively.

🔹 Benefits of Stacking (General)

Improves accuracy compared to individual models.

Combines the strengths of multiple algorithms.

Reduces both variance and bias.

Works for both classification and regression problems.

🔹 Limitations

Computationally expensive (multiple models trained).

Risk of data leakage if out-of-fold predictions are not properly handled.

More difficult to interpret than single models.

Requires careful cross-validation to ensure generalization.

🔹 Best Practices

Use diverse base models (different algorithms or architectures).

Always generate out-of-fold predictions to train the meta-model.

Keep the meta-learner simple (e.g., linear or logistic regression).

Evaluate using nested cross-validation for robust performance estimates.

Use regularization to prevent overfitting in the meta-model.
