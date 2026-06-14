# Diabetes Prediction from Scratch using KNN

An elegant and lightweight implementation of the **K-Nearest Neighbors (KNN)** classification algorithm, built entirely from scratch using native Python to predict diabetes based on clinical patient data. 

By implementing custom data preprocessing and cleaning techniques without relying on external machine learning libraries for the core algorithm, the model achieves a highly stable predictive accuracy of **85.43%**.

##  Project Overview
The primary objective of this project is to demonstrate how a fundamental distance-based machine learning algorithm operates at a low level. It utilizes the widely referenced **Pima Indians Diabetes Database** to classify whether a patient has diabetes based on 8 diagnostic measurements.

### Key Constraints & Features
- **Zero ML Libraries for Core Logic:** The distance calculation, neighbor sorting, and majority voting mechanisms are implemented using raw Python (no `scikit-learn` or `pandas` used in the algorithm).
- **Advanced Data Cleaning:** Biologically impossible zero values (e.g., in Glucose and BMI) are systematically filtered out to eliminate noise.
- **Custom Min-Max Normalization:** Features are scaled manually to a `[0, 1]` range to prevent larger numerical fields (like Insulin) from biasing the Euclidean distance calculations.

---

##  Performance & Evaluation
Through hyperparameter tuning, the optimal number of neighbors was determined to be **K = 19**, yielding the following performance metrics evaluated on a 20% stratified test set:

- **Accuracy:** `85.43%` (Successfully cleared the 85% target threshold)
- **Recall (Sensitivity):** `74.51%` (Critical for minimizing False Negatives in clinical screening)

### Confusion Matrix
| N = 151 (Test Set) | Predicted: Healthy (0) | Predicted: Diabetic (1) |
|---|---|---|
| **Actual: Healthy (0)** | **91** (True Negative) | **9** (False Positive) |
| **Actual: Diabetic (1)** | **13** (False Negative) | **38** (True Positive) |

---

##  Methodology & Mathematical Framework
1. **Euclidean Distance:** Multi-dimensional geometric distance is hardcoded as:
   $$\delta = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$$
2. **Sorting & Selection:** Neighbors are stored in a structured list, sorted in ascending order based on proximity, and the top $K$ instances are selected.
3. **Majority Voting:** The final class is predicted using a frequency-based voting mechanism on the selected neighbors' outcomes.

---

##  Project Structure
```text
├── diabetes.csv          # Pima Indians Diabetes dataset
├── knn_model.py          # Pure Python KNN implementation & visualization script
└── README.md             # Project documentation
