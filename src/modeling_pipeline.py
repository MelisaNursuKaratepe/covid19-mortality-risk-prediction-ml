import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


DATA_PATH = "data/Covid19Data.csv"
FIGURES_PATH = "figures/"
TEST_SIZE = 0.2
RANDOM_STATE = 42


numeric_features = ["AGE"]

categorical_features = [
    "USMER",
    "MEDICAL_UNIT",
    "SEX",
    "PATIENT_TYPE",
    "CLASIFFICATION_FINAL",
]

binary_features = [
    "PNEUMONIA",
    "PREGNANT",
    "DIABETES",
    "COPD",
    "ASTHMA",
    "INMSUPR",
    "HIPERTENSION",
    "OTHER_DISEASE",
    "CARDIOVASCULAR",
    "OBESITY",
    "RENAL_CHRONIC",
    "TOBACCO",
]


# ============
# Data Loading
# ============
df = pd.read_csv(DATA_PATH)


# ===================
# Feature Engineering
# ===================

## Removal of Potential Data Leakage Features
df = df.drop(columns=["INTUBED", "ICU"])


## Pregnancy Variable Preprocessing
df["PREGNANT"] = df["PREGNANT"].replace([97, 98], 2)  # 1: pregnant, 2: not pregnant


## Filtering Uncertain and Invalid Values
mask = df[binary_features].isin([1, 2]).all(axis=1)
df = df[mask]


## Binary Feature Encoding
df[binary_features] = df[binary_features].replace(2, 0)


# ============================
# Target Variable Construction
# ============================
df["IS_DEAD"] = (df["DATE_DIED"] != "9999-99-99").astype(int)   # 1: Deceased, 0: Alive
df = df.drop(columns=["DATE_DIED"])


# ===============================
# Exploratory Data Analysis (EDA)
# ===============================

## Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    fmt=".1f",
    cmap="coolwarm",
    linewidths=0.5,
)
plt.title("Correlation Heatmap", fontsize=16)
plt.savefig(f"{FIGURES_PATH}correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()


## Age Distribution
plt.figure(figsize=(8, 6))
sns.histplot(df["AGE"], bins=30, kde=True, color='skyblue')
plt.title("Age Distribution", fontsize=16)
plt.xlabel("Age")
plt.ylabel("Number of Patients")
plt.savefig(f"{FIGURES_PATH}age_distribution.png", dpi=300, bbox_inches="tight")
plt.close()


# =============================
# Data Preparation for Modeling
# =============================

## Feature–Target Split
X = df.drop(columns=["IS_DEAD"])
y = df["IS_DEAD"]


## Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)


# ======================================================
# Pipeline-Based Baseline Modeling (Logistic Regression)
# ======================================================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("bin", "passthrough", binary_features),
        ("cat", "passthrough", categorical_features),
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

pipeline.fit(X_train, y_train)


# ================
# Model Evaluation
# ================
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("\n\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nROC-AUC Score:")
print(roc_auc_score(y_test, y_prob))
print("\n\n")


## Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix", fontsize=16)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(f"{FIGURES_PATH}confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()
