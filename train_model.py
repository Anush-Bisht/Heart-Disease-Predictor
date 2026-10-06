"""Run once:  python train_model.py   (heart.csv must be in this folder)"""
import joblib, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv("heart.csv")

# Same cleaning idea as your notebook: 0 is a missing value for these columns
means = {}
for col in ["Cholesterol", "RestingBP"]:
    means[col] = float(df.loc[df[col] != 0, col].mean())
    df[col] = df[col].replace(0, means[col])

num = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
cat = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
features = num + ["FastingBS"] + cat

pre = ColumnTransformer([
    ("num", StandardScaler(), num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
], remainder="passthrough")  # FastingBS passes through as 0/1

pipe = Pipeline([("pre", pre), ("svm", SVC(probability=True, random_state=42))])

X, y = df[features], df["HeartDisease"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
pipe.fit(X_tr, y_tr)
pred = pipe.predict(X_te)
acc, f1 = accuracy_score(y_te, pred), f1_score(y_te, pred)
print(f"Hold-out accuracy: {acc:.4f}  F1: {f1:.4f}")

# 5-fold cross-validation on all data: a more reliable accuracy estimate
cv = StratifiedKFold(5, shuffle=True, random_state=42)
cv_acc = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy").mean()
print(f"Cross-validated accuracy: {cv_acc:.4f}")
pipe.fit(X, y)  # final model trained on all available data

joblib.dump({"pipeline": pipe, "features": features, "means": means,
             "accuracy": round(cv_acc * 100, 1)}, "model.pkl")
print("Saved model.pkl")
