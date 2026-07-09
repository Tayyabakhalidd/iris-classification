
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["species"] = iris.target


print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nLast 5 rows:")
print(df.tail())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())
print("\nBasic statistics:")
print(df.describe())


X = df.drop("species", axis=1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nTraining size:", X_train.shape)
print("Testing size:", X_test.shape)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nFirst 5 scaled rows:")
print(X_train[:5])


model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

print("\nModel trained successfully!")


predictions = model.predict(X_test)

print("\nPredictions:")
print(predictions)
print("\nActual answers:")
print(y_test.values)


print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=iris.target_names
))


cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Purples",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)
plt.title("Confusion Matrix — Iris Classifier")
plt.ylabel("Actual Species")
plt.xlabel("Predicted Species")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

print("\nConfusion matrix saved!")
print("\nProject 2 Complete! 🎉")