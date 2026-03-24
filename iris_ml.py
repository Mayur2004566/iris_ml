import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ── Load Dataset ──────────────────────────────────────────────
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({
    0: 'setosa',
    1: 'versicolor',
    2: 'virginica'
})

print("Dataset Loaded Successfully!\n")
print(df.head())

# ── Features & Target ─────────────────────────────────────────
X = df.drop('species', axis=1)
y = df['species']

# ── Train-Test Split ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Model Training ────────────────────────────────────────────
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# ── Prediction ────────────────────────────────────────────────
y_pred = model.predict(X_test)

# ── Evaluation ────────────────────────────────────────────────
print("\n" + "=" * 40)
print("MODEL PERFORMANCE")
print("=" * 40)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ── Visualization 1: Pairplot ────────────────────────────────
sns.pairplot(df, hue='species')
plt.suptitle("Iris Dataset Pairplot", y=1.02)
plt.savefig('iris_pairplot.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Visualization 2: Species Count ───────────────────────────
plt.figure(figsize=(8, 5))
sns.countplot(x='species', hue='species', data=df, palette='Set2', legend=False)
plt.title("Iris Species Count")
plt.savefig('iris_countplot.png', dpi=150, bbox_inches='tight')
plt.close()

