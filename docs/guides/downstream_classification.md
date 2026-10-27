# Downstream Classification & Authorship Attribution

Gram2Vec vectors are standard 2D numeric matrices packaged as `pandas.DataFrame` objects, making them directly compatible with machine learning libraries like `scikit-learn`, `xgboost`, and `lightgbm`.

---

## Authorship Attribution Workflow

Here is an end-to-end example training a classifier to attribute authorship while retaining full interpretability:

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from gram2vec import Gram2Vec

# 1. Prepare corpus
documents = [
    "Text sample by Author A...",
    "Another piece of writing by Author A...",
    "A distinct writing style from Author B...",
    "Author B typically writes with different syntax...",
]
labels = ["Author A", "Author A", "Author B", "Author B"]

# 2. Extract Gram2Vec features
g2v = Gram2Vec(model="en_core_web_lg")
X = g2v.from_documents(documents)
y = labels

# 3. Train a classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# 4. Interpret top discriminative features
importances = pd.Series(clf.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(10)
print("Top Stylistic Features Distinguishing Authors:")
print(top_features)
```

Because each column in `X` corresponds to an exact grammatical or stylistic construct (e.g., `dep_advcl`, `pos_ADJ`, `syntax_passive`), you can explain precisely *why* the model attributed a text to Author A versus Author B.
