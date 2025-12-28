import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# بارگذاری داده‌ست
iris = load_iris()
X, y = iris.data, iris.target

# فرض: selected_features یک بردار باینری از BAOA-SA (اینجا همه ویژگی‌ها برای نمونه)
selected_features = np.ones(X.shape[1], dtype=bool)

# 10-fold cross-validation
skf = StratifiedKFold(n_splits=10)
acc_scores, f1_scores, roc_scores = [], [], []

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X[train_idx][:, selected_features], X[test_idx][:, selected_features]
    y_train, y_test = y[train_idx], y[test_idx]
    
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test) if len(np.unique(y)) > 2 else None  # برای ROC
    
    acc_scores.append(accuracy_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred, average='macro'))
    if y_prob is not None:
        roc_scores.append(roc_auc_score(y_test, y_prob, multi_class='ovr'))

print(f"میانگین Accuracy: {np.mean(acc_scores):.4f}")
print(f"میانگین F1-Score: {np.mean(f1_scores):.4f}")
print(f"میانگین ROC-AUC: {np.mean(roc_scores):.4f}")