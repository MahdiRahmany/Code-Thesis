import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from ucimlrepo import fetch_ucirepo

# بارگذاری داده‌ست Heart Disease
heart_disease = fetch_ucirepo(id=45)
X = heart_disease.data.features
y = heart_disease.data.targets['num']  # کلاس‌ها (0: سالم، 1: بیمار)

# گام 1: مدیریت مقادیر گمشده (با میانگین)
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# گام 2: نرمال‌سازی min-max
scaler = MinMaxScaler()
X_normalized = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

# گام 3: حذف ناهنجاری‌ها (IQR برای هر ویژگی)
Q1 = X_normalized.quantile(0.25)
Q3 = X_normalized.quantile(0.75)
IQR = Q3 - Q1
X_no_outliers = X_normalized[~((X_normalized < (Q1 - 1.5 * IQR)) | (X_normalized > (Q3 + 1.5 * IQR))).any(axis=1)]

# به‌روزرسانی y بر اساس شاخص‌های جدید
y_no_outliers = y[X_no_outliers.index]

# گام 4: متعادل‌سازی با SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_no_outliers, y_no_outliers)

print(f"شکل داده‌ها پس از پیش‌پردازش: {X_balanced.shape}")
print(f"تعداد کلاس‌ها: {np.bincount(y_balanced)}")