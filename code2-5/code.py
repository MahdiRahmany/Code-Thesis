# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from ucimlrepo import fetch_ucirepo

# ---------------------------
# تنظیمات تکرارپذیری
# ---------------------------
SEED = 42
rng = np.random.default_rng(SEED)
np.random.seed(SEED)

# ---------------------------
# پارامترهای BAOA-SA
# ---------------------------
N = 50        # اندازه‌ی جمعیت
T_max = 100   # حداکثر تکرارها
alpha = 0.5   # پارامتر MOP
T0 = 100.0    # دمای اولیه شبیه‌سازی تبرید
beta = 0.95   # نرخ خنک‌سازی
min_moa = 0.2
max_moa = 1.0
epsilon = 1e-8   # برای جلوگیری از تقسیم بر صفر
CLIP = 8.0       # کلیپ کردن بردارهای پیوسته برای پایداری سیگموید

# ---------------------------
# توابع کمکی
# ---------------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def to_binary_mask(vec: np.ndarray, stochastic: bool = False, thresh: float = 0.5) -> np.ndarray:
    """
    vec: بردار پیوسته (نمره‌ی هر ویژگی) یا باینری.
    اگر stochastic=True باشد، باینری‌سازی تصادفی بر اساس سیگموید انجام می‌شود.
    """
    v = np.asarray(vec, dtype=float)
    if stochastic:
        mask = sigmoid(v) > np.random.rand(v.size)
    else:
        mask = sigmoid(v) > thresh  # آستانه‌ی ثابت
    # تضمین حداقل یک ویژگی
    if not mask.any():
        mask[np.random.randint(0, v.size)] = True
    return mask

def fitness(selected_mask: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """
    تابع برازش: acc - (نسبت تعداد ویژگی‌های انتخاب‌شده)
    """
    mask = np.asarray(selected_mask, dtype=bool)
    if mask.sum() == 0:
        return -np.inf

    X_sel = X[:, mask]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sel, y, test_size=0.2, random_state=SEED, stratify=y
    )
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    acc = accuracy_score(y_te, pred)

    penalty = mask.sum() / X.shape[1]
    return acc - penalty

# ---------------------------
# بارگذاری و پیش‌پردازش داده
# ---------------------------
heart_disease = fetch_ucirepo(id=45)
X_df = heart_disease.data.features.copy()
y_ser = heart_disease.data.targets['num'].copy()

feature_names = list(X_df.columns)

# تبدیل برچسب‌ها به دودویی: 0 (سالم)، 1 (بیمار)
y_bin = (y_ser.values > 0).astype(int)

# مدیریت مقادیر گمشده و نرمال‌سازی
imputer = SimpleImputer(strategy='mean')
scaler = MinMaxScaler()
X_num = imputer.fit_transform(X_df.values)
X_num = scaler.fit_transform(X_num)

# بالانس با SMOTE
smote = SMOTE(random_state=SEED)
X_balanced, y_balanced = smote.fit_resample(X_num, y_bin)

d = X_balanced.shape[1]  # تعداد ویژگی‌ها

# ---------------------------
# مقداردهی اولیه جمعیت و کش برازش
# ---------------------------
population = np.random.uniform(-1.0, 1.0, size=(N, d))  # بردارهای پیوسته
current_masks = np.array([to_binary_mask(population[i], stochastic=False, thresh=0.5)
                          for i in range(N)], dtype=bool)
current_fits = np.array([fitness(current_masks[i], X_balanced, y_balanced) for i in range(N)])

best_idx = int(np.argmax(current_fits))
best_fitness = float(current_fits[best_idx])
best_solution = current_masks[best_idx].copy()

T = T0

# ---------------------------
# حلقه‌ی اصلی BAOA-SA
# ---------------------------
for t in range(T_max):
    moa = min_moa + t * (max_moa - min_moa) / T_max
    # Mop مطابق فرمول مقاله (تقریب چندهدفه)
    mop = 1.0 - (t ** (1.0 / alpha) / (T_max ** (1.0 / alpha)))

    for i in range(N):
        # انتخاب عملگر حسابی تصادفی
        operator = np.random.choice(['+', '-', '*', '/'])
        r = np.random.rand(d)

        if operator == '+':
            update = population[i] + mop * r
        elif operator == '-':
            update = population[i] - mop * r
        elif operator == '*':
            update = population[i] * (mop * r)
        else:  # '/'
            update = population[i] / (mop * r + epsilon)

        # پایداری: کلیپ‌کردن برای جلوگیری از اشباع شدید سیگموید
        update = np.clip(update, -CLIP, CLIP)

        # باینری‌سازی کاندید جدید (استوکستیک → تنوع بهتر)
        binary_update = to_binary_mask(update, stochastic=True)

        # محاسبه‌ی برازش کاندید
        new_fit = fitness(binary_update, X_balanced, y_balanced)

        # SA: مقایسه با حالت فعلی (از کش)
        delta = new_fit - current_fits[i]
        accept = (delta > 0) or (np.exp(delta / max(T, 1e-12)) > np.random.rand())

        if accept:
            population[i] = update
            current_masks[i] = binary_update
            current_fits[i] = new_fit

            # به‌روزرسانی بهترین
            if new_fit > best_fitness:
                best_fitness = float(new_fit)
                best_solution = binary_update.copy()

    # خنک‌سازی
    T *= beta

# ---------------------------
# نتایج
# ---------------------------
selected_idx = np.where(best_solution)[0]
selected_names = [feature_names[j] for j in selected_idx]

print(f"بهترین fitness: {best_fitness:.6f}")
print(f"تعداد ویژگی‌های انتخاب‌شده: {best_solution.sum()} از {d}")
print("نمایه‌های ویژگی‌های انتخاب‌شده:", selected_idx.tolist())
print("نام ویژگی‌های انتخاب‌شده:", selected_names)
