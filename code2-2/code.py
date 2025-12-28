import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# پارامترهای BAOA-SA (بر اساس مقالات)
N = 50  # اندازه جمعیت
T_max = 100  # حداکثر تکرارها
alpha = 0.5  # برای MOP
min_moa = 0.2
max_moa = 1.0
T0 = 100  # دمای اولیه SA
beta = 0.95  # نرخ خنک‌سازی
w_acc = 0.95  # وزن دقت (افزایش برای اولویت دقت)
w_feat = 0.05  # وزن تعداد ویژگی (کاهش برای انتخاب بیشتر ویژگی‌ها)
stagnation_limit = 20  # معیار توقف

# تابع transfer S-shaped (سیگموید) برای باینری‌سازی
def transfer_s(x):
    return 1 / (1 + np.exp(-x))

def binarize(position):
    return np.where(np.random.rand(*position.shape) < transfer_s(position), 1, 0)

# تابع هدف با cross-validation
def fitness(X, y, selected_features):
    if np.sum(selected_features) == 0:
        return 0, 0, 0
    selected_idx = np.where(selected_features == 1)[0]
    clf = KNeighborsClassifier(n_neighbors=5)
    scores = cross_val_score(clf, X[:, selected_idx], y, cv=5, scoring='accuracy')
    acc = scores.mean()
    num_features = np.sum(selected_features)
    combined_fitness = w_acc * acc - w_feat * (num_features / X.shape[1])
    return acc, num_features, combined_fitness

# بارگذاری Iris
iris = load_iris()
X = iris.data
y = iris.target
d = X.shape[1]  # 4 ویژگی

# مقداردهی اولیه جمعیت (پیوسته بین 0-1، سپس باینری)
population = np.random.rand(N, d)
binary_population = binarize(population)
fitness_values = np.zeros(N)
acc_values = np.zeros(N)
feat_values = np.zeros(N)

# فیتنس اولیه
for i in range(N):
    acc, num_feat, combined = fitness(X, y, binary_population[i])
    fitness_values[i] = combined
    acc_values[i] = acc
    feat_values[i] = num_feat

# بهترین اولیه
best_idx = np.argmax(fitness_values)
best_position = population[best_idx].copy()
best_binary = binary_population[best_idx].copy()
best_fitness = fitness_values[best_idx]
best_acc = acc_values[best_idx]
best_num_feat = feat_values[best_idx]
stagnation_count = 0

print("شروع بهینه‌سازی BAOA-SA...")

# حلقه اصلی
T = T0
for t in range(1, T_max + 1):  # از 1 شروع برای MOA/MOP
    moa = min_moa + (t / T_max) * (max_moa - min_moa)
    mop = 1 - (t ** (1 / alpha) / T_max ** (1 / alpha))
    
    improved = False
    for i in range(N):
        r1 = np.random.rand()
        if r1 < moa:  # فاز کاوشگر (Exploration: Division/Multiplication)
            r2 = np.random.rand()
            if r2 >= 0.5:
                new_pos = population[i] / (mop + 1e-10)  # Division
            else:
                new_pos = population[i] * mop  # Multiplication
        else:  # فاز بهره‌برداری (Exploitation: Subtraction/Addition)
            r3 = np.random.rand()
            if r3 >= 0.5:
                new_pos = best_position - mop * (np.random.rand(d) * (best_position - population[i]))  # Subtraction
            else:
                new_pos = best_position + mop * (np.random.rand(d) * (best_position - population[i]))  # Addition
        
        # باینری‌سازی
        new_binary = binarize(new_pos)
        
        # فیتنس جدید
        new_acc, new_num_feat, new_fitness = fitness(X, y, new_binary)
        
        # SA: پذیرش یا رد
        delta = new_fitness - fitness_values[i]
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            population[i] = new_pos
            binary_population[i] = new_binary
            fitness_values[i] = new_fitness
            acc_values[i] = new_acc
            feat_values[i] = new_num_feat
        
        # به‌روزرسانی بهترین
        if new_fitness > best_fitness:
            best_position = new_pos.copy()
            best_binary = new_binary.copy()
            best_fitness = new_fitness
            best_acc = new_acc
            best_num_feat = new_num_feat
            improved = True
    
    # کاهش دما
    T *= beta
    
    # چک stagnation (حالا خارج از حلقه i)
    if improved:
        stagnation_count = 0
    else:
        stagnation_count += 1
        if stagnation_count >= stagnation_limit:
            print(f"توقف در تکرار {t}: فیتنس برای {stagnation_limit} تکرار بهبود نیافت.")
            break
    
    # چاپ پیشرفت
    if t % 10 == 0:
        print(f"تکرار {t}: بهترین دقت = {best_acc:.4f}, تعداد ویژگی = {best_num_feat}, فیتنس = {best_fitness:.4f}")

# نتایج نهایی
print("\nنتایج نهایی برای Iris با BAOA-SA:")
print(f"دقت (cross-validation): {best_acc:.4f}")
print(f"تعداد ویژگی‌های انتخاب‌شده: {best_num_feat} (از {d} ویژگی)")
print(f"فیتنس ترکیبی: {best_fitness:.4f}")
print(f"ویژگی‌های انتخاب‌شده: {best_binary} (1: انتخاب‌شده)")

# ذخیره
np.save('best_features_iris.npy', best_binary)
print("ویژگی‌های انتخاب‌شده در 'best_features_iris.npy' ذخیره شد.")