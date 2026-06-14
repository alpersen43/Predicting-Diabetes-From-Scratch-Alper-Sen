
import random

# 1. Loading and Cleaning the data
def load_and_clean_data(file_path):
    dataset = []
    with open(file_path, 'r') as file:
        header = file.readline() # Başlığı atla
        for line in file:
            if not line.strip():
                continue
            parts = line.strip().split(',')
            row = [float(x) for x in parts[:-1]]
            row.append(int(parts[-1]))

# There are some patients whose Glucose and BMI values are 0 which is not possible.
# So we delete those datas.
            if row[1] != 0.0 and row[5] != 0.0:
                dataset.append(row)
    return dataset

# 2. Normalization ()
def normalize_dataset(dataset):
    col_count = len(dataset[0]) - 1
    for i in range(col_count):
        col_values = [row[i] for row in dataset]
        col_min = min(col_values)
        col_max = max(col_values)

        for row in dataset:
            if col_max - col_min == 0:
                row[i] = 0.0
            else:
                row[i] = (row[i] - col_min) / (col_max - col_min)

# 3. Spliting and Mixing the Data
def split_dataset(dataset, train_ratio=0.8):
    random.seed(206)
    random.shuffle(dataset) # We are randomly mixing the data.

    train_size = int(len(dataset) * train_ratio)
    train_set = dataset[:train_size]
    test_set = dataset[train_size:]
    return train_set, test_set

# 4. Euclidean Distance
def calculate_euclidean_distance(patient_a, patient_b):
    distance_squared = 0
    for i in range(len(patient_a) - 1):
        distance_squared += (patient_a[i] - patient_b[i]) ** 2
    return distance_squared ** 0.5

# 5. Finding the closest neighbours
def get_neighbors(train_set, test_row, k):
    distances = []
    for train_row in train_set:
        dist = calculate_euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))

    distances.sort(key=lambda x: x[1])

    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors

# 6. Prediction
def predict_classification(train_set, test_row, k):
    neighbors = get_neighbors(train_set, test_row, k)
    outcomes = [row[-1] for row in neighbors]
    prediction = max(set(outcomes), key=outcomes.count)
    return prediction

# Main Programme
file_name = "diabetes.csv"
k_degeri = 19 # We chose 19 to have higher accuracy.

try:
    full_data = load_and_clean_data(file_name)
    normalize_dataset(full_data)
    train_data, test_data = split_dataset(full_data)

    print(f"Cleaned total data: {len(full_data)}")
    print(f"Training Set: {len(train_data)}")
    print(f"Test Set: {len(test_data)}\n")

    dogru_tahmin = 0
    gercek_degerler = []
    tahmin_edilen_degerler = []

    for row in test_data:
        tahmin = predict_classification(train_data, row, k_degeri)
        gercek_deger = row[-1]

        gercek_degerler.append(gercek_deger)
        tahmin_edilen_degerler.append(tahmin)

        if tahmin == gercek_deger:
            dogru_tahmin += 1

    accuracy = (dogru_tahmin / len(test_data)) * 100
    print(f"Success rate: %{accuracy:.2f}")

except FileNotFoundError:
    print("Hata: 'diabetes.csv' dosyası bulunamadı!")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, classification_report

y_true = gercek_degerler
y_pred = tahmin_edilen_degerler

# Metrikleri Hesaplama
acc = accuracy_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print("\n--- MODEL DEĞERLENDİRME METRİKLERİ ---")
print(f"Accuracy (Doğruluk) : %{acc*100:.2f}")
print(f"Recall (Duyarlılık - Gerçek hastaları tahmin etme): %{recall*100:.2f}")

# Confusion Matrix Visualization
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Sağlıklı (0)', 'Diyabet (1)'],
            yticklabels=['Sağlıklı (0)', 'Diyabet (1)'],
            annot_kws={"size": 14})
plt.title('KNN Model - Karmaşıklık Matrisi', fontsize=15)
plt.ylabel('Gerçek Değerler', fontsize=12)
plt.xlabel('Tahmin Edilen Değerler', fontsize=12)
plt.show()
