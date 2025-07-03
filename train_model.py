# train_model_by_letter.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# 1. Učitavanje podataka
try:
    data = np.load('features_labels_by_letter.npz', allow_pickle=True)
    X = data['X']
    y = data['y']
    # Učitavamo i mapu labela koju smo spremili
    label_map = data['label_map'].item()
    class_names = [label_map[i] for i in range(len(label_map))]
    
    print("Podaci su uspješno učitani iz 'features_labels_by_letter.npz'.")
    print(f"Ukupno uzoraka: {X.shape[0]}")
    print(f"Klase za klasifikaciju: {class_names}")
except FileNotFoundError:
    print("Greška: Datoteka 'features_labels_by_letter.npz' nije pronađena.")
    print("Molimo prvo pokrenite skriptu 'extraction_by_letter.py'.")
    exit()

# 2. Podjela podataka (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\nSkup za učenje: {X_train.shape[0]} uzoraka")
print(f"Skup za testiranje: {X_test.shape[0]} uzoraka")

# 3. Kreiranje i treniranje modela
model = GaussianNB()
print("\nTreniranje Naivnog Bayesovog modela...")
model.fit(X_train, y_train)
print("Model je uspješno istreniran.")

# 4. Predikcija na testnom skupu
y_pred = model.predict(X_test)

# 5. Evaluacija performansi
accuracy = accuracy_score(y_test, y_pred)
print(f"\nUkupna točnost modela: {accuracy * 100:.2f}%")

# 6. Kreiranje i vizualizacija matrice zabune za više klasa
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Matrica zabune za klasifikaciju fonema', fontsize=16)
plt.ylabel('Stvarna klasa', fontsize=12)
plt.xlabel('Predviđena klasa', fontsize=12)
plt.show()