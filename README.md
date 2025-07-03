# Prepoznavanje fonema korištenjem MFCC i Naivnog Bayesovog klasifikatora

## Struktura projekta

- `main.py` – Izračun i vizualizacija MFCC koeficijenata, rekonstrukcija signala.
- `extraction.py` – Ekstrakcija MFCC značajki i priprema podataka za treniranje.
- `train_model.py` – Treniranje i evaluacija klasifikatora fonema.
- `lab/` – Label datoteke s fonemima.
- `wav/` – Audio datoteke (WAV format).
- `features_labels_by_letter.npz` – Generirani skup značajki i labela.
- `requirements.txt` – Popis potrebnih Python biblioteka.

## Kako pokrenuti kod

1. **Instalirajte potrebne biblioteke:**
   ```sh
   pip install -r requirements.txt
   ```

2. **Pripremite podatke:**
   - podatci preuzeti iz VEPRAD baze podataka
   - U direktorij `wav/` stavite sve WAV datoteke.
   - U direktorij `lab/` stavite sve pripadajuće `.lab` datoteke.

3. **Ekstrahirajte značajke:**
   Pokrenite:
   ```sh
   python extraction.py
   ```
   Time se generira datoteka `features_labels_by_letter.npz`.

4. **Trenirajte i evaluirajte model:**
   Pokrenite:
   ```sh
   python train_model.py
   ```
   Prikazat će se:
   - Ukupna točnost modela
   - MCC metrika
   - ROC AUC
   - Izvještaj o klasifikaciji po fonemima (preciznost, odziv, F1-mjera)
   - Matrica zabune (confusion matrix)

## Potrebne datoteke

- Svi WAV i LAB parovi u `wav/` i `lab/`
- `features_labels_by_letter.npz` (generira se pokretanjem `extraction.py`)

## Primjer rezultata (uspješnost klasifikacije po fonemima)

Izlaz iz `train_model.py` sadrži tablicu poput ove:

```
Ukupno uzoraka: 54128
Klase za klasifikaciju: ['a', 'e', 'i', 'o', 'u', 'p', 's', 't']

Skup za učenje: 43302 uzoraka
Skup za testiranje: 10826 uzoraka

Treniranje Naivnog Bayesovog modela...
Model je uspješno istreniran.

--- Metrike performansi modela ---
Ukupna točnost (Accuracy): 75.69%
Matthews Correlation Coefficient (MCC): 0.718
ROC AUC (ponderirano): 0.957

--- Izvještaj o klasifikaciji po fonemima ---
              precision    recall  f1-score   support

           a       0.71      0.77      0.74      2031
           e       0.64      0.66      0.65      1625
           i       0.83      0.81      0.82      1599
           o       0.73      0.67      0.70      1441
           u       0.78      0.59      0.68       748
           p       0.67      0.74      0.71       750
           s       0.94      0.97      0.95      1315
           t       0.77      0.77      0.77      1317

    accuracy                           0.76     10826
   macro avg       0.76      0.75      0.75     10826
weighted avg       0.76      0.76      0.76     10826

```
---

Za dodatne informacije pogledajte komentare u kodu