import numpy as np
import librosa
import matplotlib.pyplot as plt
import os

# --- Postavke ---
# AŽURIRANO: Nazivi datoteka koje ste vi snimili
# FILE_A = 'slovo_a.wav'
# FILE_I = 'slovo_i.wav'
FILE_A = 'A.wav'
FILE_I = 'I.wav'
N_MFCC = 13  # Broj MFCC koeficijenata koje ćemo koristiti

def analyze_and_plot_phonemes(file_a, file_i):
    """
    Učitava dva audio zapisa, računa prosječne MFCC vektore,
    vizualizira ih i računa udaljenost među njima.
    """
    # Provjera postoje li datoteke
    if not os.path.exists(file_a) or not os.path.exists(file_i):
        print(f"Greška: Jedna od datoteka ('{file_a}' ili '{file_i}') nije pronađena.")
        print("Molimo provjerite jesu li datoteke u istom folderu kao i skripta.")
        return

    try:
        # --- Obrada prvog fonema ("a") ---
        # Učitavanje audio signala
        signal_a, sr_a = librosa.load(file_a, sr=None)
        # Izračun MFCC-a
        mfccs_a = librosa.feature.mfcc(y=signal_a, sr=sr_a, n_mfcc=N_MFCC)
        # Računanje prosječnog MFCC vektora ("potpisa")
        mfcc_a_mean = np.mean(mfccs_a, axis=1)

        # --- Obrada drugog fonema ("i") ---
        signal_i, sr_i = librosa.load(file_i, sr=None)
        mfccs_i = librosa.feature.mfcc(y=signal_i, sr=sr_i, n_mfcc=N_MFCC)
        mfcc_i_mean = np.mean(mfccs_i, axis=1)

        print(f"Prosječni MFCC 'potpis' za '{file_a}':\n{np.round(mfcc_a_mean, 2)}")
        print("-" * 30)
        print(f"Prosječni MFCC 'potpis' za '{file_i}':\n{np.round(mfcc_i_mean, 2)}")
        print("-" * 30)

        # --- Izračun Euklidske udaljenosti ---
        distance = np.linalg.norm(mfcc_a_mean - mfcc_i_mean)
        print(f"Euklidska udaljenost između MFCC potpisa: {distance:.2f}")
        print("Što je veća ova udaljenost, to ih je lakše strojno razlikovati.")
        print("-" * 30)

        # --- Vizualizacija ---
        fig, ax = plt.subplots(figsize=(12, 7))
        
        x = np.arange(N_MFCC)
        width = 0.35

        rects1 = ax.bar(x - width/2, mfcc_a_mean, width, label=f'Slovo "a" ({os.path.basename(file_a)})')
        rects2 = ax.bar(x + width/2, mfcc_i_mean, width, label=f'Slovo "i" ({os.path.basename(file_i)})')

        ax.set_title('Usporedba prosječnih MFCC "potpisa" za slova "a" i "i"', fontsize=16)
        ax.set_ylabel('Vrijednost MFCC koeficijenta', fontsize=12)
        ax.set_xlabel('MFCC koeficijent', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(np.arange(1, N_MFCC + 1))
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        fig.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Dogodila se greška tijekom obrade: {e}")

# --- Glavni dio programa ---
if __name__ == "__main__":
    analyze_and_plot_phonemes(FILE_A, FILE_I)