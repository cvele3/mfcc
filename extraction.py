# extraction_by_letter.py

import os
import numpy as np
import librosa

# ==============================================================================
# 1) Definicija ciljanih fonema i mapiranje u brojeve
# ==============================================================================
# Biramo 8 karakterističnih fonema za klasifikaciju.
# Možete dodati ili ukloniti foneme po želji.
TARGET_PHONEMES = ['a', 'e', 'i', 'o', 'u', 'p', 's', 't']

# Kreiramo rječnik koji svakom fonemu dodjeljuje jedinstveni broj (0, 1, 2...)
# Ovo je nužno jer se modeli strojnog učenja treniraju na brojevima, ne na tekstu.
phoneme_to_label = {phoneme: i for i, phoneme in enumerate(TARGET_PHONEMES)}
# Obrnuti rječnik, koristan za kasniju analizu rezultata
label_to_phoneme = {i: phoneme for i, phoneme in enumerate(TARGET_PHONEMES)}

# 2) Čitanje .lab datoteke (ostaje isto)
def read_lab(lab_path):
    segments = []
    with open(lab_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3: continue
            start_s = int(parts[0]) * 1e-7
            end_s   = int(parts[1]) * 1e-7
            ph_label = parts[2]
            segments.append((start_s, end_s, ph_label))
    return segments

# 3) Ekstrakcija MFCC po segmentima (modificirano za ciljane foneme)
def extract_features_and_labels_from_segments(wav_path, lab_path):
    y, sr = librosa.load(wav_path, sr=16000)
    segments = read_lab(lab_path)

    n_mfcc = 13
    features = []
    labels = []

    for start_s, end_s, ph_label in segments:
        # Uzimamo u obzir samo foneme koje smo definirali u TARGET_PHONEMES
        if ph_label not in TARGET_PHONEMES:
            continue

        start_sample, end_sample = int(start_s * sr), int(end_s * sr)
        if end_sample <= start_sample or len(y[start_sample:end_sample]) < 32:
            continue
            
        segment = y[start_sample:end_sample]
        
        try:
            mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc, n_fft=512, hop_length=256)
        except Exception as e:
            print(f"Greška pri izračunu MFCC za segment {ph_label}: {e}")
            continue

        if mfcc.shape[1] < 1: continue

        mfcc_mean = np.mean(mfcc, axis=1)
        
        # Dodajemo značajke i odgovarajuću numeričku labelu
        features.append(mfcc_mean)
        labels.append(phoneme_to_label[ph_label])

    return np.array(features), np.array(labels)

# ==============================================================================
# 4) Glavni dio: prolaz kroz sve datoteke
# ==============================================================================
wav_dir = 'wav'
lab_dir = 'lab'
X_all, y_all = [], []

if not os.path.isdir(wav_dir) or not os.path.isdir(lab_dir):
    print("Greška: Direktoriji 'wav' i/ili 'lab' nisu pronađeni.")
else:
    for wav_filename in os.listdir(wav_dir):
        if wav_filename.endswith('.wav'):
            wav_filepath = os.path.join(wav_dir, wav_filename)
            lab_filepath = os.path.join(lab_dir, wav_filename.replace('.wav', '.lab'))

            if not os.path.exists(lab_filepath):
                continue
            
            print(f"Obrada: {wav_filename}")
            mfcc_feats, frame_labels = extract_features_and_labels_from_segments(wav_filepath, lab_filepath)
            
            if mfcc_feats.size > 0:
                X_all.append(mfcc_feats)
                y_all.append(frame_labels)

# 5) Spajanje i spremanje
if X_all:
    X = np.vstack(X_all)
    y = np.hstack(y_all)
    
    # Spremamo i rječnik za mapiranje kako bismo ga mogli koristiti u skripti za treniranje
    output_file = 'features_labels_by_letter.npz'
    np.savez_compressed(output_file, X=X, y=y, label_map=label_to_phoneme)

    print("\n-------------------------------------------")
    print("EKSTRAKCIJA PO SLOVIMA ZAVRŠENA")
    print("Ukupno segmenata (uzoraka):", X.shape[0])
    print("Broj klasa (fonema):", len(TARGET_PHONEMES))
    print(f"Spremljeni podaci u datoteku: {output_file}")
    print("-------------------------------------------")
else:
    print("Nije pronađen niti jedan od ciljanih fonema za obradu.")