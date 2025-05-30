import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct # Dodan idct za inverziju
from scipy.io import wavfile as scipy_wav
# import sounddevice as sd # Više nije potrebno za automatsku reprodukciju

# Attempt to import python_speech_features for comparison
try:
    from python_speech_features import mfcc as psf_mfcc
    from python_speech_features import logfbank # For filterbank energies if needed
    PYTHON_SPEECH_FEATURES_AVAILABLE = True
except ImportError:
    PYTHON_SPEECH_FEATURES_AVAILABLE = False
    print("Upozorenje: biblioteka python_speech_features nije pronađena. Usporedba s njom će biti preskočena.")

# --- Configuration Constants ---
AUDIO_FILE = "sm04010103201.wav"  # Zamijenite s vašom govornom WAV datotekom
# MFCC parameters (typical values, can be tuned)
FRAME_LENGTH_MS = 25
FRAME_STEP_MS = 10
NUM_CEP = 13               # Broj kepstralnih koeficijenata za zadržavanje (obično 12-13)
NUM_MEL_FILTERS = 26       # Broj Mel filtera (obično 20-40)
PREEMPH_ALPHA = 0.97       # Koeficijent prednaglašavanja
N_FFT_MUST_BE_POWER_OF_2 = True # Ako je True, n_fft će biti najmanja potencija broja 2 >= frame_length_samples
# For quantization and compression
QUANTIZATION_BITS = 8      # Broj bitova za kvantizaciju svake MFCC vrijednosti

# --- Helper Functions ---

def ensure_sample_wav(filepath="sample.wav", sr=16000, duration=2, freq=440):
    """
    Provjerava postoji li primjer WAV datoteke. Ako ne, generira jednostavnu.
    """
    if not os.path.exists(filepath):
        print(f"'{filepath}' nije pronađena. Generira se dummy WAV datoteka za demonstraciju.")
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        tone = 0.5 * np.sin(2 * np.pi * freq * t)
        tone_int16 = np.int16(tone * 32767)
        scipy_wav.write(filepath, sr, tone_int16)
        print(f"Dummy '{filepath}' (trajanje={duration}s, freq={freq}Hz, sr={sr}Hz, 16-bit PCM) stvorena.")
        print("Molimo zamijenite 'sample.wav' stvarnom govornom datotekom za smislene rezultate.")
    else:
        print(f"Koristi se postojeća audio datoteka: '{filepath}'.")

def load_audio(filepath):
    """
    Učitava audio datoteku koristeći librosa.
    Vraća signal, frekvenciju uzorkovanja i originalnu veličinu datoteke u bajtovima.
    """
    try:
        signal, sr = librosa.load(filepath, sr=None)
    except Exception as e:
        print(f"Greška pri učitavanju audio datoteke {filepath}: {e}")
        print("Molimo osigurajte valjanu WAV datoteku.")
        return None, None, 0
    try:
        original_filesize_bytes = os.path.getsize(filepath)
    except OSError:
        original_filesize_bytes = signal.nbytes
        print(f"Nije bilo moguće dobiti veličinu datoteke za {filepath}. Procjenjuje se iz duljine podataka signala.")
    return signal, sr, original_filesize_bytes

def preemphasis(signal, alpha=0.97):
    """
    Primjenjuje filter prednaglašavanja na signal.
    """
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])

def compute_mfccs_manual(signal, sr, frame_length_ms=25, frame_step_ms=10,
                         num_cep=13, num_mel_filters=26, preemph_alpha=0.97,
                         n_fft_pow2=True):
    """
    Izračunava Mel-frekvencijske kepstralne koeficijente (MFCC) prateći standardne korake.
    """
    print("\n--- Izračun MFCC-a (Ručna implementacija korak-po-korak) ---")
    signal_preemphasized = preemphasis(signal, preemph_alpha)
    print(f"1. Primjena prednaglašavanja s alpha={preemph_alpha}...")

    frame_length_samples = int(round(frame_length_ms / 1000 * sr))
    frame_step_samples = int(round(frame_step_ms / 1000 * sr))
    print(f"   Duljina okvira: {frame_length_samples} uzoraka ({frame_length_ms}ms)")
    print(f"   Korak okvira: {frame_step_samples} uzoraka ({frame_step_ms}ms)")

    if n_fft_pow2:
        n_fft = 1
        while n_fft < frame_length_samples:
            n_fft *= 2
    else:
        n_fft = frame_length_samples
    print(f"   N_FFT: {n_fft}")

    print("2. Uokviravanje signala...")
    slen = len(signal_preemphasized)
    if slen <= frame_length_samples:
        num_frames = 1
    else:
        num_frames = 1 + int(np.ceil((1.0 * slen - frame_length_samples) / frame_step_samples))
    pad_len = int((num_frames - 1) * frame_step_samples + frame_length_samples)
    zeros_to_pad = np.zeros((pad_len - slen,))
    pad_signal = np.concatenate((signal_preemphasized, zeros_to_pad))
    indices = np.tile(np.arange(0, frame_length_samples), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step_samples, frame_step_samples), (frame_length_samples, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    print(f"   Broj okvira: {num_frames}")

    print("3. Primjena Hamming prozora na svaki okvir...")
    frames_windowed = frames * np.hamming(frame_length_samples)

    print("4. Izračun energetskog spektra (FFT -> |.|^2)...")
    complex_spectrum = np.fft.rfft(frames_windowed, n=n_fft)
    power_spectrum = (1.0 / n_fft) * (np.abs(complex_spectrum) ** 2)

    print(f"5. Primjena Mel filterbanke ({num_mel_filters} filtera)...")
    mel_filters = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=num_mel_filters, fmin=0, fmax=sr / 2.0)
    mel_energies = np.dot(power_spectrum, mel_filters.T)

    print("6. Logaritmiranje Mel energija filterbanke...")
    mel_energies = np.where(mel_energies == 0, np.finfo(float).eps, mel_energies)
    log_mel_energies = np.log(mel_energies)

    print("7. Primjena diskretne kosinusne transformacije (DCT)...")
    mfccs_full = dct(log_mel_energies, type=2, axis=1, norm='ortho')
    mfccs = mfccs_full[:, :num_cep]
    print(f"   Zadržavanje prvih {num_cep} MFCC koeficijenata.")
    print(f"   Konačni oblik MFCC-a: {mfccs.shape} (okviri, koeficijenti)")
    print("--- Izračun MFCC-a (ručni) završen. ---")
    return mfccs

def quantize_mfccs(mfccs, num_bits):
    """
    Izvodi jednostavnu uniformnu kvantizaciju na MFCC-ima.
    """
    mfccs_min = np.min(mfccs)
    mfccs_max = np.max(mfccs)
    num_levels = 2**num_bits
    q_step = (mfccs_max - mfccs_min) / (num_levels - 1) if (num_levels > 1 and mfccs_max > mfccs_min) else 1.0
    if q_step == 0: q_step = 1.0
    quantized_indices = np.round((mfccs - mfccs_min) / q_step).astype(int)
    quantized_indices = np.clip(quantized_indices, 0, num_levels - 1)
    return quantized_indices, mfccs_min, q_step

def calculate_compression_info(original_filesize_bytes, mfccs, quantization_bits):
    """
    Izračunava veličinu kvantiziranih MFCC-a i omjer kompresije.
    """
    num_frames, num_coeffs_per_frame = mfccs.shape
    overhead_bits_for_quant_params = 2 * 32
    compressed_data_bits = (num_frames * num_coeffs_per_frame * quantization_bits)
    total_compressed_bits = compressed_data_bits + overhead_bits_for_quant_params
    compressed_size_bytes = np.ceil(total_compressed_bits / 8.0)
    if original_filesize_bytes == 0: compression_ratio = 0
    elif compressed_size_bytes == 0: compression_ratio = float('inf')
    else: compression_ratio = original_filesize_bytes / compressed_size_bytes
    return compression_ratio, compressed_size_bytes, original_filesize_bytes

def plot_results(signal, sr, mfccs, title_suffix="Ručni izračun",
                 frame_length_ms=25, frame_step_ms=10):
    """
    Vizualizira: originalni valni oblik, spektrogram, MFCC heatmap.
    """
    plt.figure(figsize=(12, 10))
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(signal, sr=sr)
    plt.title(f"Originalni valni oblik ({os.path.basename(AUDIO_FILE)})")

    frame_length_samples = int(round(frame_length_ms / 1000 * sr))
    frame_step_samples = int(round(frame_step_ms / 1000 * sr))
    n_fft_spec = 1
    while n_fft_spec < frame_length_samples: n_fft_spec *= 2
    S = librosa.stft(signal, n_fft=n_fft_spec, hop_length=frame_step_samples, win_length=frame_length_samples)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    plt.subplot(3, 1, 2)
    librosa.display.specshow(S_db, sr=sr, hop_length=frame_step_samples, x_axis='time', y_axis='log')
    plt.title("Spektrogram")
    plt.colorbar(format='%+2.0f dB')

    plt.subplot(3, 1, 3)
    librosa.display.specshow(mfccs.T, sr=sr, hop_length=frame_step_samples, x_axis='time')
    plt.title(f"MFCC Heatmap ({title_suffix}, {mfccs.shape[1]} Koef.)")
    plt.ylabel("MFCC Koeficijenti")
    plt.colorbar()
    plt.tight_layout()
    plt.show(block=False) # Promijenjeno da ne blokira za ostale grafove

def plot_compression_vs_num_cep(signal, sr, original_filesize_bytes,
                                max_num_cep=20, quantization_bits=8,
                                frame_length_ms=25, frame_step_ms=10,
                                num_mel_filters=26, preemph_alpha=0.97):
    """
    Izračunava i crta omjer kompresije za različit broj MFCC koeficijenata.
    """
    num_ceps_range = range(1, max_num_cep + 1)
    compression_ratios = []
    print(f"\n--- Izračun omjera kompresije za 1 do {max_num_cep} MFCC koeficijenata ---")
    for n_cep_iter in num_ceps_range:
        mfccs_iter = compute_mfccs_manual(signal, sr, num_cep=n_cep_iter,
                                     frame_length_ms=frame_length_ms, frame_step_ms=frame_step_ms,
                                     num_mel_filters=num_mel_filters, preemph_alpha=preemph_alpha)
        ratio, _, _ = calculate_compression_info(original_filesize_bytes, mfccs_iter, quantization_bits)
        compression_ratios.append(ratio)
        print(f"  Broj CEP: {n_cep_iter:2d}, Omjer kompresije: {ratio:7.2f}")

    plt.figure(figsize=(10, 6))
    plt.plot(num_ceps_range, compression_ratios, marker='o', linestyle='-')
    plt.title(f"Omjer kompresije vs. Broj MFCC koeficijenata\n(Kvantizirano na {quantization_bits} bita po koeficijentu)")
    plt.xlabel("Broj zadržanih MFCC koeficijenata (num_cep)")
    plt.ylabel("Omjer kompresije (Originalna veličina / Komprimirana MFCC veličina)")
    plt.grid(True)
    plt.xticks(np.arange(min(num_ceps_range), max(num_ceps_range)+1, step=max(1,max_num_cep//10)))
    plt.tight_layout()
    plt.show(block=False) # Promijenjeno da ne blokira

# --- Nova funkcija za rekonstrukciju ---
def reconstruct_audio_from_mfccs(mfccs, sr, n_fft, hop_length_samples, win_length_samples,
                                 num_mel_filters, preemph_alpha, num_cep_original,
                                 griffin_lim_iters=32):
    """
    Rekonstruira audio signal iz MFCC koeficijenata.
    """
    print("\n--- Rekonstrukcija audio signala iz MFCC-a ---")

    if mfccs.shape[1] < num_mel_filters:
        mfccs_padded = np.zeros((mfccs.shape[0], num_mel_filters))
        mfccs_padded[:, :mfccs.shape[1]] = mfccs
    else:
        mfccs_padded = mfccs

    print("1. Primjena inverznog DCT-a...")
    log_mel_energies = idct(mfccs_padded, type=2, axis=1, norm='ortho')

    print("2. Primjena inverznog logaritma (eksponenciranje)...")
    mel_energies = np.exp(log_mel_energies)

    print("3. Primjena inverzne Mel filterbanke (procjena STFT magnitude)...")
    estimated_power_spectrogram = librosa.feature.inverse.mel_to_stft(
        mel_energies.T, sr=sr, n_fft=n_fft, power=2.0
    )

    print(f"4. Primjena Griffin-Lim algoritma ({griffin_lim_iters} iteracija)...")
    reconstructed_signal_emphasized = librosa.griffinlim(
        estimated_power_spectrogram, n_iter=griffin_lim_iters,
        hop_length=hop_length_samples, win_length=win_length_samples,
        window='hamming'
    )

    print("5. Primjena inverznog prednaglašavanja (de-emphasis)...")
    reconstructed_signal = librosa.effects.deemphasis(reconstructed_signal_emphasized, coef=preemph_alpha)
    
    print("--- Rekonstrukcija završena. ---")
    return reconstructed_signal

# --- Glavni Izvršni Dio ---
def main():
    print("Izračun i prikaz Mel-kepstralnih koeficijenata (MFCC)")
    print("=======================================================")

    ensure_sample_wav(AUDIO_FILE)
    signal, sr, original_filesize_bytes = load_audio(AUDIO_FILE)
    if signal is None: return

    print(f"\nUčitan audio: '{AUDIO_FILE}'")
    print(f"Frekvencija uzorkovanja: {sr} Hz")
    print(f"Duljina signala: {len(signal)} uzoraka ({len(signal)/sr:.2f} sekundi)")
    print(f"Originalna veličina datoteke: {original_filesize_bytes} bajtova")

    frame_length_samples = int(round(FRAME_LENGTH_MS / 1000 * sr))
    frame_step_samples = int(round(FRAME_STEP_MS / 1000 * sr))
    if N_FFT_MUST_BE_POWER_OF_2:
        n_fft = 1
        while n_fft < frame_length_samples: n_fft *= 2
    else:
        n_fft = frame_length_samples

    mfccs_manual = compute_mfccs_manual(signal, sr,
                                        frame_length_ms=FRAME_LENGTH_MS, frame_step_ms=FRAME_STEP_MS,
                                        num_cep=NUM_CEP, num_mel_filters=NUM_MEL_FILTERS,
                                        preemph_alpha=PREEMPH_ALPHA, n_fft_pow2=N_FFT_MUST_BE_POWER_OF_2)

    print(f"\n--- Kvantizacija i procjena kompresije ---")
    print(f"Kvantiziranje MFCC-a na {QUANTIZATION_BITS} bita po koeficijentu...")
    quantized_mfcc_indices, q_min, q_step = quantize_mfccs(mfccs_manual, QUANTIZATION_BITS)
    compression_ratio, compressed_size_bytes, _ = calculate_compression_info(
        original_filesize_bytes, mfccs_manual, QUANTIZATION_BITS
    )
    print(f"Originalna veličina signala: {original_filesize_bytes} bajtova")
    print(f"Procijenjena komprimirana veličina MFCC-a: {compressed_size_bytes:.0f} bajtova "
          f"(koristeći {NUM_CEP} koef., {QUANTIZATION_BITS}-bitnu kvantizaciju)")
    print(f"Omjer kompresije: {compression_ratio:.2f}")

    plot_results(signal, sr, mfccs_manual,
                 title_suffix=f"Ručni ({NUM_CEP} Koef., {QUANTIZATION_BITS}-bit Kvant.)",
                 frame_length_ms=FRAME_LENGTH_MS, frame_step_ms=FRAME_STEP_MS)

    # --- MODIFICIRANO: Samo spremanje rekonstruiranog zvuka ---
    reconstructed_audio = reconstruct_audio_from_mfccs(
        mfccs_manual, sr=sr, n_fft=n_fft,
        hop_length_samples=frame_step_samples, win_length_samples=frame_length_samples,
        num_mel_filters=NUM_MEL_FILTERS, preemph_alpha=PREEMPH_ALPHA,
        num_cep_original=NUM_CEP, griffin_lim_iters=60 # Povećan broj iteracija za potencijalno bolju kvalitetu
    )
    # Normalizacija rekonstruiranog signala da bude u [-1, 1] rasponu za spremanje
    if np.max(np.abs(reconstructed_audio)) > 0:
         reconstructed_audio = reconstructed_audio / np.max(np.abs(reconstructed_audio))

    reconstructed_filename = "reconstructed_audio_from_mfcc.wav" # Malo deskriptivnije ime
    try:
        # Spremanje kao float32 WAV. Scipy će automatski skalirati ako je int.
        # Za float, očekuje se da je u rasponu [-1, 1]
        scipy_wav.write(reconstructed_filename, sr, reconstructed_audio.astype(np.float32))
        print(f"\nRekonstruirani audio spremljen kao: {reconstructed_filename}")
        print("Možete ga sada ručno reproducirati iz te datoteke.")

    except Exception as e:
        print(f"Greška prilikom spremanja rekonstruiranog zvuka: {e}")

    # Opcionalno: Usporedba s python_speech_features bibliotekom (ako je dostupna)
    if PYTHON_SPEECH_FEATURES_AVAILABLE:
        print("\n--- Izračun MFCC-a koristeći biblioteku python_speech_features za usporedbu ---")
        psf_frame_len_samples = int(round(FRAME_LENGTH_MS / 1000 * sr))
        psf_nfft = 1
        while psf_nfft < psf_frame_len_samples: psf_nfft *= 2
        mfccs_psf = psf_mfcc(signal, samplerate=sr,
                             winlen=FRAME_LENGTH_MS / 1000, winstep=FRAME_STEP_MS / 1000,
                             numcep=NUM_CEP, nfilt=NUM_MEL_FILTERS, nfft=psf_nfft,
                             lowfreq=0, highfreq=sr / 2, preemph=PREEMPH_ALPHA,
                             ceplifter=0, appendEnergy=False, winfunc=np.hamming)
        print(f"MFCC-i iz python_speech_features oblika: {mfccs_psf.shape}")
        plot_results(signal, sr, mfccs_psf,
                     title_suffix=f"python_speech_features ({NUM_CEP} Koef.)",
                     frame_length_ms=FRAME_LENGTH_MS, frame_step_ms=FRAME_STEP_MS)
        if mfccs_manual.shape == mfccs_psf.shape:
            diff = np.mean(np.abs(mfccs_manual - mfccs_psf))
            print(f"\nSrednja apsolutna razlika između ručnih MFCC-a i MFCC-a iz python_speech_features: {diff:.4f}")
        else:
            print("\nOblici ručnih MFCC-a i MFCC-a iz python_speech_features se ne podudaraju, preskače se numerička usporedba.")

    plot_compression_vs_num_cep(signal, sr, original_filesize_bytes,
                                max_num_cep=20, quantization_bits=QUANTIZATION_BITS,
                                frame_length_ms=FRAME_LENGTH_MS, frame_step_ms=FRAME_STEP_MS,
                                num_mel_filters=NUM_MEL_FILTERS, preemph_alpha=PREEMPH_ALPHA)
    
    # Dodano da se svi grafovi prikažu na kraju ako su neblokirajući
    plt.show() 
    print("\nProces završen.")

if __name__ == "__main__":
    main()