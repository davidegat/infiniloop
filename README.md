# INFINI LOOP - Infinite AI Music Generation

**INFINI LOOP** is an advanced AI-powered music generation tool that creates seamless, infinite loops in real-time. Its purpose is to automatically generate the desired audio fragment using AI, find the perfect loop point, play it seamlessly while preparing the next track, and then transition smoothly via crossfade—thus creating your personal, truly infinite, uninterrupted musical playlist.

By combining AI music generation, optimized zero-crossing detection, and intelligent playback management, INFINI LOOP enables endless high-quality audio without breaks or user intervention. Ideal for ambient soundscapes, background music, creative flow, and live environments.

> ⚠️ **Terminal version (`ilterm.py`)** and **lightweight GUI version (`il1.py`)** are more stable and support **real crossfade** between different loops.
> **Advanced GUI version (`il2.py`)** is experimental and **does not perform crossfade**; playback is sequential.

---

## Versions Available

### `ilterm.py` - Terminal Version (Stable)

* Full command-line and interactive interface
* Robust audio validation and recovery
* Crash-resistant process management
* Fixed optimal settings for stability
* **Supports real crossfade between audio loops**
* **Most stable version**

### `il1.py` - Lightweight GUI Version (Stable)

* Minimal graphical interface with essential controls
* Same audio engine as terminal version
* PyAudio playback with crossfade support
* **As stable as terminal version**
* **Supports real crossfade between tracks**
* **No waveform/spectrum visualization**

### `il2.py` - Advanced GUI Version (Experimental)

* Graphical interface with waveform and spectrum visualization
* Configurable generation settings and loop analysis
* Visual loop metric analysis
* PyAudio low-latency playback (if available)
* **Does not support crossfading between tracks**
* **Experimental - less stable**

---

## Features (All Versions)

* AI-based music generation using [MusicGPT](https://github.com/gabotechs/MusicGPT)
* Seamless, beat-aware loops with optimized zero-crossing
* Automatic crossfading (except `il2.py`)
* Multi-metric loop detection:

  * Spectral similarity
  * Waveform continuity
  * Beat alignment
  * Phase continuity
    
* Audio normalization and safe playback via ffplay or PyAudio

> 🧠 This software was developed with the assistance of an AI language model and includes code generated and refined through AI-human collaboration.

> 🛈 The first time a generation is started, the model will be downloaded automatically. The initial generation may take significantly longer than the following ones.

---

## System Requirements

* **OS**: Linux (Ubuntu 20.04+ recommended)
* **Python**: 3.8 or higher
* **RAM**: 4 GB (GUI) / 2 GB (Terminal)
* **Storage**: At least 2 GB free

---

## Dependencies

### Core (All Versions)

```bash
pip install librosa soundfile scipy numpy pydub
```

### GUI Versions (`il1.py` and `il2.py`)

```bash
pip install matplotlib pillow
# Optional for low-latency playback:
sudo apt install portaudio19-dev python3-pyaudio
pip install pyaudio
```

### System Dependencies (All Versions)

```bash
# Ubuntu/Debian:
sudo apt install ffmpeg pulseaudio-utils alsa-utils
```

### External Binary Required

* Download `musicgpt-x86_64-unknown-linux-gnu` from: [https://github.com/gabotechs/MusicGPT](https://github.com/gabotechs/MusicGPT)

```bash
chmod +x musicgpt-x86_64-unknown-linux-gnu
```

---

# HOW TO USE

## TERMINAL VERSION (`ilterm.py`)

### Basic Command

```bash
python ilterm.py --prompt "ambient lofi loop"
```

### Interactive Mode

```bash
python ilterm.py
```

### One-shot Loop Export

```bash
python ilterm.py --generate-only "jazz groove" output.wav
```

### Key Features

* True crossfade blending
* Automatic validation of loop integrity
* Debug mode with detailed logs
* Safe recovery from crash or corruption
* Stable even on low-spec systems

---

## LIGHTWEIGHT GUI VERSION (`il1.py`)

### Launch

```bash
python il1.py
```

### Workflow

1. Enter your prompt (e.g. `lofi calm rap`)
2. Choose model (Small / Medium / Large)

   * ⚠️ **Avoid Small model**: often produces low-quality results
   * ✅ **Recommended**: Medium model (balanced quality and speed)
   * ⚠️ **Not recommended**: High model (slow and sometimes inconsistent)
3. Set generation duration (5–30s recommended)
4. Choose audio driver (pulse / alsa / oss)
5. Click **AVVIA** to generate infinite loop
6. Crossfade is active between tracks
7. Click **SALVA** to export current loop

### Notes

* Same backend engine as the terminal version
* GUI is clean, minimalist, fast
* Recommended for users who prefer a visual interface with stability

---

## ADVANCED GUI VERSION (`il2.py`)

### Launch

```bash
python il2.py
```

### Workflow

1. Enter your prompt (e.g. `lofi calm rap`)
2. Choose algorithm: Advanced or Classic
3. Choose model

   * ⚠️ **Avoid Small model**: often produces low-quality results
   * ✅ **Recommended**: Medium model
   * ⚠️ **Not recommended**: High model
4. Click **AVVIA** to generate infinite loop
5. Adjust crossfade slider (not yet functional)
6. Click **SALVA** to export current loop

### Visualization

* Real-time waveform and spectral plot
* Highlighted loop zones with loop length analysis
* Visual feedback useful for creators and testers

> ❗ Playback is sequential, not overlapped.

### Limitations

* Experimental GUI – may crash or freeze on some systems
* Does not support crossfade blending
* Heavier resource usage

---

## License

Released under [GPL v3](https://www.gnu.org/licenses/gpl-3.0.html). Only for non-commercial use.

---

## Bug Reporting / Support

For technical issues or bug reports:

* Open a GitHub issue
* Specify version used (`ilterm.py`, `il1.py`, `il2.py`)
* Include your OS version, error logs, and reproduction steps

---

# INFINI LOOP - Generazione Musicale Infinita con AI

**INFINI LOOP** è uno strumento avanzato per la generazione musicale continua con intelligenza artificiale. Il suo scopo è generare automaticamente un frammento musicale desiderato, trovare il punto di loop perfetto, suonarlo senza interruzioni mentre viene preparata la prossima traccia, e passare senza soluzione di continuità a quella successiva tramite crossfade. Il risultato è una playlist musicale infinita, continua e fluida, ideale per ambienti rilassanti, lavoro creativo o uso dal vivo.

> ⚠️ La **versione da terminale (`ilterm.py`)** e la **versione GUI leggera (`il1.py`)** sono stabili e supportano il **crossfade reale** tra i loop.
> La **versione GUI avanzata (`il2.py`)** è sperimentale e **non supporta il crossfade**: la riproduzione è sequenziale.

---

## Versioni Disponibili

### `ilterm.py` - Versione Terminale (Stabile)

* Interfaccia completamente testuale o interattiva
* Validazione audio e recupero automatico
* Gestione robusta in caso di crash
* Impostazioni ottimizzate per stabilità
* **Supporta il crossfade reale**
* **Versione più stabile in assoluto**

### `il1.py` - Versione GUI Leggera (Stabile)

* Interfaccia minimale con controlli essenziali
* Stesso motore audio della versione da terminale
* Riproduzione tramite PyAudio con crossfade attivo
* **Stabilità identica alla versione terminale**
* **Supporto al crossfade reale**
* **Nessuna visualizzazione grafica del segnale audio**

### `il2.py` - Versione GUI Avanzata (Sperimentale)

* Interfaccia grafica con spettrogramma e waveform
* Impostazioni configurabili e analisi del loop
* Analisi visiva multi-metrica
* Riproduzione tramite PyAudio (se disponibile)
* **Non supporta il crossfade tra tracce**
* **Versione sperimentale e meno stabile**

---

## Funzionalità Comuni

* Generazione musicale AI tramite [MusicGPT](https://github.com/gabotechs/MusicGPT)
* Loop perfetti, consapevoli del beat e allineati ai punti di zero-crossing
* Rilevamento del loop tramite:

  * Similarità spettrale
  * Continuità della forma d'onda
  * Allineamento dei beat
  * Continuità di fase
* Normalizzazione audio e riproduzione sicura con ffplay o PyAudio

> 🧠 Il software è stato sviluppato con l'assistenza di un modello linguistico AI, con porzioni di codice generate e poi ottimizzate manualmente.

> 🛈 Alla prima generazione, il modello AI verrà scaricato automaticamente. Questa operazione può richiedere più tempo rispetto alle generazioni successive.

---

## Requisiti di Sistema

* **OS**: Linux (consigliato Ubuntu 20.04+)
* **Python**: 3.8 o superiore
* **RAM**: 4 GB (GUI) / 2 GB (Terminale)
* **Spazio su disco**: almeno 2 GB liberi

---

## Dipendenze

### Base (tutte le versioni)

```bash
pip install librosa soundfile scipy numpy pydub
```

### GUI (`il1.py` e `il2.py`)

```bash
pip install matplotlib pillow
sudo apt install portaudio19-dev python3-pyaudio
pip install pyaudio
```

### Sistema

```bash
sudo apt install ffmpeg pulseaudio-utils alsa-utils
```

### Binario Esterno

Scarica `musicgpt-x86_64-unknown-linux-gnu` da: [https://github.com/gabotechs/MusicGPT](https://github.com/gabotechs/MusicGPT)

```bash
chmod +x musicgpt-x86_64-unknown-linux-gnu
```

---

# COME SI USA

## VERSIONE TERMINALE (`ilterm.py`)

### Esempio semplice

```bash
python ilterm.py --prompt "ambient lofi loop"
```

### Modalità interattiva

```bash
python ilterm.py
```

### Solo generazione (esportazione diretta)

```bash
python ilterm.py --generate-only "jazz groove" output.wav
```

### Caratteristiche principali

* Crossfade reale tra le tracce
* Validazione automatica del loop
* Modalità debug con log dettagliati
* Resiliente ai crash
* Ottima stabilità anche su hardware limitato

---

## VERSIONE GUI LEGGERA (`il1.py`)

### Avvio

```bash
python il1.py
```

### Procedura

1. Inserisci il prompt (es. `lofi calm rap`)
2. Scegli il modello:

   * ⚠️ **Sconsigliato**: Small (risultati spesso scadenti)
   * ✅ **Consigliato**: Medium (ottimo bilanciamento qualità/velocità)
   * ⚠️ **Non raccomandato**: High (lento e poco affidabile)
3. Imposta la durata (consigliato 5–30s)
4. Seleziona il driver audio (pulse / alsa / oss)
5. Clicca **AVVIA** per iniziare
6. Crossfade attivo tra le tracce
7. Clicca **SALVA** per esportare

---

## VERSIONE GUI AVANZATA (`il2.py`)

### Avvio

```bash
python il2.py
```

### Procedura

1. Inserisci il prompt
2. Seleziona algoritmo: Classico o Avanzato
3. Scegli il modello:

   * ⚠️ **Sconsigliato**: Small
   * ✅ **Consigliato**: Medium
   * ⚠️ **Non raccomandato**: High
4. Premi **AVVIA**
5. Regola lo slider del crossfade (attualmente non funziona)
6. Premi **SALVA** per esportare il loop

### Visualizzazione

* Forma d'onda e spettrogramma in tempo reale
* Evidenziazione delle zone di loop
* Metriche visive utili per test e creazione

> ❗ La riproduzione è sequenziale, non sovrapposta.

### Limitazioni

* GUI sperimentale: instabile su alcuni sistemi
* Non supporta il crossfade
* Più pesante in termini di risorse

---

## Licenza

Distribuito con licenza [GPL v3](https://www.gnu.org/licenses/gpl-3.0.html). Solo per uso non commerciale.

---

## Supporto

Per segnalazioni o problemi tecnici:

* Apri una issue su GitHub
* Specifica la versione usata (`ilterm.py`, `il1.py`, `il2.py`)
* Indica sistema operativo, log d’errore ed eventuali passaggi per riprodurre il bug
