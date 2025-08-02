# INFINI LOOP - Infinite AI Music Generation

**INFINI LOOP** is an advanced AI-powered music generation tool that creates seamless, infinite loops in real-time. Using sophisticated loop detection algorithms and AI music generation, it provides continuous, high-quality musical experiences perfect for ambient music, background tracks, and creative compositions.

> ⚠️ **Terminal lightweight version** is more stable and supports **actual crossfade** between tracks.
> **GUI version** is experimental and does not perform crossfade; playback is sequential.

<img width="1234" height="679" alt="immagine" src="https://github.com/user-attachments/assets/19390959-ba49-476e-8b26-df606c6dad36" />

---

## Versions Available

### `il2.py` - GUI Version (Experimental)

* Graphical interface with waveform/spectrum visualization
* Configurable generation settings and loop analysis
* Visual loop metric analysis
* PyAudio low-latency playback (if available)
* **Does not support crossfading between tracks**

### `ilterm.py` - Terminal Version (Stable)

* Fully command-line interface
* Robust audio validation and recovery
* Crash-resistant process management
* Fixed optimal settings for stability
* **Supports real crossfade between audio loops**

---

## Features (Both Versions)

* AI-based music generation using [MusicGPT](https://github.com/gabotechs/MusicGPT)
* Seamless, beat-aware loops with optimized zero-crossing
* Multi-metric loop detection:

  * Spectral similarity
  * Waveform continuity
  * Beat alignment
  * Phase continuity
* Audio normalization and safe playback via ffplay or PyAudio

---

## System Requirements

* **OS**: Linux (Ubuntu 20.04+ recommended)
* **Python**: 3.8 or higher
* **RAM**: 4 GB (GUI) / 2 GB (Terminal)
* **Storage**: At least 2 GB free

---

## Dependencies

### Core (Both Versions)

```bash
pip install librosa soundfile scipy numpy pydub
```

### GUI Only

```bash
pip install matplotlib pillow
# Optional for low-latency playback:
sudo apt install portaudio19-dev python3-pyaudio
pip install pyaudio
```

### System Dependencies (Both Versions)

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

## GUI VERSION (`il2.py`)

### Launch

```bash
python il2.py
```

### Workflow

1. Enter your prompt (e.g. `lofi calm rap`)
2. Choose algorithm: Advanced or Classic
3. Click **AVVIA** to generate infinite loop
4. Adjust crossfade (has no effect yet)
5. Click **SALVA** to export current loop

### Settings

* Model: Small / Medium / Large
* Duration: 5-30s
* Audio Driver: pulse / alsa / oss

> ❗ GUI playback is sequential, not overlapped.

---

## TERMINAL VERSION (`ilterm.py`)

### Basic Command

```bash
python ilterm.py --prompt "ambient lofi loop"
```

### Interactive Mode

```bash
python ilterm.py --interactive
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

---

## Notes on AI-generated Code

Some parts of the code (especially logic for beat/phase analysis and loop optimization) were generated with the help of AI and refined manually. The architecture prioritizes fault tolerance and audio quality.

---

# INFINI LOOP - Generazione Musicale Infinita con AI

**INFINI LOOP** è uno strumento per la generazione musicale continua con AI. Include una versione con interfaccia grafica (sperimentale) e una versione da terminale (stabile). Parti del codice sono state scritte con supporto AI.

> ⚠️ La **versione terminale** è più stabile e supporta il **crossfade reale**.
> La **versione GUI** è sperimentale: funziona ma **non esegue il mix fra tracce**.

---

## Versioni

### `il2.py` - Versione GUI (Sperimentale)

* Interfaccia visuale con grafici in tempo reale
* Analisi del loop visiva
* Selezione modello AI e parametri
* Riproduzione con PyAudio (se disponibile)
* **Nessun supporto al crossfade tra loop**

### `ilterm.py` - Versione Terminale (Stabile)

* Interfaccia testuale completa
* Controllo interattivo o a comandi
* Robustezza contro errori e file corrotti
* Impostazioni fisse per stabilità
* **Supporto completo al crossfade reale**

---

## Requisiti di Sistema

* **OS**: Linux (consigliato Ubuntu 20.04+)
* **Python**: 3.8+
* **RAM**: 4 GB (GUI) / 2 GB (Term)
* **Spazio disco**: Minimo 2 GB

---

## Dipendenze

### Core (entrambe le versioni)

```bash
pip install librosa soundfile scipy numpy pydub
```

### Solo GUI

```bash
pip install matplotlib pillow
# Per audio a bassa latenza:
sudo apt install portaudio19-dev python3-pyaudio
pip install pyaudio
```

### Dipendenze di sistema

```bash
sudo apt install ffmpeg pulseaudio-utils alsa-utils
```

### Binario Esterno Richiesto

Scaricare da [https://github.com/gabotechs/MusicGPT](https://github.com/gabotechs/MusicGPT)

```bash
chmod +x musicgpt-x86_64-unknown-linux-gnu
```

---

# COME SI USA

## VERSIONE GUI (`il2.py`)

### Avvio

```bash
python il2.py
```

### Procedura

1. Inserisci un prompt (es: `lofi calm rap`)
2. Scegli algoritmo: Avanzato o Classico
3. Premi **AVVIA** per iniziare
4. Regola il crossfade (in futuro funzionante)
5. Premi **SALVA** per esportare

> ❗ La riproduzione è sequenziale, non sovrapposta.

---

## VERSIONE TERMINALE (`ilterm.py`)

### Esempio semplice

```bash
python ilterm.py --prompt "ambient lofi loop"
```

### Modalità interattiva

```bash
python ilterm.py --interactive
```

### Solo generazione

```bash
python ilterm.py --generate-only "jazz groove" output.wav
```

### Funzionalità principali

* Crossfade reale tra loop
* Validazione automatica dei file
* Modalità debug con log avanzati
* Recupero da errori e file danneggiati

---

## Licenza

Rilasciato sotto licenza [GPL v3](https://www.gnu.org/licenses/gpl-3.0.html). Solo per uso non commerciale.

---

## Segnalazioni / Supporto

Per segnalazioni tecniche o bug:

* Apri un issue su GitHub
* Specifica versione (GUI o Terminale) e OS
* Includi log e dettagli utili
