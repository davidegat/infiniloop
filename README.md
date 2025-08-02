# INFINI LOOP - Infinite AI Music Generation

**INFINI LOOP** is an advanced AI-powered music generation tool that creates seamless, infinite loops in real-time. Using sophisticated loop detection algorithms and AI music generation, it provides continuous, high-quality musical experiences perfect for ambient music, background tracks, and creative compositions.

⚠️ **Terminal version (`ilterm.py`)** and **lightweight GUI version (`il1.py`)** are more stable and support **real crossfade** between tracks.
**Advanced GUI version (`il2.py`)** is experimental and **does not perform crossfade**; playback is sequential.
 
Experimental GUI version:

<img width="1234" height="679" alt="immagine" src="https://github.com/user-attachments/assets/19390959-ba49-476e-8b26-df606c6dad36" />

Lightweight GUI version:

<img width="645" height="220" alt="immagine" src="https://github.com/user-attachments/assets/07bb2f14-af01-415e-a7fe-020309d928de" />

Terminal version:

<img width="791" height="460" alt="immagine" src="https://github.com/user-attachments/assets/02e13d3c-14bd-4ea5-b0b2-7a65f56ddc28" />

---

## Versions Available

### `ilterm.py` - Terminal Version (Stable)

* Fully command-line interface
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
* Multi-metric loop detection:

  * Spectral similarity
  * Waveform continuity
  * Beat alignment
  * Phase continuity
* Audio normalization and safe playback via ffplay or PyAudio

> 🧠 This software was developed with the assistance of an AI language model and includes code generated and refined through AI-human collaboration.

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
no prompt, means entering interactive mode.

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
3. Click **AVVIA** to generate infinite loop
4. Adjust crossfade slider (not yet functional)
5. Click **SALVA** to export current loop

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

## AI Assistance Disclosure

Parts of this software (loop optimization, waveform analysis, audio detection logic) were developed with the support of an AI language model and manually revised for consistency and performance. This hybrid development improves rapid prototyping and audio logic safety.

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

**INFINI LOOP** è uno strumento avanzato per la generazione di musica infinita con l'ausilio dell'intelligenza artificiale. Offre tre versioni: una da terminale (stabile), una GUI leggera (`il1.py`) stabile con supporto al crossfade, e una GUI avanzata (`il2.py`) sperimentale con grafici e analisi visiva ma senza mix tra le tracce.

> ⚠️ La **versione terminale (`ilterm.py`)** e la **GUI leggera (`il1.py`)** sono stabili e supportano il **crossfade reale**.
> La **GUI avanzata (`il2.py`)** riproduce i loop in sequenza e non supporta la sovrapposizione.

---

## Versioni Disponibili

### `ilterm.py` - Terminale (Stabile)

* Interfaccia testuale completa
* Validazione automatica e gestione dei crash
* Motore audio stabile con crossfade tra loop
* Uso consigliato per utenti avanzati e automazioni

### `il1.py` - GUI Leggera (Stabile)

* Interfaccia semplificata
* Playback con crossfade
* Nessuna visualizzazione della forma d'onda
* Funzionalità identiche alla versione da terminale
* Ottima stabilità

### `il2.py` - GUI Avanzata (Sperimentale)

* Visualizzazione grafica avanzata
* Indicatori visivi del loop
* Selezione dell'algoritmo (Classico / Avanzato)
* **Nessun supporto al crossfade**
* Risorse richieste maggiori

---

## Funzionalità Comuni

* Generazione AI con [MusicGPT](https://github.com/gabotechs/MusicGPT)
* Loop perfetti grazie a:

  * Allineamento dei beat
  * Continuità spettrale e di fase
  * Rilevamento ottimizzato di zero-crossing
* Normalizzazione automatica
* Riproduzione via ffplay o PyAudio

> 🧠 Parti del software sono state sviluppate con l'assistenza di un modello linguistico AI e riviste manualmente per garantirne qualità e affidabilità.

---

## Requisiti di Sistema

* **Linux** (consigliato Ubuntu 20.04+)
* **Python** 3.8 o superiore
* **RAM**: 4 GB (GUI), 2 GB (terminale)
* **Spazio Disco**: ≥ 2 GB

---

## Dipendenze

```bash
pip install librosa soundfile scipy numpy pydub
```

### Solo GUI:

```bash
pip install matplotlib pillow
sudo apt install portaudio19-dev python3-pyaudio
pip install pyaudio
```

### Sistema:

```bash
sudo apt install ffmpeg pulseaudio-utils alsa-utils
```

### MusicGPT:

Scarica da:
[https://github.com/gabotechs/MusicGPT](https://github.com/gabotechs/MusicGPT)

```bash
chmod +x musicgpt-x86_64-unknown-linux-gnu
```

---

# COME SI USA

## Terminale (`ilterm.py`)

```bash
python ilterm.py --prompt "ambient lofi loop"
```

Interattivo:
Se non si specificano prompt o comandi, si entra automaticamente in modalità interattiva.

```bash
python ilterm.py
```

Solo generazione:

```bash
python ilterm.py --generate-only "jazz groove" output.wav
```

## GUI Leggera (`il1.py`)

```bash
python il1.py
```

1. Inserisci prompt (es. `chill lofi rap`)
2. Scegli modello e durata
3. Seleziona driver audio
4. Premi **AVVIA**
5. Crossfade attivo
6. Premi **SALVA** per esportare

## GUI Avanzata (`il2.py`)

```bash
python il2.py
```

1. Inserisci prompt
2. Scegli algoritmo
3. Avvia generazione
4. Regola crossfade (non funziona)
5. Salva loop generato

> ❗ Riproduzione in sequenza, senza sovrapposizione.

---

## Licenza

Distribuito con licenza [GPL v3](https://www.gnu.org/licenses/gpl-3.0.html). Solo per uso non commerciale.

---

## Supporto

* Apri una issue GitHub
* Specifica versione e sistema operativo
* Includi log di errore se possibile
