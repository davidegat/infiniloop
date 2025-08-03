# INFINI LOOP - Infinite Local AI Music Generation

INFINI LOOP is an AI-powered music system designed to generate seamless, infinite audio loops in real-time.
It automatically creates new musical fragments using AI, detects the best loop points, and plays them continuously while preparing the next one — resulting in a smooth, never-ending, always new stream of instrumental music.

At startup, one of two pre-included .wav files will play, so you can enjoy music immediately while the first AI generation is being prepared.

Once set up and running, your machine becomes a local AI music station, continuously producing new tracks with smooth transitions and automatic loop detection. Local, private, more personal then any YouTube or Spotify playlist.

Experimental GUI version (**not recomended**):

<img width="1234" height="679" alt="immagine" src="https://github.com/user-attachments/assets/19390959-ba49-476e-8b26-df606c6dad36" />


Lightweight GUI version  (**MOST recomended**):

<img width="696" height="880" alt="immagine" src="https://github.com/user-attachments/assets/e423e138-4214-44e4-9616-4d989ceb9286" />


Terminal version (**recomended**):

<img width="791" height="460" alt="immagine" src="https://github.com/user-attachments/assets/02e13d3c-14bd-4ea5-b0b2-7a65f56ddc28" />


## Table of Contents

- [Features](#features)
- [Available Versions](#available-versions)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Features

- **Local AI Music Generation**: Powered by MusicGPT for high-quality audio synthesis (https://github.com/gabotechs/MusicGPT).
- **Intelligent Loop Detection**: Multi-metric analysis including spectral similarity, waveform continuity, beat alignment, and phase coherence
- **Seamless Playback**: Automatic crossfading between loops (terminal and lightweight GUI versions)
- **Zero-Crossing Optimization**: Ensures smooth transitions
- **Multiple Interfaces**: Terminal, lightweight GUI, and advanced GUI with visualizations
- **Crash Recovery**: Robust error handling and automatic file validation
- **Export Functionality**: Save generated loops for later use

## Available Versions

### 1. Terminal Version (`ilterm.py`) - Stable

- Command-line interface with interactive mode
- Real crossfade support between loops
- Comprehensive audio validation and error recovery
- Debug mode for troubleshooting

### 2. Lightweight GUI Version (`il1.py`) - Most recomended!

- Simple graphical interface with essential controls
- Same audio engine as terminal version
- Real crossfade support
- Clean, minimalist design
- Low resource consumption

### 3. Advanced GUI Version (`il2.py`) - Experimental

- Full graphical interface with audio visualizations
- Real-time waveform and spectrum analysis
- Visual loop metrics display
- Configurable generation parameters
- **Note**: Does not support crossfading between tracks
- Higher resource usage

## System Requirements

### Minimum Requirements

- **Operating System**: Linux (Ubuntu 20.04+ or equivalent)
- **Python**: 3.8 or higher
- **Audio**: Working audio subsystem (PulseAudio, ALSA, or OSS)

### Recommended Requirements

- **Operating System**: Ubuntu 22.04 LTS
- **Python**: 3.10 or higher
- **Processor**: Multi-core CPU for better performance
- **Audio**: PulseAudio with low-latency configuration

At a minimum, your machine must be capable of running a medium‑sized audio model smoothly:
- MusicGPT (medium) currently behaves best on a modern, high‑frequency CPU (GPU support is still marked experimental).
- You should have at least 16 GB of system RAM—something on par with a mid‑range gaming PC.
- Falling short on RAM or CPU performance often leads to memory allocation failures or severely slow inference.

## Installation

### Step 1: Install System Dependencies

```bash
# For Ubuntu/Debian:
sudo apt update
sudo apt install -y ffmpeg pulseaudio-utils alsa-utils python3-pip python3-dev portaudio19-dev python3-tk
```

### Step 2: Install Python Dependencies

```bash
pip install librosa soundfile scipy numpy pydub matplotlib pillow pyaudio
```

### Step 3: Download INFINI LOOP

```bash
# Clone the repository or download the files
git clone https://github.com/yourusername/infiniloop.git
cd infiniloop
```

### Step 4: Download MusicGPT Binary

1. Visit the [MusicGPT releases page](https://github.com/gabotechs/MusicGPT/releases)
2. Download `musicgpt-x86_64-unknown-linux-gnu`
3. Place it in the same directory as the Python scripts (infinloop)
5. Make it executable:

```bash
chmod +x musicgpt-x86_64-unknown-linux-gnu
```

## Usage
Note: MusicGPT will download the selected model (medium by default) at first startup. The very first generation will be significantly slower than subsequent ones. This applies to every version of INFINI LOOP.

### Terminal Version (`ilterm.py`)

#### Quick Start

```bash
# Interactive mode
python ilterm.py

# With custom settings
python ilterm.py --prompt "electronic dance" --duration 20 --driver pulse
```

#### Interactive Commands

When in interactive mode:

- `start <prompt>` - Start infinite loop with given prompt
- `stop` - Stop current playback
- `status` - Show system status
- `save <filename>` - Save current loop
- `set duration` - Change generation duration (5-30 seconds)
- `set driver` - Change audio driver (pulse/alsa/dsp)
- `debug on/off` - Toggle debug messages
- `help` - Show all commands
- `quit` - Exit program

### Lightweight GUI Version (`il1.py`)

```bash
python il1.py
```

1. Enter a musical prompt (e.g., "calm acoustic guitar")
2. Adjust duration if needed (10-15 seconds optimal)
3. Click "START" to start generation
4. Use "SAVE" to export the current loop
5. Click "QUIT" to quit
6. Click "STATUS" to see what's going on

### Advanced GUI Version (`il2.py`)

```bash
python il2.py
```

1. Enter your musical prompt
2. Choose algorithm type:
   - Advanced: Multi-metric analysis (recommended)
   - Classic: Spectral similarity only
3. Select model and duration
4. Click "AVVIA" to begin
5. Monitor visualizations for loop analysis
6. Adjust overlap slider (currently non-functional)
7. Save loops using "SALVA" button

## Technical Details

### Loop Detection Algorithm

INFINI LOOP employs sophisticated multi-metric analysis:

1. **Spectral Similarity**: Compares frequency content at potential loop points
2. **Waveform Continuity**: Ensures smooth amplitude transitions
3. **Beat Alignment**: Synchronizes loops with detected rhythm
4. **Phase Continuity**: Maintains phase coherence for natural sound
5. **Zero-Crossing Optimization**: Fine-tunes loop points

### Audio Processing Pipeline

1. AI generates raw audio using MusicGPT
2. Loop detection algorithm analyzes the audio
3. Optimal loop points are identified and refined
4. Audio is normalized and prepared for playback
5. Crossfade regions are calculated (terminal and lightweight GUI only)
6. Continuous playback begins while next segment generates

### Model Recommendations

- **Small Model**: Fast but often low quality - not recommended
- **Medium Model**: Best balance of quality and speed - recommended
- **Large Model**: Highest quality but slow and resource-intensive

## Troubleshooting

### Common Issues

**Issue**: "File audio generato con errori dalla AI"
- **Solution**: Ensure MusicGPT binary is executable and in the correct location

**Issue**: Audio playback stutters or skips
- **Solution**: Try different audio driver (pulse, alsa, or dsp)

**Issue**: Generation takes too long
- **Solution**: Use medium model and shorter duration (10-15 seconds)

**Issue**: No audio output
- **Solution**: Check system audio settings and ensure audio subsystem is running

### Debug Mode

Enable debug mode for detailed logging:

```bash
# Terminal version
python ilterm.py --prompt "test" --debug

# In interactive mode
debug on
```

### To-Do

- Preset support for music generation
- Visual model integration, music change based on screen activities
- Advanced GUI crossfade support and general enhancements
- Fade out/in while mixing tracks

## License

This project is released under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0). For commercial use, please contact the authors.

## Credits

- MusicGPT by gabotechs for AI music generation: https://github.com/gabotechs/MusicGPT
- Developed with assistance from AI language models
- Audio processing powered by librosa and soundfile

## Contributing

Bug reports and feature requests are welcome. Please include:

- Version used (ilterm.py, il1.py, or il2.py)
- Operating system and Python version
- Complete error messages
- Steps to reproduce the issue

---

# INFINI LOOP - Generazione Musicale Infinita con AI locale

INFINI LOOP è un sistema musicale basato su intelligenza artificiale progettato per generare in tempo reale loop audio infiniti e senza interruzioni.
Crea automaticamente nuovi frammenti musicali, rileva i migliori punti di loop e li riproduce in sequenza continua, mentre prepara il segmento successivo — offrendo un flusso musicale ininterrotto e sempre nuovo.

All’avvio, uno dei due file .wav inclusi verrà riprodotto subito, così potrai ascoltare musica anche mentre l’IA prepara la prima generazione.

Una volta configurato, il tuo computer diventa una stazione musicale AI locale, in grado di produrre nuovi brani con transizioni fluide e rilevamento automatico del loop. Locale, privato, e piu personale di qualsiasi playlist di YouTube o Spotify.

## Indice

- [Caratteristiche](#caratteristiche)
- [Versioni Disponibili](#versioni-disponibili)
- [Requisiti di Sistema](#requisiti-di-sistema)
- [Installazione](#installazione-1)
- [Utilizzo](#utilizzo)
- [Dettagli Tecnici](#dettagli-tecnici)
- [Risoluzione Problemi](#risoluzione-problemi)
- [Licenza](#licenza)

## Caratteristiche

- **Generazione Musicale AI in locale**: Utilizza MusicGPT per sintesi audio di alta qualità (https://github.com/gabotechs/MusicGPT)
- **Rilevamento Loop Intelligente**: Analisi multi-metrica inclusi similarità spettrale, continuità della forma d'onda, allineamento dei beat e coerenza di fase
- **Riproduzione Continua**: Crossfade automatico tra i loop (versioni terminale e GUI leggera)
- **Ottimizzazione Zero-Crossing**: Garantisce transizioni fluide
- **Interfacce Multiple**: Terminale, GUI leggera e GUI avanzata con visualizzazioni
- **Recupero da Crash**: Gestione errori robusta e validazione automatica dei file
- **Funzione Esportazione**: Salva i loop generati per uso futuro

## Versioni Disponibili

### 1. Versione Terminale (`ilterm.py`) - Più Stabile

- Interfaccia a riga di comando con modalità interattiva
- Supporto crossfade reale tra i loop
- Validazione audio completa e recupero errori
- Modalità debug per diagnostica
- Utilizzo minimo di risorse

### 2. Versione GUI Leggera (`il1.py`) - La più raccomandata!

- Interfaccia grafica semplice con controlli essenziali
- Stesso motore audio della versione terminale
- Supporto crossfade reale
- Design pulito e minimalista
- Basso consumo di risorse

### 3. Versione GUI Avanzata (`il2.py`) - Sperimentale

- Interfaccia grafica completa con visualizzazioni audio
- Analisi forma d'onda e spettro in tempo reale
- Display visivo delle metriche del loop
- Parametri di generazione configurabili
- **Nota**: Non supporta il crossfading tra le tracce
- Maggiore utilizzo di risorse

## Requisiti di Sistema

### Requisiti Minimi

- **Sistema Operativo**: Linux (Ubuntu 20.04+ o equivalente)
- **Python**: 3.8 o superiore
- **Audio**: Sottosistema audio funzionante (PulseAudio, ALSA o OSS)

### Requisiti Consigliati

- **Sistema Operativo**: Ubuntu 22.04 LTS
- **Python**: 3.10 o superiore
- **Processore**: CPU multi-core per migliori prestazioni
- **Audio**: PulseAudio con configurazione a bassa latenza

Come minimo, il tuo sistema deve avere le seguenti caratteristiche per far girare con decenza un modello audio di medie dimensioni:
- Il modello MusicGPT “medium” funziona meglio su una CPU moderna e veloce (il supporto GPU è al momento indicato come sperimentale).
- È consigliabile avere almeno 16 GB di RAM di sistema, in linea con un PC da gaming di fascia media.
- Con RAM insufficiente o prestazioni CPU limitate, è molto probabile incorrere in errori di allocazione di memoria o in una inferenza estremamente lenta.
    
## Installazione

### Passo 1: Installa Dipendenze di Sistema

```bash
pip install librosa soundfile scipy numpy pydub matplotlib pillow pyaudio
```

### Passo 2: Installa Dipendenze Python

```bash
# Installa dipendenze principali
pip install librosa soundfile scipy numpy pydub matplotlib pillow pyaudio
```

### Passo 3: Scarica INFINI LOOP

```bash
# Clona il repository o scarica i file
git clone https://github.com/yourusername/infiniloop.git
cd infiniloop
```

### Passo 4: Scarica Binario MusicGPT

1. Visita la [pagina releases di MusicGPT](https://github.com/gabotechs/MusicGPT/releases)
2. Scarica `musicgpt-x86_64-unknown-linux-gnu`
3. Posizionalo nella stessa directory degli script Python (infiniloop)
4. Rendilo eseguibile:

```bash
chmod +x musicgpt-x86_64-unknown-linux-gnu
```
## Utilizzo
Nota: Al primo avvio, MusicGPT scaricherà il modello selezionato (di default medium). La primissima generazione sarà significativamente più lenta rispetto alle successive. Questo vale per ogni versione di INFINI LOOP.

### Versione Terminale (`ilterm.py`)

#### Avvio Rapido

```bash
# Modalità interattiva
python ilterm.py

# Con impostazioni personalizzate
python ilterm.py --prompt "electronic dance" --duration 20 --driver pulse
```

#### Comandi Interattivi

In modalità interattiva:

- `start <prompt>` - Avvia loop infinito con il prompt dato
- `stop` - Ferma riproduzione corrente
- `status` - Mostra stato sistema
- `save <nomefile>` - Salva loop corrente
- `set duration` - Cambia durata generazione (5-30 secondi)
- `set driver` - Cambia driver audio (pulse/alsa/dsp)
- `debug on/off` - Attiva/disattiva messaggi debug
- `help` - Mostra tutti i comandi
- `quit` - Esci dal programma

### Versione GUI Leggera (`il1.py`)

```bash
python il1.py
```

1. Inserisci un prompt musicale (es. "calm acoustic guitar")
2. Regola durata se necessario (10-15 secondi ottimali)
3. Clicca "START" per iniziare generazione
4. Usa "SAVE" per esportare il loop corrente
5. Clicca "QUIT" per uscire
6. Clicca "STATUS" per sapere che succede

### Versione GUI Avanzata (`il2.py`)

```bash
python il2.py
```

1. Inserisci il tuo prompt musicale
2. Scegli tipo algoritmo:
   - Avanzato: Analisi multi-metrica (consigliato)
   - Classico: Solo similarità spettrale
3. Seleziona modello e durata
4. Clicca "AVVIA" per iniziare
5. Monitora le visualizzazioni per analisi loop
6. Regola slider overlap (attualmente non funzionale)
7. Salva loop usando pulsante "SALVA"

## Dettagli Tecnici

### Algoritmo Rilevamento Loop

INFINI LOOP utilizza analisi multi-metrica sofisticata:

1. **Similarità Spettrale**: Confronta contenuto in frequenza ai potenziali punti loop
2. **Continuità Forma d'Onda**: Assicura transizioni fluide di ampiezza
3. **Allineamento Beat**: Sincronizza loop con ritmo rilevato
4. **Continuità Fase**: Mantiene coerenza di fase per suono naturale
5. **Ottimizzazione Zero-Crossing**: Perfeziona punti loop

### Pipeline Elaborazione Audio

1. AI genera audio grezzo usando MusicGPT
2. Algoritmo rilevamento loop analizza l'audio
3. Punti loop ottimali vengono identificati e raffinati
4. Audio viene normalizzato e preparato per riproduzione
5. Regioni crossfade vengono calcolate (solo terminale e GUI leggera)
6. Riproduzione continua inizia mentre genera segmento successivo

### Raccomandazioni Modelli

- **Modello Small**: Veloce ma spesso bassa qualità - non consigliato
- **Modello Medium**: Miglior bilanciamento qualità/velocità - consigliato
- **Modello Large**: Qualità massima ma lento e intensivo di risorse

## Risoluzione Problemi

### Problemi Comuni

**Problema**: "File audio generato con errori dalla AI"
- **Soluzione**: Assicurati che binario MusicGPT sia eseguibile e nella posizione corretta

**Problema**: Riproduzione audio salta o balbetta
- **Soluzione**: Prova driver audio diverso (pulse, alsa o dsp)

**Problema**: Generazione richiede troppo tempo
- **Soluzione**: Usa modello medium e durata più breve (10-15 secondi)

**Problema**: Nessun output audio
- **Soluzione**: Controlla impostazioni audio sistema e assicurati che sottosistema audio sia in esecuzione

### Modalità Debug

Abilita modalità debug per log dettagliati:

```bash
# Versione terminale
python ilterm.py --debug

# In modalità interattiva
debug on
```

### To-Do

- Supporto preset per generare musica con un click
- Integrazione con modello AI visuale, con cambio musica in base alle attività rilevate su schermo
- Supporto crossfade sulla Advanced GUI e miglioramenti generali
- Fade out/in durante il mixaggio delle tracce

## Licenza

Questo progetto è rilasciato sotto licenza Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0). Per uso commerciale, contattare gli autori.

## Crediti

- MusicGPT di gabotechs per generazione musicale AI in locale: https://github.com/gabotechs/MusicGPT
- Sviluppato con assistenza di modelli linguistici AI
- Elaborazione audio basata su librosa e soundfile

## Contribuire

Segnalazioni bug e richieste funzionalità sono benvenute. Si prega di includere:

- Versione utilizzata (ilterm.py, il1.py o il2.py)
- Sistema operativo e versione Python
- Messaggi di errore completi
- Passaggi per riprodurre il problema
