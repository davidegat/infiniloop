# INFINI LOOP - Infinite Local AI Music Generation

INFINI LOOP is an AI-powered music system designed to generate seamless, infinite audio loops. It automatically creates new musical fragments using AI, detects optimal loop points, and plays them continuously while preparing the next segment—resulting in a smooth, never-ending stream of always-fresh instrumental music.

At startup, one of two pre-included .wav files will play, so you can enjoy music immediately while the first AI generation is being prepared.

Once set up and running, your machine becomes a local AI music station, continuously producing new tracks with smooth transitions. It's local, private, and more personal than any YouTube or Spotify playlist.

INFINI LOOP is powered by MusicGPT (https://github.com/gabotechs/MusicGPT) and MusicGen models by Meta (https://huggingface.co/spaces/facebook/MusicGen). According to Meta, the model was trained on licensed data from the following sources: the Meta Music Initiative Sound Collection, Shutterstock music collection, and the Pond5 music collection. See the paper for more details about the training set and corresponding preprocessing.

All audios (and statistics) generated with INFINI LOOP are produced locally and never sent to, or require, any external service. Music generated with MusicGen models is entirely owned by the user, who retains full rights to use, distribute, modify, and commercialize it without restrictions. **If you obtain a loop by using INFINI LOOP, anyway, you are not allowed to use it commercially.** (See [License](#license))

GUI version:

[Kooha-2025-08-07-02-49-42.webm](https://github.com/user-attachments/assets/410cd31e-7a3e-4932-a73f-c118d45d50b7)

Terminal version:

<img width="500" height="949" alt="immagine" src="https://github.com/user-attachments/assets/0c75be8f-b8b7-4f3b-be23-db726700feae" />

## Table of Contents

- [Features](#features)
- [Available Versions](#available-versions)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Test Results](#test-results)
- [Tips and Tricks](#tips-and-tricks)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Features

- **Local AI Music Generation**: Powered by MusicGPT for high-quality audio synthesis
- **Advanced Loop Detection**: Multi-metric analysis with configurable minimum loop length
- **Seamless Playback**: Native infinite looping with crossfade transitions
- **Flexible Generation**: Configurable AI sample duration (5-30s) and minimum loop requirements
- **Smart Loop Management**: Configurable minimum song duration before switching (10-300s)
- **Multiple Interfaces**: Terminal and GUI versions with real-time monitoring
- **Preset System**: Quick generation with pre-configured musical styles
- **Benchmark System**: Performance monitoring and generation time statistics
- **Export Functionality**: Save generated loops for later use

## Available Versions

### 1. GUI Version (`il1.py`) - Recommended ⭐

- Clean graphical interface with tabbed organization
- Preset System for quick generation
- Statistics Tab: Benchmark data and generation time tracking
- Settings Tab: Configure all generation parameters and audio driver
- Loop Information: Display random titles, artists, duration, and genre
- Real-time status monitoring with timing information
- Save current loop functionality

### 2. Terminal Version (`ilterm.py`) - Most Stable ⭐

- Command-line interface with full interactive mode
- Debug mode with detailed logging
- Generation-only mode for single loops
- Complete settings configuration including model selection
- Lowest resource consumption
- Benchmark system for tracking generation performance

## System Requirements

- **Operating System**: Ubuntu 22.04 LTS
- **Python**: 3.10 or higher
- **RAM**: 16 GB system memory
- **CPU**: Modern high-frequency CPU (3+ GHz, 4+ cores)
- **Audio**: PulseAudio with low-latency configuration
- **Storage**: 8 GB or more, depending on the model 

## Installation

### Step 1: Install System Dependencies

```bash
# For Ubuntu/Debian:
sudo apt update
sudo apt install -y ffmpeg pulseaudio-utils alsa-utils python3-pip python3-dev \
                    portaudio19-dev python3-tk build-essential libasound2-dev \
                    libportaudio2 libportaudiocpp0 portaudio19-dev
```

### Step 2: Install Python Dependencies

```bash
# Install core audio processing libraries
pip install librosa soundfile scipy numpy pydub matplotlib pillow \
           pyaudio psutil pyloudnorm

# Optional: Install with specific versions for stability
pip install librosa==0.10.1 soundfile==0.12.1 scipy==1.11.4 \
           numpy==1.24.3 pyloudnorm==0.2.1
```

### Step 3: Download INFINI LOOP

```bash
# Clone the repository or download the files
git clone https://github.com/yourusername/infiniloop.git
cd infiniloop

# Make scripts executable
chmod +x *.py
```

### Step 4: Download and Setup MusicGPT Binary

1. Visit the [MusicGPT releases page](https://github.com/gabotechs/MusicGPT/releases)
2. Download `musicgpt-x86_64-unknown-linux-gnu` (latest version)
3. Place it in the same directory as the Python scripts
4. Make it executable:

```bash
chmod +x musicgpt-x86_64-unknown-linux-gnu

# Verify the binary works
./musicgpt-x86_64-unknown-linux-gnu --help
```

### Step 5: Optional: Configure Low-Latency Audio

```bash
echo "default-sample-rate = 44100" >> ~/.pulse/daemon.conf
echo "alternate-sample-rate = 48000" >> ~/.pulse/daemon.conf
pulseaudio --kill && pulseaudio --start
```
### Important: Model Download

MusicGPT will download the selected model on first use. The very first generation will be significantly slower than subsequent ones. GUI Version has an initial support for model downloading and deleting, but you may want to download and test your desired model from terminal first:

```bash
# For small model (fast, good quality, less variety - recommended)
./musicgpt-x86_64-unknown-linux-gnu "Create a relaxing LoFi song" --model small

# For medium model (balanced quality and speed)
./musicgpt-x86_64-unknown-linux-gnu "Create a relaxing LoFi song" --model medium

# For large model (highest quality but very slow - not recommended)
./musicgpt-x86_64-unknown-linux-gnu "Create a relaxing LoFi song" --model large
```

This will download the required model and give you an idea of generation times before using INFINI LOOP.

## Usage

### GUI Version (`il1.py`) - Recommended

```bash
python il1.py
```

**Usage Steps:**
1. **Enter Prompt**: Type your musical description (e.g., "calm acoustic guitar")
2. **Choose Model**: Select small (fast), medium (balanced), or large (slow but high quality)
3. **Set Sample Duration**: Configure AI generation length (5-30 seconds)
4. **Set Minimum Loop Length**: Set required loop duration for acceptance
5. **Choose Preset**: Click preset buttons for quick setups
6. **Configure Settings**: 
   - **Minimum Song Duration**: How long each loop plays before switching (10-300 seconds)
   - **Audio Driver**: pulse/alsa/dsp
7. **Start Generation**: Click "▶️ START LOOP" 
8. **Monitor Progress**: Watch status bar, loop information, and timing
9. **View Statistics**: Check generation time benchmarks in Statistics tab
10. **Save Loops**: Use "💾 Save Current Loop" to export

### Terminal Version (`ilterm.py`)

#### Quick Start

```bash
# Interactive mode (recommended)
python ilterm.py

# Direct generation with specific model
python ilterm.py --prompt "electronic dance loop" --model small

#### Interactive Commands

```bash
🎛️ > start 'ambient electronic loop'     # Start infinite loop
🎛️ > stop                                # Stop playback
🎛️ > status                              # Show detailed status
🎛️ > save my_loop.wav                   # Export current loop
🎛️ > set model                          # Change AI model (small/medium/large)
🎛️ > set duration                        # Change sample generation length (5-30s)
🎛️ > set minlength                       # Change minimum loop length for acceptance
🎛️ > set minduration                     # Change minimum song duration (10-300s)
🎛️ > set driver                          # Change audio driver
🎛️ > debug on/off                        # Toggle debug mode
🎛️ > help                                # Show all commands
🎛️ > quit                                # Exit program
```

#### Command Line Options

```bash
python ilterm.py [OPTIONS]

Options:
  -p, --prompt TEXT        Music generation prompt
  -m, --model [small|medium|large]  AI model selection
  -d, --duration INTEGER   Sample generation duration (5-30 seconds)
  --minlength INTEGER      Minimum loop length for acceptance (seconds)
  --driver [pulse|alsa|dsp]  Audio driver selection
  -v, --verbose            Detailed output
  -h, --help               Show help message
```

## Test results

**Test environment**  
Garuda Linux (Wayland), KDE Plasma 6.3.5, Kernel 6.15.1-zen, Intel® Core™ Ultra 5 125H (18 threads), 30.9 GiB RAM, Intel® Arc, GEEKOM GT1 Mega.

**Parameters in GUI**
- **Sample length** → length of generated sample
- **Song duration** → minimum play time before loop change
- **Loop length** → minimum accepted loop length
- Generation times estimated using local statistics (Benchmark tab)

### Per-model results

| Model | Size (approx.) | Recommended Sample length | Observed gen time | Suggested Song duration |
|---|---:|---:|---:|---|
| **small** | ~1.6 GB | 7–10 s | 22–30 s | 30 s ≈ near-continuous generation (higher CPU) • 60 s = generation pauses until end (lower CPU) |
| **medium** | ~7 GB | 7 s | 60–70 s | 60 s ≈ near-continuous generation (higher CPU) • 90 s = generation pauses until end (lower CPU) |

**How we measured**  
Generation times were calculated using INFINI LOOP's built-in benchmarking system, visible in the *Benchmark tab*. For each generation, the software records the **Sample length** setting and the actual processing time. Data is saved and averaged across multiple sessions, providing a realistic value for your specific system.

## Tips and Tricks

- **Rapid Looping**: Set the song duration lower than the average AI generation time to force quick transitions. Enable statistics to monitor timing data in the Stats tab.
- **Genre-Specific Tuning**: Different genres may require different settings. Try adjusting the minimum loop length and the sample duration for better results.  
  Suggested keywords: *"4/4"*, *"loop"*, *"seamless"*, and specific instrument or mood terms relevant to the genre (e.g., *"dub bass"*, *"jazz piano"*, *"ambient pad"*).
- **Shorter Samples Work Better (Sometimes)**: For some genres (like lofi or jazz), shorter audio samples tend to yield cleaner, more natural loops.
- **Loop Quality Depends on Structure**: The system prioritizes loops that align with musical phrasing, especially 4, 8, or 16 beat segments. Prompts that imply strong structure help detection.  
  Suggested keywords: *"4/4"*, *"nointro"*, *"loop"*, *"tight rhythm"*, *"groove locked"*.
- **Beat Alignment Matters**: Rhythmic consistency improves loop quality. The detection system rewards predictable, well-aligned beats.  
  Suggested keywords: *"punchy drums"*, *"tight bass"*, *"drum machine"*, *"quantized"*.
- **Model Size vs. Variety**: Larger models (e.g., "large") offer more variety in style and instrumentation, but not necessarily better audio quality. Smaller models are faster and more consistent.
- **Start with Presets**: The built-in presets are designed with tested keyword combinations. Use them as a base and modify as needed for your style.
- **Zero-Crossing Enhancements**: Loop transitions are smoothed using zero-crossing detection. Samples with clean fades and minimal background noise produce better results.  
  Suggested keywords: *"clean mix"*, *"no noise"*, *"no fade out"*.
- **Structure Bonuses**: The system gives bonus scores to loops that match musical phrases (like 4, 8, or 16 beats). Matching the prompt and duration to a musical phrase improves results.  
  Suggested keywords: *"8 bar phrase"*, *"tight measure"*, *"loopable section"*.
- **Beat Consistency Matters**: The detection algorithm favors samples with stable tempo and rhythm. If you're getting inconsistent results, use terms that imply mechanical or steady timing.  
  Suggested keywords: *"steady tempo"*, *"quantized beat"*, *"drum machine"*, *"metronome feel"*.
- **Benchmarking Helps**: Enable benchmark tracking to compare sample duration with actual AI generation time. This helps you find efficient duration settings for your system and model.
- **Adjust Minimum Loop Length**: If loops feel too short or too long, experiment with the minimum loop length setting. Increase for more complex loops, decrease for minimal or repetitive styles.




## Technical Details

### Model Comparison & Recommendations

- **Small Model** ⭐⭐⭐ **RECOMMENDED**: 
  - Generation time: Fast (typically 15-30 seconds)
  - Quality: Good audio fidelity
  - Variety: Less musical complexity
  - Best for: Quick testing, longer loops, fast iteration

- **Medium Model** ⭐⭐ **RECOMMENDED**:
  - Generation time: Moderate (typically 30-60 seconds)
  - Quality: Better audio fidelity with good musical structure
  - Variety: Excellent balance of creativity and coherence
  - Best for: General use, balanced performance

- **Large Model**:
  - Generation time: Very slow (typically 2-5 minutes)
  - Quality: High fidelity and musical complexity
  - Variety: Maximum creative diversity
  - Best for: Special projects only - **not recommended for regular use**

## Troubleshooting

### Common Issues and Solutions

#### Generation Issues

**Issue**: Very slow generation
- **Cause**: Wrong model selection or insufficient system resources
- **Solutions**:
  - Try small model for faster generation
  - Test generation times from terminal first using the command provided in the Usage section
  - Use shorter duration (8-12 seconds)
  - Close other applications

**Issue**: "No interesting loop" or poor quality
- **Cause**: AI generated audio not suitable for looping or minimum loop length too restrictive
- **Solutions**:
  - Increase AI sample duration to 15-20 seconds
  - Lower minimum loop length requirement (try 2.0-3.0 seconds)
  - Try different models (medium often works better than small for complex prompts)
  - Use preset buttons for proven prompts
  - Add keywords: "seamless", "loopable", "continuous"

#### Audio Playback Issues

**Issue**: No audio output
- **Solutions**:
  ```bash
  # Test audio system
  speaker-test -t wav -c 2
  
  # Try different drivers in settings
  # GUI: Settings tab → Audio Driver
  # Terminal: set driver
  ```

**Issue**: Audio stuttering
- **Cause**: Audio buffer issues or driver problems
- **Solutions**:
  - Try different audio drivers (pulse → alsa → dsp)
  - Close unnecessary applications
  - Check system resource usage

### Performance Optimization

#### For Quick Generation:
- Use **small model** with 10-20 second duration
- Set minimum loop length to 3-5 seconds
- Use shorter minimum song duration (20-30 seconds)

#### For More Variety:
- Use **medium model** with 10-15 second duration  
- Set minimum loop length to 2.5-4 seconds
- Use longer minimum song duration (60-120 seconds)

Always test your chosen model from the terminal first to understand generation times:
```bash
./musicgpt-x86_64-unknown-linux-gnu "test prompt" --model your_chosen_model
```
Or use the statistics feature to log generation times.

### Custom Preset Creation

Modify presets in `il1.py`:

```python
presets = {
    "Custom Jazz": "smooth jazz piano trio seamless loop",
    "Epic Cinema": "cinematic orchestral epic soundtrack loop",
    # Add your own presets here
}
```
UI configuration of presets may be a future feature.

## License

This project is released under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0). 

**Permissions:**
- ✅ Share and adapt the material
- ✅ Use for personal and educational purposes
- ✅ Modify and build upon the code

**Restrictions:**
- ❌ Commercial use without permission
- ❌ Distribution without attribution

**If you obtain a loop by using INFINI LOOP, anyway, you are not allowed to use it commercially.**
For commercial use of this software, please contact the authors.

## Credits

- **MusicGPT** by gabotechs for AI music generation: https://github.com/gabotechs/MusicGPT
- **MusicGen models** by Meta for AI audio synthesis: https://huggingface.co/spaces/facebook/MusicGen
- **librosa** team for advanced audio analysis and processing: https://librosa.org/
- **SciPy community** for scientific computing algorithms: https://scipy.org/
- **NumPy developers** for numerical computing foundation: https://numpy.org/
- **SoundFile** by bastibe for audio file I/O: https://github.com/bastibe/python-soundfile
- **pydub** by jiaaro for audio manipulation: https://github.com/jiaaro/pydub
- **pyloudnorm** by csteinmetz1 for audio loudness normalization: https://github.com/csteinmetz1/pyloudnorm
- **psutil** by giampaolo for system monitoring: https://github.com/giampaolo/psutil
- **FFmpeg team** for multimedia framework and ffplay: https://ffmpeg.org/
- **PulseAudio developers** for Linux audio system: https://www.freedesktop.org/wiki/Software/PulseAudio/
- **Python Software Foundation** for Python programming language and tkinter GUI toolkit: https://www.python.org/
- Developed with assistance from language models
---
**🎵 Enjoy infinite AI music with INFINI LOOP! 🎵**

---
# INFINI LOOP - Generazione Musicale AI Locale Infinita

INFINI LOOP è un sistema musicale basato su AI progettato per generare loop audio continui e fluidi. Crea automaticamente nuovi frammenti musicali usando l'AI, rileva i punti di loop ottimali e li riproduce continuamente mentre prepara il segmento successivo—risultando in un flusso fluido e infinito di musica strumentale sempre fresca.

All'avvio, uno dei due file .wav pre-inclusi verrà riprodotto, così potrai goderti la musica immediatamente mentre la prima generazione AI viene preparata.

Una volta configurato e in esecuzione, la tua macchina diventa una stazione musicale AI locale, che produce continuamente nuove tracce con transizioni fluide. È locale, privato e più personale di qualsiasi playlist di YouTube o Spotify.

INFINI LOOP è alimentato da MusicGPT (https://github.com/gabotechs/MusicGPT) e i modelli MusicGen di Meta (https://huggingface.co/spaces/facebook/MusicGen). Secondo Meta, il modello è stato addestrato su dati con licenza dalle seguenti fonti: la Meta Music Initiative Sound Collection, la collezione musicale Shutterstock e la collezione musicale Pond5. Vedi il documento per maggiori dettagli sul set di addestramento e il preprocessing corrispondente.

Tutti i file audio (e le statistiche) generati con INFINI LOOP sono prodotti localmente e non vengono inviati a, nè fanno uso di, servizi esterni. La musica generata dai modelli MusicGen è interamente di proprietà dell'utente, che mantiene tutti i diritti per usare, distribuire, modificare e commercializzare senza restrizioni. **Se hai creato un loop usando INFINI LOOP, tuttavia, non sei autorizzato ad utilizzarlo per scopi commerciali.** (Vedi [Licenza](#licenza))

Versione GUI:

[Kooha-2025-08-07-02-49-42.webm](https://github.com/user-attachments/assets/410cd31e-7a3e-4932-a73f-c118d45d50b7)

Versione Terminale:

<img width="500" height="949" alt="immagine" src="https://github.com/user-attachments/assets/0c75be8f-b8b7-4f3b-be23-db726700feae" />

## Indice

- [Caratteristiche](#caratteristiche)
- [Versioni Disponibili](#versioni-disponibili)
- [Requisiti di Sistema](#requisiti-di-sistema)
- [Installazione](#installazione)
- [Utilizzo](#utilizzo)
- [Test Eseguiti](#test-eseguiti)
- [Trucchi e consigli](#trucchi-e-consigli)
- [Dettagli Tecnici](#dettagli-tecnici)
- [Risoluzione dei Problemi](#risoluzione-dei-problemi)
- [Licenza](#licenza)

## Caratteristiche

- **Generazione Musicale AI Locale**: Alimentato da MusicGPT per sintesi audio di alta qualità
- **Rilevamento Loop Avanzato**: Analisi multi-metrica con lunghezza minima del loop configurabile
- **Riproduzione Fluida**: Loop infinito nativo con transizioni crossfade
- **Generazione Flessibile**: Durata campione AI configurabile (5-30s) e requisiti minimi di loop
- **Gestione Loop Intelligente**: Durata minima canzone configurabile prima del cambio (10-300s)
- **Interfacce Multiple**: Versioni terminale e GUI con monitoraggio in tempo reale
- **Sistema Preset**: Generazione rapida con stili musicali pre-configurati
- **Sistema Benchmark**: Monitoraggio delle prestazioni e statistiche dei tempi di generazione
- **Funzionalità Esportazione**: Salva i loop generati per uso successivo

## Versioni Disponibili

### 1. Versione GUI (`il1.py`) - Raccomandata ⭐

- Interfaccia grafica pulita con organizzazione a schede
- Sistema Preset per generazione rapida
- Scheda Statistiche: Dati benchmark e tracciamento tempi di generazione
- Scheda Impostazioni: Configura tutti i parametri di generazione e driver audio
- Informazioni Loop: Visualizza titoli casuali, artisti, durata e genere
- Monitoraggio stato in tempo reale con informazioni temporali
- Funzionalità salvataggio loop corrente

### 2. Versione Terminale (`ilterm.py`) - Più Stabile ⭐

- Interfaccia a riga di comando con modalità interattiva completa
- Modalità debug con logging dettagliato
- Modalità solo-generazione per loop singoli
- Configurazione impostazioni completa inclusa selezione modello
- Consumo risorse minimo
- Sistema benchmark per tracciare le prestazioni di generazione

## Requisiti Raccomandati

- **Sistema Operativo**: Ubuntu 22.04 LTS
- **Python**: 3.10 o superiore
- **RAM**: 16 GB memoria di sistema
- **CPU**: CPU moderna ad alta frequenza (3+ GHz, 4+ core)
- **Audio**: PulseAudio con configurazione a bassa latenza
- **Archiviazione**: 8 GB o più, a seconda del modello

## Installazione

### Passo 1: Installa Dipendenze di Sistema

```bash
# Per Ubuntu/Debian:
sudo apt update
sudo apt install -y ffmpeg pulseaudio-utils alsa-utils python3-pip python3-dev \
                    portaudio19-dev python3-tk build-essential libasound2-dev \
                    libportaudio2 libportaudiocpp0 portaudio19-dev
```

### Passo 2: Installa Dipendenze Python

```bash
# Installa librerie core per elaborazione audio
pip install librosa soundfile scipy numpy pydub matplotlib pillow \
           pyaudio psutil pyloudnorm

# Opzionale: Installa con versioni specifiche per stabilità
pip install librosa==0.10.1 soundfile==0.12.1 scipy==1.11.4 \
           numpy==1.24.3 pyloudnorm==0.2.1
```

### Passo 3: Scarica INFINI LOOP

```bash
# Clona il repository o scarica i file
git clone https://github.com/yourusername/infiniloop.git
cd infiniloop

# Rendi eseguibili gli script
chmod +x *.py
```

### Passo 4: Scarica e Configura il Binario MusicGPT

1. Visita la [pagina release di MusicGPT](https://github.com/gabotechs/MusicGPT/releases)
2. Scarica `musicgpt-x86_64-unknown-linux-gnu` (versione più recente)
3. Posizionalo nella stessa directory degli script Python
4. Rendilo eseguibile:

```bash
chmod +x musicgpt-x86_64-unknown-linux-gnu

# Verifica che il binario funzioni
./musicgpt-x86_64-unknown-linux-gnu --help
```

### Passo 5: Opzionale: Configura Audio a Bassa Latenza

```bash
echo "default-sample-rate = 44100" >> ~/.pulse/daemon.conf
echo "alternate-sample-rate = 48000" >> ~/.pulse/daemon.conf
pulseaudio --kill && pulseaudio --start
```

### Importante: Download del Modello

MusicGPT scaricherà il modello selezionato al primo utilizzo. La primissima generazione sarà significativamente più lenta delle successive. La Versione GUI ha un supporto iniziale per il download e l'eliminazione dei modelli scaricati, ma potresti voler scaricare e testare il modello desiderato dal terminale prima:

```bash
# Per modello small (veloce, buona qualità, meno varietà - raccomandato)
./musicgpt-x86_64-unknown-linux-gnu "Crea una canzone LoFi rilassante" --model small

# Per modello medium (qualità e velocità bilanciate)
./musicgpt-x86_64-unknown-linux-gnu "Crea una canzone LoFi rilassante" --model medium

# Per modello large (massima qualità ma molto lento - non raccomandato)
./musicgpt-x86_64-unknown-linux-gnu "Crea una canzone LoFi rilassante" --model large
```

Questo scaricherà il modello richiesto e ti darà un'idea dei tempi di generazione prima di usare INFINI LOOP.

## Utilizzo

### Versione GUI (`il1.py`) - Raccomandata

```bash
python il1.py
```

**Passi di Utilizzo:**
1. **Inserisci Prompt**: Digita la tua descrizione musicale (es., "chitarra acustica calma")
2. **Scegli Modello**: Seleziona small (veloce), medium (bilanciato), o large (lento ma alta qualità)
3. **Imposta Durata Campione**: Configura la lunghezza di generazione AI (5-30 secondi)
4. **Imposta Lunghezza Minima Loop**: Imposta la durata loop richiesta per l'accettazione
5. **Scegli Preset**: Clicca i pulsanti preset per configurazioni rapide
6. **Configura Impostazioni**: 
   - **Durata Minima Canzone**: Quanto tempo ogni loop suona prima del cambio (10-300 secondi)
   - **Driver Audio**: pulse/alsa/dsp
7. **Avvia Generazione**: Clicca "▶️ AVVIA LOOP"
8. **Monitora Progresso**: Guarda la barra di stato, informazioni loop e tempistiche
9. **Visualizza Statistiche**: Controlla i benchmark dei tempi di generazione nella scheda Statistiche
10. **Salva Loop**: Usa "💾 Salva Loop Corrente" per esportare

### Versione Terminale (`ilterm.py`)

#### Avvio Rapido

```bash
# Modalità interattiva (raccomandata)
python ilterm.py

# Generazione diretta con modello specifico
python ilterm.py --prompt "loop elettronico dance" --model small
```

#### Comandi Interattivi

```bash
🎛️ > start 'loop elettronico ambient'     # Avvia loop infinito
🎛️ > stop                                # Ferma riproduzione
🎛️ > status                              # Mostra stato dettagliato
🎛️ > save mio_loop.wav                   # Esporta loop corrente
🎛️ > set model                           # Cambia modello AI (small/medium/large)
🎛️ > set duration                        # Cambia lunghezza generazione campione (5-30s)
🎛️ > set minlength                       # Cambia lunghezza minima loop per accettazione
🎛️ > set minduration                     # Cambia durata minima canzone (10-300s)
🎛️ > set driver                          # Cambia driver audio
🎛️ > debug on/off                        # Attiva/disattiva modalità debug
🎛️ > help                                # Mostra tutti i comandi
🎛️ > quit                                # Esci dal programma
```

#### Opzioni Riga di Comando

```bash
python ilterm.py [OPZIONI]

Opzioni:
  -p, --prompt TEXT        Prompt per generazione musicale
  -m, --model [small|medium|large]  Selezione modello AI
  -d, --duration INTEGER   Durata generazione campione (5-30 secondi)
  --minlength INTEGER      Lunghezza minima loop per accettazione (secondi)
  --driver [pulse|alsa|dsp]  Selezione driver audio
  -v, --verbose            Output dettagliato
  -h, --help               Mostra messaggio aiuto
```

## Test eseguit

**Ambiente di test**  
Garuda Linux (Wayland), KDE Plasma 6.3.5, Kernel 6.15.1-zen, Intel® Core™ Ultra 5 125H (18 thread), 30.9 GiB RAM, Intel® Arc, GEEKOM GT1 Mega.

**Parametri nella GUI**
- **Sample length** → lunghezza del campione generato
- **Song duration** → durata minima di riproduzione del loop prima del cambio
- **Loop length** → lunghezza minima del loop accettato
- I tempi di generazione sono stimati usando le statistiche locali (Benchmark tab)

### Risultati per modello

| Modello | Dimensione (circa) | Sample length consigliata | Tempo di generazione osservato | Song duration suggerita |
|---|---:|---:|---:|---|
| **small** | ~1.6 GB | 7–10 s | 22–30 s | 30 s ≈ generazione quasi continua (CPU ↑) • 60 s = pausa fino a fine brano (CPU ↓) |
| **medium** | ~7 GB | 7 s | 60–70 s | 60 s ≈ generazione quasi continua (CPU ↑) • 90 s = pausa fino a fine brano (CPU ↓) |

**Come sono stati misurati**  
I tempi di generazione sono stati calcolati utilizzando il sistema di benchmarking interno di INFINI LOOP, visibile nel *Benchmark tab*. Per ogni generazione, il software registra la durata impostata in **Sample length** e il tempo reale di elaborazione. I dati vengono salvati e mediati tra più sessioni, fornendo un valore realistico per il proprio sistema.


## Trucchi e consigli ##

- **Loop Rapidi**: Imposta la durata della canzone a un valore inferiore al tempo medio di generazione dell’AI per forzare transizioni rapide. Attiva le statistiche per monitorare i tempi nella scheda Statistiche.
- **Regolazioni per Genere**: Generi diversi possono richiedere impostazioni diverse. Prova a modificare la lunghezza minima del loop e la durata del campione per ottenere risultati migliori.  
  Parole chiave consigliate: *"4/4"*, *"loop"*, *"seamless"*, e termini specifici per strumenti o atmosfera (es. *"dub bass"*, *"jazz piano"*, *"ambient pad"*).
- **Campioni più Corti Funzionano Meglio (a Volte)**: In alcuni generi (come lofi o jazz), campioni audio più brevi generano loop più puliti e naturali.
- **La Qualità del Loop Dipende dalla Struttura**: Il sistema dà priorità ai loop che seguono frasi musicali ben definite, in particolare segmenti di 4, 8 o 16 battute. Prompt che suggeriscono una struttura forte aiutano il rilevamento.  
  Parole chiave consigliate: *"4/4"*, *"nointro"*, *"loop"*, *"tight rhythm"*, *"groove locked"*.
- **L’Allineamento del Beat è Importante**: Una ritmica coerente migliora la qualità del loop. Il sistema di rilevamento premia beat regolari e ben allineati.  
  Parole chiave consigliate: *"punchy drums"*, *"tight bass"*, *"drum machine"*, *"quantized"*.
- **Dimensione del Modello vs. Varietà**: I modelli più grandi (es. "large") offrono maggiore varietà nello stile e negli strumenti, ma non necessariamente una qualità audio superiore. I modelli più piccoli sono più veloci e coerenti.
- **Usa i Preset**: I preset integrati sono progettati con combinazioni di parole chiave testate. Usali come base e modificali in base al tuo stile.
- **Ottimizzazione Zero-Crossing**: Le transizioni nei loop sono rese più fluide grazie al rilevamento zero-crossing. I campioni con dissolvenze pulite e poco rumore di fondo danno risultati migliori.  
  Parole chiave consigliate: *"clean mix"*, *"no noise"*, *"no fade out"*.
- **Bonus Strutturali**: Il sistema assegna punteggi bonus ai loop che seguono frasi musicali standard (come 4, 8 o 16 battute). Abbinare il prompt e la durata a una frase musicale migliora i risultati.  
  Parole chiave consigliate: *"8 bar phrase"*, *"tight measure"*, *"loopable section"*.
- **Costanza del Beat**: L’algoritmo di rilevamento favorisce i campioni con tempo e ritmo stabili. Se ottieni risultati incoerenti, usa termini che suggeriscano un timing regolare o meccanico.  
  Parole chiave consigliate: *"steady tempo"*, *"quantized beat"*, *"drum machine"*, *"metronome feel"*.
- **Il Benchmark è Utile**: Attiva il tracciamento delle performance per confrontare la durata richiesta con il tempo reale di generazione. Questo ti aiuta a trovare le durate più efficienti per il tuo sistema e modello.
- **Regola la Lunghezza Minima del Loop**: Se i loop risultano troppo brevi o troppo lunghi, sperimenta con il parametro di lunghezza minima. Aumentalo per strutture più complesse, riducilo per stili minimalisti o ripetitivi.


## Dettagli Tecnici

### Confronto Modelli e Raccomandazioni

- **Modello Small** ⭐ **RACCOMANDATO**: 
  - Tempo di generazione: Veloce (tipicamente 15-30 secondi)
  - Qualità: Buona fedeltà audio
  - Varietà: Meno complessità musicale
  - Migliore per: Test rapidi, loop più lunghi, iterazione veloce

- **Modello Medium** ⭐⭐ **RACCOMANDATO**:
  - Tempo di generazione: Moderato (tipicamente 30-60 secondi)
  - Qualità: Migliore fedeltà audio con buona struttura musicale
  - Varietà: Eccellente bilanciamento di creatività e coerenza
  - Migliore per: Uso generale, prestazioni bilanciate

- **Modello Large**:
  - Tempo di generazione: Molto lento (tipicamente 2-5 minuti)
  - Qualità: Alta fedeltà e complessità musicale
  - Varietà: Massima diversità creativa
  - Migliore per: Solo progetti speciali - **non raccomandato per uso regolare**

## Risoluzione dei Problemi

### Problemi Comuni e Soluzioni

#### Problemi di Generazione

**Problema**: Generazione molto lenta
- **Causa**: Selezione modello sbagliata o risorse di sistema insufficienti
- **Soluzioni**:
  - Prova il modello small per generazione più veloce
  - Testa i tempi di generazione dal terminale prima usando il comando fornito nella sezione Utilizzo
  - Usa durata più breve (8-12 secondi)
  - Chiudi altre applicazioni

**Problema**: "No interesting loop" o qualità scarsa
- **Causa**: Audio generato dall'AI non adatto per loop o lunghezza minima loop troppo restrittiva
- **Soluzioni**:
  - Aumenta durata campione AI a 15-20 secondi
  - Abbassa requisito lunghezza minima loop (prova 2.0-3.0 secondi)
  - Prova modelli diversi (medium spesso funziona meglio di small per prompt complessi)
  - Usa pulsanti preset per prompt provati
  - Aggiungi parole chiave: "seamless", "loopable", "continuous"

#### Problemi Riproduzione Audio

**Problema**: Nessun output audio
- **Soluzioni**:
  ```bash
  # Testa sistema audio
  speaker-test -t wav -c 2
  
  # Prova driver diversi nelle impostazioni
  # GUI: Scheda Impostazioni → Driver Audio
  # Terminale: set driver
  ```

**Problema**: Audio frammentato
- **Causa**: Problemi buffer audio o problemi driver
- **Soluzioni**:
  - Prova driver audio diversi (pulse → alsa → dsp)
  - Chiudi applicazioni non necessarie
  - Controlla uso risorse di sistema

### Ottimizzazione Prestazioni

#### Per Generazione Rapida:
- Usa **modello small** con durata 10-20 secondi
- Imposta lunghezza minima loop a 3-5 secondi
- Usa durata minima canzone più breve (20-30 secondi)

#### Per Più Varietà:
- Usa **modello medium** con durata 10-15 secondi  
- Imposta lunghezza minima loop a 2.5-4 secondi
- Usa durata minima canzone più lunga (60-120 secondi)

Testa sempre il tuo modello scelto dal terminale prima per capire i tempi di generazione:
```bash
./musicgpt-x86_64-unknown-linux-gnu "prompt di test" --model tuo_modello_scelto
```
O usa la funzionalità statistiche per registrare i tempi di generazione.

### Creazione Preset Personalizzati

Modifica i preset in `il1.py`:

```python
presets = {
    "Jazz Personalizzato": "trio piano jazz smooth loop continuo",
    "Cinema Epico": "orchestrale cinematico epico soundtrack loop",
    # Aggiungi i tuoi preset qui
}
```
La configurazione UI dei preset potrebbe essere una funzionalità futura.

## Licenza

Questo progetto è rilasciato sotto la Licenza Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0). 

**Permessi:**
- ✅ Condividere e adattare il materiale
- ✅ Usare per scopi personali ed educativi
- ✅ Modificare e costruire sul codice

**Restrizioni:**
- ❌ Uso commerciale senza permesso
- ❌ Distribuzione senza attribuzione

**Se hai creato un loop usando INFINI LOOP, tuttavia, non sei autorizzato ad utilizzarlo per scopi commerciali.**
Per l'uso commerciale di questo software, contatta gli autori.

## Crediti

- **MusicGPT** di gabotechs per la generazione musicale AI: https://github.com/gabotechs/MusicGPT
- **Modelli MusicGen** di Meta per la sintesi audio AI: https://huggingface.co/spaces/facebook/MusicGen
- **Team librosa** per l'analisi e elaborazione audio avanzata: https://librosa.org/
- **Comunità SciPy** per algoritmi di calcolo scientifico: https://scipy.org/
- **Sviluppatori NumPy** per le fondamenta del calcolo numerico: https://numpy.org/
- **SoundFile** di bastibe per I/O file audio: https://github.com/bastibe/python-soundfile
- **pydub** di jiaaro per la manipolazione audio: https://github.com/jiaaro/pydub
- **pyloudnorm** di csteinmetz1 per la normalizzazione loudness audio: https://github.com/csteinmetz1/pyloudnorm
- **psutil** di giampaolo per il monitoraggio di sistema: https://github.com/giampaolo/psutil
- **Team FFmpeg** per il framework multimediale e ffplay: https://ffmpeg.org/
- **Sviluppatori PulseAudio** per il sistema audio Linux: https://www.freedesktop.org/wiki/Software/PulseAudio/
- **Python Software Foundation** per il linguaggio Python e toolkit GUI tkinter: https://www.python.org/
- Sviluppato con l'assistenza di modelli linguistici

---
**🎵 Goditi musica AI infinita con INFINI LOOP! 🎵**
