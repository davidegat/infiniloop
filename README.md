# INFINI LOOP - Infinite Local AI Music Generation

INFINI LOOP is an AI-powered music system designed to generate seamless, infinite audio loops.
It automatically creates new musical fragments using AI, detects the best loop points, and plays them continuously while preparing the next one — resulting in a smooth, never-ending, always new stream of instrumental music.

At startup, one of two pre-included .wav files will play, so you can enjoy music immediately while the first AI generation is being prepared.

Once set up and running, your machine becomes a local AI music station, continuously producing new tracks with transitions. Local, private, more personal than any YouTube or Spotify playlist.

Advanced GUI version (**Experimental - NOT recommended**):

<img width="500" alt="immagine" src="https://github.com/user-attachments/assets/19390959-ba49-476e-8b26-df606c6dad36" />

Terminal version (**recommended for advanced users**):

<img width="500" alt="immagine" src="https://github.com/user-attachments/assets/9a95d2dd-8690-4d00-8735-530511ef9498" />

Lightweight GUI version (**MOST recommended**):

<img width="500" height="881" alt="immagine" src="https://github.com/user-attachments/assets/1bcbf69c-a16e-47ec-afe1-2749a3fc2228" />

## Table of Contents

- [Features](#features)
- [Available Versions](#available-versions)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)
- [Development and Contributing](#development-and-contributing)
- [License](#license)

## Features

- **Local AI Music Generation**: Powered by MusicGPT for high-quality audio synthesis (https://github.com/gabotechs/MusicGPT)
- **Advanced Loop Detection**: Multi-metric analysis with adaptive algorithms
  - Spectral similarity analysis
  - Waveform continuity measurement
  - Beat alignment and rhythm preservation
  - Phase coherence optimization
  - Zero-crossing refinement
- **Seamless Playback**: Native infinite looping with crossfade transitions
- **Intelligent Generation**: Retry system with quality validation and error recovery
- **Multiple Interfaces**: Terminal, lightweight GUI, and advanced GUI with visualizations
- **Benchmark System**: Performance monitoring and generation time statistics
- **Smart Loop Management**: Configurable minimum duration before loop switching
- **Preset System**: Quick generation with pre-configured musical styles
- **Process Management**: CPU/IO priority optimization and safe termination
- **Export Functionality**: Save generated loops for later use
- **Settings Persistence**: Automatic save/restore of user preferences
- **Debug Mode**: Comprehensive logging and file state tracking

## Available Versions

### 1. Lightweight GUI Version (`il1.py`) - Most Recommended ⭐

- Clean graphical interface with tabbed organization
- Same robust audio engine as terminal version
- Real crossfade transition support with visual feedback
- **Preset System** for quick generation
- **Statistics Tab**: Benchmark data and generation time tracking
- **Settings Tab**: Configure generation duration, minimum song duration, audio driver
- **Loop Information**: Display random titles, artists, duration, and genre
- Real-time status monitoring with timing information
- Settings persistence across sessions
- Save current loop functionality

### 2. Terminal Version (`ilterm.py`) - Most Stable ⭐

- Command-line interface with full interactive mode
- Real transition support between loops  
- Advanced audio validation and error recovery
- Debug mode with detailed logging
- Generation-only mode for single loops
- Complete settings configuration
- Lowest resource consumption
- **Benchmark system** for tracking generation performance

**Interactive Commands:**
- `start '<prompt>'` - Start infinite loop
- `stop` - Stop playback
- `status` - Detailed system status with timing information
- `save <file.wav>` - Export current loop
- `set duration/minduration/driver` - Change settings including minimum song duration
- `debug on/off` - Toggle debug mode
- `validate current/next/both` - Check file integrity
- `help` - Show all commands

### 3. Advanced GUI Version (`il2.py`) - Experimental

- Full graphical interface with audio visualizations
- Real-time waveform and spectrum analysis
- Visual loop metrics display
- Configurable generation parameters
- **Note**: Does not support transition between tracks
- Higher resource usage due to visualizations

## System Requirements

### Minimum Requirements

- **Operating System**: Linux (Ubuntu 20.04+ or equivalent)
- **Python**: 3.8 or higher
- **RAM**: 8 GB system memory
- **CPU**: Multi-core processor (2+ cores recommended)
- **Audio**: Working audio subsystem (PulseAudio, ALSA, or OSS)
- **Storage**: 500 MB free space for temporary files

### Recommended Requirements

- **Operating System**: Ubuntu 22.04 LTS
- **Python**: 3.10 or higher
- **RAM**: 16 GB system memory
- **CPU**: Modern high-frequency CPU (3+ GHz, 4+ cores)
- **Audio**: PulseAudio with low-latency configuration
- **Storage**: 2 GB free space

### Performance Notes

- **MusicGPT (medium)** performs best on modern, high-frequency CPUs
- **GPU support** is experimental and not required
- **Memory allocation failures** occur with insufficient RAM (<8 GB)
- **Slow inference** happens with weak CPUs (<2 cores or <2 GHz)
- **Audio dropouts** may occur without proper audio driver configuration

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
chmod +x ilterm.py il1.py il2.py
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

### Step 5: Setup Audio System (Linux)

```bash
# Ensure audio services are running
sudo systemctl --user enable pulseaudio
sudo systemctl --user start pulseaudio

# Test audio output
speaker-test -t wav -c 2

# Optional: Configure low-latency audio
echo "default-sample-rate = 44100" >> ~/.pulse/daemon.conf
echo "alternate-sample-rate = 48000" >> ~/.pulse/daemon.conf
pulseaudio --kill && pulseaudio --start
```

## Usage note

MusicGPT will download the selected model (medium by default) on first startup. The very first generation will be significantly slower than subsequent ones. This applies to all versions of INFINI LOOP.

It is suggested to run a musicgpt test, which will download the model, before running INFINILOOP:

'''bash
./musicgpt-x86_64-unknown-linux-gnu "Create a relaxing LoFi song" --model medium
'''

This will produce a test sample, and download the required model.

### Lightweight GUI Version (`il1.py`) - Recommended

```bash
python il1.py
```

**Usage Steps:**
1. **Enter Prompt**: Type your musical description (e.g., "calm acoustic guitar")
2. **Choose Preset**: Click preset buttons for quick setups
3. **Configure Settings**: 
   - **Generation Duration**: 5-30 seconds (10-15s optimal for short loops)
   - **Minimum Song Duration**: 10-300 seconds (how long each loop plays before switching)
   - **Audio Driver**: pulse/alsa/dsp
4. **Start Generation**: Click "▶️ START LOOP" 
5. **Monitor Progress**: Watch status bar, loop information, and timing
6. **View Statistics**: Check generation time benchmarks in Statistics tab
7. **Save Loops**: Use "💾 Save Current Loop" to export
8. **Debug**: Enable debug mode in Settings for troubleshooting

**Loop Information Display:**
- **Title**: Random title from word combinations
- **Artist**: Random artist name from word combinations
- **Duration**: Actual loop length and elapsed playback time
- **Genre**: Current generation prompt

**Real-time Timing Information:**
- Shows elapsed time since current loop started
- Displays remaining time until minimum duration is satisfied
- Indicates when loop is ready to switch to next generation

### Terminal Version (`ilterm.py`)

#### Quick Start

```bash
# Interactive mode (recommended)
python ilterm.py

# Direct generation with prompt
python ilterm.py --prompt "electronic dance loop"

# Custom settings with minimum duration
python ilterm.py --prompt "ambient chill" --duration 20 --driver pulse

# Generate single loop and exit
python ilterm.py --generate-only "jazz piano" output.wav

# Debug mode
python ilterm.py --prompt "test loop" --verbose
```

#### Interactive Commands

When in interactive mode, use these commands:

```bash
🎛️ > start 'ambient electronic loop'     # Start infinite loop
🎛️ > stop                                # Stop playback
🎛️ > status                              # Show detailed status with timing info
🎛️ > save my_favorite_loop.wav          # Export current loop
🎛️ > set duration                        # Change generation length (5-30s)
🎛️ > set minduration                     # Change minimum song duration (10-300s)
🎛️ > set driver                          # Change audio driver
🎛️ > debug on                            # Enable debug logging
🎛️ > validate both                       # Check file integrity
🎛️ > help                                # Show all commands
🎛️ > quit                                # Exit program
🎛️ > set minduration                     # Configure minimum song duration
# Range: 10-300 seconds (5 minutes max)
# Current loop will play at least this long before switching
# Tip: 30-60s for variety, 120s+ for longer listening sessions
```

#### Command Line Options

```bash
python ilterm.py [OPTIONS]

Options:
  -p, --prompt TEXT        Music generation prompt
  -i, --interactive        Start in interactive mode
  -g, --generate-only PROMPT OUTPUT  Generate single loop and save
  -d, --duration INTEGER   Generation duration (5-30 seconds)
  --driver [pulse|alsa|dsp]  Audio driver selection
  -v, --verbose            Detailed output
  -q, --quiet              Minimal output
  --no-debug               Disable debug mode
  -h, --help               Show help message
```

### Advanced GUI Version (`il2.py`) - EXPERIMENTAL

```bash
python il2.py
```

1. Enter your musical prompt in the text field
2. Choose algorithm type:
   - **Advanced**: Multi-metric analysis (recommended)
   - **Classic**: Spectral similarity only
3. Select model (medium recommended) and duration
4. Click "AVVIA" to begin generation
5. Monitor real-time visualizations for loop analysis
6. Use "SALVA" button to save loops

**Note**: This version does not support crossfading between loops.

## Technical Details

### Audio Processing Pipeline

1. **AI Generation**: MusicGPT creates raw audio using text prompts
2. **Quality Validation**: Multi-stage file integrity checking
3. **Loop Analysis**: Advanced multi-metric algorithm detects optimal loop points
4. **Peak Normalization**: Simple amplitude-based standardization to 0.7 peak level
5. **Zero-Crossing Optimization**: Fine-tune loop points for seamless transitions
6. **Continuous Playback**: Native infinite looping with background generation and crossfade
7. **Minimum Duration Control**: Configurable timing before allowing loop switches

### Loop Detection Algorithm

INFINI LOOP uses a sophisticated two-stage approach:

#### Stage 1: Advanced Multi-Metric Analysis
1. **Spectral Similarity**: Mel-frequency cepstral analysis of loop boundaries
2. **Waveform Continuity**: Cross-correlation and RMS matching
3. **Beat Alignment**: Rhythm preservation using beat tracking
4. **Phase Continuity**: STFT phase coherence analysis
5. **Composite Scoring**: Weighted combination of all metrics

#### Stage 2: Beat-Focused Fallback
1. **Tempo Detection**: BPM analysis with consistency measurement
2. **Musical Structure**: Preference for 1, 2, 4, 8-measure loops
3. **Beat Grid Alignment**: Snap to detected beat positions
4. **Rhythm Preservation**: Maintain musical coherence

### Audio Normalization (Simplified)

**Peak-based normalization system:**
- **Target Peak**: 0.7 amplitude (70% of maximum)
- **Simple Scaling**: Linear gain adjustment based on current peak
- **Clipping Prevention**: Additional limiting to 0.95 if needed
- **Fallback**: RMS-based normalization for very quiet signals
- **Consistency**: Uniform loudness across generated loops

*Note: Previous LUFS normalization has been replaced with this simpler peak-based approach for better reliability.*

### Benchmark System

**Performance Tracking:**
- Records generation time for each sample duration
- Groups data by requested duration (5s, 10s, 15s, etc.)
- Calculates average generation times
- Displays statistics in GUI Statistics tab
- Stores data in `benchdata.json` file
- Provides insights for optimal duration selection

### Smart Loop Management

**Configurable Minimum Duration:**
- Default: 30 seconds minimum playback time
- Range: 10-300 seconds (5 minutes maximum)
- Prevents rapid switching between loops
- Allows appreciation of each generated piece
- Real-time timing display shows progress

**Loop Information System:**
- Random title generation from word combinations
- AI artist name selection
- Real-time duration and timing display
- Genre information from current prompt

### Process Management

- **CPU Priority**: Background generation with `nice` and `ionice`
- **CPU Affinity**: Core assignment for optimal performance
- **Memory Management**: Improved temporary file cleanup and leak prevention  
- **Safe Termination**: Graceful process shutdown with timeout handling
- **Error Recovery**: Automatic retry with exponential backoff
- **Buffer Management**: Smart audio buffer handling with proper cleanup

### Model Recommendations

- **Small Model**: Fast but often low quality - not recommended for music
- **Medium Model**: Best balance of quality and speed - **recommended**
- **Large Model**: Highest quality but very slow and resource-intensive

## Troubleshooting

### Common Issues and Solutions

#### Generation Issues

**Issue**: "File audio generato con errori dalla AI"
- **Cause**: MusicGPT binary not found or not executable
- **Solution**: 
  ```bash
  chmod +x musicgpt-x86_64-unknown-linux-gnu
  ./musicgpt-x86_64-unknown-linux-gnu --help  # Test binary
  ```

**Issue**: Very slow generation (>60 seconds)
- **Cause**: Insufficient CPU power or memory
- **Solutions**:
  - Use shorter duration (8-12 seconds)
  - Close other applications
  - Ensure 16+ GB RAM available
  - Check CPU isn't throttling due to heat
  - Check benchmark statistics to find optimal duration

**Issue**: "No interesting loop" or low quality output
- **Cause**: AI generated audio not suitable for looping
- **Solutions**:
  - Use preset buttons in GUI for proven prompts
  - Add keywords: "seamless", "loopable", "nointro"
  - Try different prompts: avoid "song", "verse", "chorus"
  - Use medium model instead of small
  - Increase generation duration to 15-20 seconds

#### Audio Playback Issues

**Issue**: No audio output or silent playback
- **Solutions**:
  ```bash
  # Test audio system
  speaker-test -t wav -c 2
  
  # Try different audio drivers in GUI Settings or terminal
  python ilterm.py --driver alsa    # or pulse, dsp
  
  # Check PulseAudio
  pulseaudio --check -v
  systemctl --user restart pulseaudio
  ```

**Issue**: Audio stuttering or dropouts
- **Cause**: Audio buffer underruns or driver issues
- **Solutions**:
  - Try different audio drivers (pulse → alsa → dsp)
  - Increase audio buffer size:
    ```bash
    echo "default-fragment-size-msec = 25" >> ~/.pulse/daemon.conf
    ```
  - Close unnecessary applications
  - Use `taskset` to assign dedicated CPU cores

**Issue**: Crackling or distorted audio
- **Cause**: Sample rate mismatch or audio driver problems
- **Solutions**:
  ```bash
  # Set consistent sample rate
  echo "default-sample-rate = 44100" >> ~/.pulse/daemon.conf
  pulseaudio --kill && pulseaudio --start
  ```

#### GUI-Specific Issues

**Issue**: Settings not saved between sessions
- **Cause**: Permission issues or missing configuration directory
- **Solution**: Check that `infiniloop_settings.json` can be created in the application directory

**Issue**: Random titles showing as "UNTITLED"
- **Cause**: Missing optional enhancement files
- **Solution**: Create `nomi.txt`, `nomi2.txt`, and `artisti.txt` files as described in installation

**Issue**: Statistics tab empty
- **Cause**: Benchmark data not yet collected or disabled
- **Solution**: Enable benchmark in Settings and generate a few loops to populate data

### Debug Mode

Enable comprehensive logging for troubleshooting:

```bash
# Terminal version
python ilterm.py --prompt "test loop" --verbose

# Interactive mode
python ilterm.py
🎛️ > debug on
🎛️ > start 'test ambient'

# GUI version: Settings tab → Enable debug mode
```

Debug output includes:
- File state tracking (creation, validation, deletion)
- Audio processing pipeline steps
- Loop detection algorithm progress
- Process creation and termination
- Memory and CPU usage
- Timing and benchmark data
- Error stack traces

### Performance Optimization

#### For Low-End Systems:
```bash
# Reduce generation duration
python ilterm.py --duration 8

# Set shorter minimum duration for quicker variety
🎛️ > set minduration  # Set to 20-30 seconds

# Use ALSA for lower latency
python ilterm.py --driver alsa

# Lower process priority
nice -n 10 python ilterm.py --prompt "ambient"
```

#### For High-End Systems:
```bash
# Use longer durations for better quality
python ilterm.py --duration 20

# Set longer minimum duration for extended listening
🎛️ > set minduration  # Set to 120-180 seconds

# Dedicated CPU cores
taskset -c 2,3 python ilterm.py --prompt "complex orchestral"
```

### Log Analysis

Common log patterns and meanings:

```
✅ Perfect loop found?                    # Loop detection succeeded
❌ No interesting loop                    # AI output not suitable for looping
🎚️ Peak norm: X → 0.7                   # Peak normalization applied
🎯 Zero-crossing optimization...         # Fine-tuning loop boundaries
🔄 Reinitializing...                     # Recovering from error
⚠️ Zero-crossing rejected               # Optimization would break rhythm
📈 Benchmark stats updated               # Performance data recorded
⏱️ Current loop: Xs, Ys left            # Minimum duration timing
✅ Song ended, waiting next loop         # Ready to switch loops
```

## Advanced Configuration

### Custom Audio Settings

Create `~/.pulse/daemon.conf` for optimal audio:

```bash
# Low-latency configuration
default-sample-rate = 44100
alternate-sample-rate = 48000
default-sample-channels = 2
default-channel-map = front-left,front-right
default-fragments = 4
default-fragment-size-msec = 25
enable-remixing = no
enable-lfe-remixing = no
high-priority = yes
nice-level = -11
realtime-scheduling = yes
realtime-priority = 5
```

### Environment Variables

```bash
# Audio driver selection
export SDL_AUDIODRIVER=pulse          # or alsa, dsp

# Audio device selection  
export PULSE_DEVICE=alsa_output.pci-0000_00_1b.0.analog-stereo

# Disable debug outputs
export PYGAME_HIDE_SUPPORT_PROMPT=1

# Memory optimization
export OMP_NUM_THREADS=4
```

### Custom Preset Creation

You can modify the presets in `il1.py` by editing the presets dictionary:

```python
presets = {
    "Custom Jazz": "smooth jazz piano trio seamless loop",
    "Epic Cinema": "cinematic orchestral epic soundtrack loop",
    "Nature Ambient": "forest sounds ambient nature seamless loop",
    # Add your own presets here
}
```

## License

This project is released under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0). 

**Permissions:**
- ✅ Share and adapt the material
- ✅ Use for personal and educational purposes
- ✅ Modify and build upon the code

**Restrictions:**
- ❌ Commercial use without permission
- ❌ Distribution without attribution

For commercial licensing, please contact the authors.

## Credits

- **MusicGPT** by gabotechs for AI music generation: https://github.com/gabotechs/MusicGPT
- **librosa** team for audio analysis: https://librosa.org/
- **pyloudnorm** by csteinmetz1 for audio processing: https://github.com/csteinmetz1/pyloudnorm
- Developed with assistance from AI language models
- Audio processing powered by librosa, soundfile, and scipy
- GUI implementation using tkinter and matplotlib

## Changelog

### Latest Version
- ➕ **Simplified Peak Normalization** replacing complex LUFS system
- ➕ **Complete GUI System** with tabbed interface and presets
- ➕ **Benchmark/Statistics System** for performance monitoring
- ➕ **Configurable Minimum Song Duration** with real-time timing
- ➕ **Preset System** with 6 musical style presets
- ➕ **Settings Persistence** - automatic save/restore configuration
- ➕ **Smart Loop Information** - random titles, artists, timing display
- ➕ **Enhanced Memory Management** - improved cleanup and resource handling
- ➕ **Advanced Status Monitoring** - real-time progress and timing information
- ➕ **Optional Enhancement Files** for custom titles and artists
- 🔧 **Improved Loop Detection** with better beat alignment fallback
- 🔧 **Better Error Recovery** with comprehensive validation
- 🔧 **Enhanced Debug System** with detailed timing and benchmark logging
- 🐛 **Fixed Audio Buffer Management** preventing memory leaks
- 🐛 **Resolved Timing Issues** in loop switching logic
- 🐛 **Improved File Validation** preventing corrupted audio playback

---
**🎵 Enjoy infinite AI music with INFINI LOOP! 🎵**
---

# INFINI LOOP - Generazione Musicale AI Locale Infinita

INFINI LOOP è un sistema musicale alimentato da AI progettato per generare loop audio continui e senza interruzioni.
Crea automaticamente nuovi frammenti musicali utilizzando l'AI, rileva i migliori punti di loop e li riproduce continuamente preparando quello successivo — risultando in un flusso fluido, infinito e sempre nuovo di musica strumentale.

All'avvio, verrà riprodotto uno dei due file .wav pre-inclusi, così potrai goderti la musica immediatamente mentre viene preparata la prima generazione AI.

Una volta configurato e in funzione, la tua macchina diventa una stazione musicale AI locale, producendo continuamente nuove tracce con transizioni. Locale, privata, più personale di qualsiasi playlist YouTube o Spotify.

## Indice

- [Caratteristiche](#caratteristiche)
- [Versioni Disponibili](#versioni-disponibili)
- [Requisiti di Sistema](#requisiti-di-sistema)
- [Installazione](#installazione)
- [Utilizzo](#utilizzo)
- [Dettagli Tecnici](#dettagli-tecnici)
- [Risoluzione Problemi](#risoluzione-problemi)
- [Configurazione Avanzata](#configurazione-avanzata)
- [Sviluppo e Contributi](#sviluppo-e-contributi)
- [Licenza](#licenza)

## Caratteristiche

- **Generazione Musicale AI Locale**: Alimentato da MusicGPT per sintesi audio di alta qualità (https://github.com/gabotechs/MusicGPT)
- **Rilevamento Loop Avanzato**: Analisi multi-metrica con algoritmi adattivi
  - Analisi similarità spettrale
  - Misurazione continuità forma d'onda
  - Allineamento beat e preservazione ritmo
  - Ottimizzazione coerenza di fase
  - Affinamento zero-crossing
- **Riproduzione Senza Interruzioni**: Loop infinito nativo con transizioni crossfade
- **Generazione Intelligente**: Sistema retry con validazione qualità e recupero errori
- **Interfacce Multiple**: Terminale, GUI leggera e GUI avanzata con visualizzazioni
- **Sistema Benchmark**: Monitoraggio performance e statistiche tempi generazione
- **Gestione Loop Intelligente**: Durata minima configurabile prima del cambio loop
- **Sistema Preset**: Generazione rapida con stili musicali pre-configurati
- **Gestione Processi**: Ottimizzazione priorità CPU/IO e terminazione sicura
- **Funzionalità Esportazione**: Salva loop generati per uso successivo
- **Persistenza Impostazioni**: Salvataggio/ripristino automatico delle preferenze utente
- **Modalità Debug**: Logging completo e tracking stato file

## Versioni Disponibili

### 1. Versione GUI Leggera (`il1.py`) - Più Raccomandata ⭐

- Interfaccia grafica pulita con organizzazione a schede
- Stesso motore audio robusto della versione terminale
- Supporto transizioni crossfade reali con feedback visivo
- **Sistema Preset** per generazione rapida
- **Scheda Statistiche**: Dati benchmark e tracking tempi generazione
- **Scheda Impostazioni**: Configura durata generazione, durata minima brano, driver audio
- **Informazioni Loop**: Visualizza titoli casuali, artisti, durata e genere
- Monitoraggio stato real-time con informazioni timing
- Persistenza impostazioni tra sessioni
- Funzionalità salvataggio loop corrente

### 2. Versione Terminale (`ilterm.py`) - Più Stabile ⭐

- Interfaccia command-line con modalità interattiva completa
- Supporto transizioni reali tra loop
- Validazione audio avanzata e recupero errori
- Modalità debug con logging dettagliato
- Modalità solo-generazione per singoli loop
- Configurazione completa impostazioni
- Consumo risorse più basso
- **Sistema benchmark** per tracking performance generazione

**Comandi Interattivi:**
- `start '<prompt>'` - Avvia loop infinito
- `stop` - Ferma riproduzione
- `status` - Stato dettagliato sistema con info timing
- `save <file.wav>` - Esporta loop corrente
- `set duration/minduration/driver` - Cambia impostazioni inclusa durata minima brano
- `debug on/off` - Attiva/disattiva modalità debug
- `validate current/next/both` - Verifica integrità file
- `help` - Mostra tutti i comandi

### 3. Versione GUI Avanzata (`il2.py`) - Sperimentale

- Interfaccia grafica completa con visualizzazioni audio
- Analisi forma d'onda e spettro real-time
- Visualizzazione metriche loop
- Parametri generazione configurabili
- **Nota**: Non supporta transizione tra tracce
- Utilizzo risorse più alto a causa delle visualizzazioni

## Requisiti di Sistema

### Requisiti Minimi

- **Sistema Operativo**: Linux (Ubuntu 20.04+ o equivalente)
- **Python**: 3.8 o superiore
- **RAM**: 8 GB memoria sistema
- **CPU**: Processore multi-core (2+ core raccomandati)
- **Audio**: Sottosistema audio funzionante (PulseAudio, ALSA, o OSS)
- **Archiviazione**: 500 MB spazio libero per file temporanei

### Requisiti Raccomandati

- **Sistema Operativo**: Ubuntu 22.04 LTS
- **Python**: 3.10 o superiore
- **RAM**: 16 GB memoria sistema
- **CPU**: CPU moderna alta frequenza (3+ GHz, 4+ core)
- **Audio**: PulseAudio con configurazione bassa latenza
- **Archiviazione**: 2 GB spazio libero

### Note Performance

- **MusicGPT (medium)** performa meglio su CPU moderne ad alta frequenza
- **Supporto GPU** è sperimentale e non richiesto
- **Errori allocazione memoria** si verificano con RAM insufficiente (<8 GB)
- **Inferenza lenta** accade con CPU deboli (<2 core o <2 GHz)
- **Interruzioni audio** possono verificarsi senza configurazione driver audio appropriata

## Installazione

### Passo 1: Installa Dipendenze Sistema

```bash
# Per Ubuntu/Debian:
sudo apt update
sudo apt install -y ffmpeg pulseaudio-utils alsa-utils python3-pip python3-dev \
                    portaudio19-dev python3-tk build-essential libasound2-dev \
                    libportaudio2 libportaudiocpp0 portaudio19-dev
```

### Passo 2: Installa Dipendenze Python

```bash
# Installa librerie processing audio core
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

# Rendi script eseguibili
chmod +x ilterm.py il1.py il2.py
```

### Passo 4: Scarica e Configura Binario MusicGPT

1. Visita la [pagina release MusicGPT](https://github.com/gabotechs/MusicGPT/releases)
2. Scarica `musicgpt-x86_64-unknown-linux-gnu` (ultima versione)
3. Posizionalo nella stessa directory degli script Python
4. Rendilo eseguibile:

```bash
chmod +x musicgpt-x86_64-unknown-linux-gnu

# Verifica che il binario funzioni
./musicgpt-x86_64-unknown-linux-gnu --help
```

### Passo 5: Configura Sistema Audio (Linux)

```bash
# Assicurati che i servizi audio siano in esecuzione
sudo systemctl --user enable pulseaudio
sudo systemctl --user start pulseaudio

# Testa output audio
speaker-test -t wav -c 2

# Opzionale: Configura audio bassa latenza
echo "default-sample-rate = 44100" >> ~/.pulse/daemon.conf
echo "alternate-sample-rate = 48000" >> ~/.pulse/daemon.conf
pulseaudio --kill && pulseaudio --start
```

## Nota Utilizzo

MusicGPT scaricherà il modello selezionato (medium di default) al primo avvio. La primissima generazione sarà significativamente più lenta delle successive. Questo si applica a tutte le versioni di INFINI LOOP.

Si suggerisce di eseguire un test musicgpt, che scaricherà il modello, prima di eseguire INFINILOOP:

```bash
./musicgpt-x86_64-unknown-linux-gnu "Create a relaxing LoFi song" --model medium
```

Questo produrrà un campione di test e scaricherà il modello richiesto.

### Versione GUI Leggera (`il1.py`) - Raccomandata

```bash
python il1.py
```

**Passi Utilizzo:**
1. **Inserisci Prompt**: Digita la tua descrizione musicale (es. "chitarra acustica calma")
2. **Scegli Preset**: Clicca pulsanti preset per configurazioni rapide
3. **Configura Impostazioni**: 
   - **Durata Generazione**: 5-30 secondi (10-15s ottimale per loop brevi)
   - **Durata Minima Brano**: 10-300 secondi (quanto a lungo ogni loop suona prima di cambiare)
   - **Driver Audio**: pulse/alsa/dsp
4. **Avvia Generazione**: Clicca "▶️ AVVIA LOOP" 
5. **Monitora Progresso**: Guarda barra stato, informazioni loop e timing
6. **Visualizza Statistiche**: Controlla benchmark tempi generazione nella scheda Statistiche
7. **Salva Loop**: Usa "💾 Salva Loop Corrente" per esportare
8. **Debug**: Abilita modalità debug in Impostazioni per troubleshooting

**Visualizzazione Informazioni Loop:**
- **Titolo**: Titolo casuale da combinazioni parole
- **Artista**: Nome artista casuale da combinazioni parole
- **Durata**: Lunghezza loop effettiva e tempo riproduzione trascorso
- **Genere**: Prompt generazione corrente

**Informazioni Timing Real-time:**
- Mostra tempo trascorso da inizio loop corrente
- Visualizza tempo rimanente fino soddisfazione durata minima
- Indica quando loop è pronto per passare alla prossima generazione

### Versione Terminale (`ilterm.py`)

#### Avvio Rapido

```bash
# Modalità interattiva (raccomandata)
python ilterm.py

# Generazione diretta con prompt
python ilterm.py --prompt "electronic dance loop"

# Impostazioni personalizzate con durata minima
python ilterm.py --prompt "ambient chill" --duration 20 --driver pulse

# Genera singolo loop ed esci
python ilterm.py --generate-only "jazz piano" output.wav

# Modalità debug
python ilterm.py --prompt "test loop" --verbose
```

#### Comandi Interattivi

Quando in modalità interattiva, usa questi comandi:

```bash
🎛️ > start 'ambient electronic loop'     # Avvia loop infinito
🎛️ > stop                                # Ferma riproduzione
🎛️ > status                              # Mostra stato dettagliato con info timing
🎛️ > save my_favorite_loop.wav          # Esporta loop corrente
🎛️ > set duration                        # Cambia lunghezza generazione (5-30s)
🎛️ > set minduration                     # Cambia durata minima brano (10-300s)
🎛️ > set driver                          # Cambia driver audio
🎛️ > debug on                            # Abilita logging debug
🎛️ > validate both                       # Verifica integrità file
🎛️ > help                                # Mostra tutti i comandi
🎛️ > quit                                # Esci programma
🎛️ > set minduration                     # Configura durata minima brano
# Range: 10-300 secondi (5 minuti max)
# Il loop corrente suonerà almeno questo tempo prima di cambiare
# Suggerimento: 30-60s per varietà, 120s+ per sessioni ascolto più lunghe
```

#### Opzioni Command Line

```bash
python ilterm.py [OPZIONI]

Opzioni:
  -p, --prompt TEXT        Prompt generazione musicale
  -i, --interactive        Avvia in modalità interattiva
  -g, --generate-only PROMPT OUTPUT  Genera singolo loop e salva
  -d, --duration INTEGER   Durata generazione (5-30 secondi)
  --driver [pulse|alsa|dsp]  Selezione driver audio
  -v, --verbose            Output dettagliato
  -q, --quiet              Output minimale
  --no-debug               Disabilita modalità debug
  -h, --help               Mostra messaggio aiuto
```

### Versione GUI Avanzata (`il2.py`) - SPERIMENTALE

```bash
python il2.py
```

1. Inserisci il tuo prompt musicale nel campo testo
2. Scegli tipo algoritmo:
   - **Avanzato**: Analisi multi-metrica (raccomandato)
   - **Classico**: Solo similarità spettrale
3. Seleziona modello (medium raccomandato) e durata
4. Clicca "AVVIA" per iniziare generazione
5. Monitora visualizzazioni real-time per analisi loop
6. Usa pulsante "SALVA" per salvare loop

**Nota**: Questa versione non supporta crossfading tra loop.

## Dettagli Tecnici

### Pipeline Processing Audio

1. **Generazione AI**: MusicGPT crea audio grezzo usando prompt testuali
2. **Validazione Qualità**: Controllo integrità file multi-stadio
3. **Analisi Loop**: Algoritmo multi-metrico avanzato rileva punti loop ottimali
4. **Normalizzazione Picco**: Standardizzazione semplice basata su ampiezza a livello picco 0.7
5. **Ottimizzazione Zero-Crossing**: Affina punti loop per transizioni senza interruzioni
6. **Riproduzione Continua**: Loop infinito nativo con generazione background e crossfade
7. **Controllo Durata Minima**: Timing configurabile prima di permettere cambi loop

### Algoritmo Rilevamento Loop

INFINI LOOP usa un approccio sofisticato a due stadi:

#### Stadio 1: Analisi Multi-Metrica Avanzata
1. **Similarità Spettrale**: Analisi mel-frequency cepstral dei confini loop
2. **Continuità Forma d'Onda**: Cross-correlazione e matching RMS
3. **Allineamento Beat**: Preservazione ritmo usando beat tracking
4. **Continuità Fase**: Analisi coerenza fase STFT
5. **Scoring Composito**: Combinazione pesata di tutte le metriche

#### Stadio 2: Fallback Focalizzato su Beat
1. **Rilevamento Tempo**: Analisi BPM con misurazione consistenza
2. **Struttura Musicale**: Preferenza per loop 1, 2, 4, 8 misure
3. **Allineamento Griglia Beat**: Aggancio a posizioni beat rilevate
4. **Preservazione Ritmo**: Mantiene coerenza musicale

### Normalizzazione Audio (Semplificata)

**Sistema normalizzazione basato su picco:**
- **Picco Target**: 0.7 ampiezza (70% del massimo)
- **Scaling Semplice**: Regolazione gain lineare basata su picco corrente
- **Prevenzione Clipping**: Limiting aggiuntivo a 0.95 se necessario
- **Fallback**: Normalizzazione basata su RMS per segnali molto quieti
- **Consistenza**: Volume uniforme attraverso loop generati

*Nota: La precedente normalizzazione LUFS è stata sostituita con questo approccio più semplice basato su picco per migliore affidabilità.*

### Sistema Benchmark

**Tracking Performance:**
- Registra tempo generazione per ogni durata campione
- Raggruppa dati per durata richiesta (5s, 10s, 15s, etc.)
- Calcola tempi generazione medi
- Visualizza statistiche nella scheda Statistiche GUI
- Memorizza dati in file `benchdata.json`
- Fornisce insights per selezione durata ottimale

### Gestione Loop Intelligente

**Durata Minima Configurabile:**
- Default: 30 secondi tempo riproduzione minimo
- Range: 10-300 secondi (5 minuti massimo)
- Previene cambio rapido tra loop
- Permette apprezzamento di ogni pezzo generato
- Visualizzazione timing real-time mostra progresso

**Sistema Informazioni Loop:**
- Generazione titolo casuale da combinazioni parole
- Selezione nome artista AI
- Visualizzazione durata e timing real-time
- Informazioni genere da prompt corrente

### Gestione Processi

- **Priorità CPU**: Generazione background con `nice` e `ionice`
- **Affinità CPU**: Assegnamento core per performance ottimale
- **Gestione Memoria**: Migliorato cleanup file temporanei e prevenzione leak
- **Terminazione Sicura**: Spegnimento processo graduale con gestione timeout
- **Recupero Errori**: Retry automatico con exponential backoff
- **Gestione Buffer**: Gestione buffer audio intelligente con cleanup appropriato

### Raccomandazioni Modello

- **Modello Small**: Veloce ma spesso qualità bassa - non raccomandato per musica
- **Modello Medium**: Miglior equilibrio tra qualità e velocità - **raccomandato**
- **Modello Large**: Qualità più alta ma molto lento e intensivo di risorse

## Risoluzione Problemi

### Problemi Comuni e Soluzioni

#### Problemi Generazione

**Problema**: "File audio generato con errori dalla AI"
- **Causa**: Binario MusicGPT non trovato o non eseguibile
- **Soluzione**: 
  ```bash
  chmod +x musicgpt-x86_64-unknown-linux-gnu
  ./musicgpt-x86_64-unknown-linux-gnu --help  # Testa binario
  ```

**Problema**: Generazione molto lenta (>60 secondi)
- **Causa**: Potenza CPU o memoria insufficiente
- **Soluzioni**:
  - Usa durata più breve (8-12 secondi)
  - Chiudi altre applicazioni
  - Assicurati 16+ GB RAM disponibili
  - Verifica CPU non throttling per calore
  - Controlla statistiche benchmark per trovare durata ottimale

**Problema**: "No interesting loop" o output bassa qualità
- **Causa**: Audio generato AI non adatto per looping
- **Soluzioni**:
  - Usa pulsanti preset in GUI per prompt provati
  - Aggiungi parole chiave: "seamless", "loopable", "nointro"
  - Prova prompt diversi: evita "song", "verse", "chorus"
  - Usa modello medium invece di small
  - Aumenta durata generazione a 15-20 secondi

#### Problemi Riproduzione Audio

**Problema**: Nessun output audio o riproduzione silenziosa
- **Soluzioni**:
  ```bash
  # Testa sistema audio
  speaker-test -t wav -c 2
  
  # Prova driver audio diversi in Impostazioni GUI o terminale
  python ilterm.py --driver alsa    # o pulse, dsp
  
  # Controlla PulseAudio
  pulseaudio --check -v
  systemctl --user restart pulseaudio
  ```

**Problema**: Balbettio audio o interruzioni
- **Causa**: Underrun buffer audio o problemi driver
- **Soluzioni**:
  - Prova driver audio diversi (pulse → alsa → dsp)
  - Aumenta dimensione buffer audio:
    ```bash
    echo "default-fragment-size-msec = 25" >> ~/.pulse/daemon.conf
    ```
  - Chiudi applicazioni non necessarie
  - Usa `taskset` per assegnare core CPU dedicati

**Problema**: Audio crepitante o distorto
- **Causa**: Mismatch sample rate o problemi driver audio
- **Soluzioni**:
  ```bash
  # Imposta sample rate consistente
  echo "default-sample-rate = 44100" >> ~/.pulse/daemon.conf
  pulseaudio --kill && pulseaudio --start
  ```

#### Problemi Specifici GUI

**Problema**: Impostazioni non salvate tra sessioni
- **Causa**: Problemi permessi o directory configurazione mancante
- **Soluzione**: Verifica che `infiniloop_settings.json` possa essere creato nella directory applicazione

**Problema**: Titoli casuali mostrano come "UNTITLED"
- **Causa**: File miglioramento opzionali mancanti
- **Soluzione**: Crea file `nomi.txt`, `nomi2.txt`, e `artisti.txt` come descritto nell'installazione

**Problema**: Scheda statistiche vuota
- **Causa**: Dati benchmark non ancora raccolti o disabilitati
- **Soluzione**: Abilita benchmark in Impostazioni e genera alcuni loop per popolare dati

### Modalità Debug

Abilita logging completo per troubleshooting:

```bash
# Versione terminale
python ilterm.py --prompt "test loop" --verbose

# Modalità interattiva
python ilterm.py
🎛️ > debug on
🎛️ > start 'test ambient'

# Versione GUI: scheda Impostazioni → Abilita modalità debug
```

Output debug include:
- Tracking stato file (creazione, validazione, cancellazione)
- Passi pipeline processing audio
- Progresso algoritmo rilevamento loop
- Creazione e terminazione processi
- Utilizzo memoria e CPU
- Timing e dati benchmark
- Stack trace errori

### Ottimizzazione Performance

#### Per Sistemi Low-End:
```bash
# Riduci durata generazione
python ilterm.py --duration 8

# Imposta durata minima più breve per varietà più rapida
🎛️ > set minduration  # Imposta a 20-30 secondi

# Usa ALSA per latenza più bassa
python ilterm.py --driver alsa

# Priorità processo più bassa
nice -n 10 python ilterm.py --prompt "ambient"
```

#### Per Sistemi High-End:
```bash
# Usa durate più lunghe per qualità migliore
python ilterm.py --duration 20

# Imposta durata minima più lunga per ascolto esteso
🎛️ > set minduration  # Imposta a 120-180 secondi

# Core CPU dedicati
taskset -c 2,3 python ilterm.py --prompt "complex orchestral"
```

### Analisi Log

Pattern log comuni e significati:

```
✅ Perfect loop found?                    # Rilevamento loop riuscito
❌ No interesting loop                    # Output AI non adatto per looping
🎚️ Peak norm: X → 0.7                   # Normalizzazione picco applicata
🎯 Zero-crossing optimization...         # Affinamento confini loop
🔄 Reinitializing...                     # Recupero da errore
⚠️ Zero-crossing rejected               # Ottimizzazione romperebbe ritmo
📈 Benchmark stats updated               # Dati performance registrati
⏱️ Current loop: Xs, Ys left            # Timing durata minima
✅ Song ended, waiting next loop         # Pronto per cambiare loop
```

## Configurazione Avanzata

### Impostazioni Audio Personalizzate

Crea `~/.pulse/daemon.conf` per audio ottimale:

```bash
# Configurazione bassa latenza
default-sample-rate = 44100
alternate-sample-rate = 48000
default-sample-channels = 2
default-channel-map = front-left,front-right
default-fragments = 4
default-fragment-size-msec = 25
enable-remixing = no
enable-lfe-remixing = no
high-priority = yes
nice-level = -11
realtime-scheduling = yes
realtime-priority = 5
```

### Variabili Ambiente

```bash
# Selezione driver audio
export SDL_AUDIODRIVER=pulse          # o alsa, dsp

# Selezione dispositivo audio
export PULSE_DEVICE=alsa_output.pci-0000_00_1b.0.analog-stereo

# Disabilita output debug
export PYGAME_HIDE_SUPPORT_PROMPT=1

# Ottimizzazione memoria
export OMP_NUM_THREADS=4
```

### Creazione Preset Personalizzati

Puoi modificare i preset in `il1.py` editando il dizionario presets:

```python
presets = {
    "Jazz Personalizzato": "smooth jazz piano trio seamless loop",
    "Cinema Epico": "cinematic orchestral epic soundtrack loop",
    "Ambient Natura": "forest sounds ambient nature seamless loop",
    # Aggiungi i tuoi preset qui
}
```

## Licenza

Questo progetto è rilasciato sotto la Licenza Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).

**Permessi:**
- ✅ Condividi e adatta il materiale
- ✅ Usa per scopi personali ed educativi
- ✅ Modifica e costruisci sul codice

**Restrizioni:**
- ❌ Uso commerciale senza permesso
- ❌ Distribuzione senza attribuzione

Per licenze commerciali, contatta gli autori.

## Crediti

- **MusicGPT** di gabotechs per generazione musicale AI: https://github.com/gabotechs/MusicGPT
- Team **librosa** per analisi audio: https://librosa.org/
- **pyloudnorm** di csteinmetz1 per processing audio: https://github.com/csteinmetz1/pyloudnorm
- Sviluppato con assistenza di modelli linguaggio AI
- Processing audio alimentato da librosa, soundfile e scipy
- Implementazione GUI usando tkinter e matplotlib

## Changelog

### Ultima Versione
- ➕ **Normalizzazione Picco Semplificata** sostituendo sistema LUFS complesso
- ➕ **Sistema GUI Completo** con interfaccia a schede e preset
- ➕ **Sistema Benchmark/Statistiche** per monitoraggio performance
- ➕ **Durata Minima Brano Configurabile** con timing real-time
- ➕ **Sistema Preset** con 6 preset stili musicali
- ➕ **Persistenza Impostazioni** - salvataggio/ripristino automatico configurazione
- ➕ **Informazioni Loop Intelligenti** - titoli casuali, artisti, visualizzazione timing
- ➕ **Gestione Memoria Migliorata** - cleanup migliorato e gestione risorse
- ➕ **Monitoraggio Stato Avanzato** - progresso real-time e informazioni timing
- ➕ **File Miglioramento Opzionali** per titoli e artisti personalizzati
- 🔧 **Rilevamento Loop Migliorato** con migliore fallback allineamento beat
- 🔧 **Migliore Recupero Errori** con validazione completa
- 🔧 **Sistema Debug Migliorato** con timing dettagliato e logging benchmark
- 🐛 **Risolto Gestione Buffer Audio** prevenendo memory leak
- 🐛 **Risolti Problemi Timing** nella logica cambio loop
- 🐛 **Migliorata Validazione File** prevenendo riproduzione audio corrotto

---

**🎵 Goditi musica AI infinita con INFINI LOOP! 🎵**
