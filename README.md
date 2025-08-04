# INFINI LOOP - Infinite Local AI Music Generation

INFINI LOOP is an AI-powered music system designed to generate seamless, infinite audio loops.
It automatically creates new musical fragments using AI, detects the best loop points, and plays them continuously while preparing the next one — resulting in a smooth, never-ending, always new stream of instrumental music.

At startup, one of two pre-included .wav files will play, so you can enjoy music immediately while the first AI generation is being prepared.

Once set up and running, your machine becomes a local AI music station, continuously producing new tracks with smooth transitions and automatic loop detection. Local, private, more personal than any YouTube or Spotify playlist.

**NEW IN THIS VERSION:**
- **Normalization**: Professional standardization
- **Enhanced Loop Detection**: Improved multi-metric analysis with beat-focused fallback
- **Zero-Crossing Optimization**: Precise loop point refinement for seamless transitions
- **Robust Error Handling**: Advanced file validation and crash recovery
- **Smart Generation**: Adaptive retry system with quality validation
- **Performance Optimizations**: Better memory management and process handling
- **Debug Mode**: Comprehensive logging for troubleshooting

Advanced GUI version (**experimental, NOT recomended**):

<img width="1234" height="679" alt="immagine" src="https://github.com/user-attachments/assets/19390959-ba49-476e-8b26-df606c6dad36" />

Lightweight GUI version (**MOST recommended**):

<img width="672" height="864" alt="immagine" src="https://github.com/user-attachments/assets/3f99ec86-661b-4a5a-8f0a-f97731340f84" />

Terminal version (**recommended**):

<img width="849" height="608" alt="immagine" src="https://github.com/user-attachments/assets/9a95d2dd-8690-4d00-8735-530511ef9498" />

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
- **Seamless Playback**: Native infinite looping with transitions
- **Intelligent Generation**: Retry system with quality validation and error recovery
- **Multiple Interfaces**: Terminal, lightweight GUI, and advanced GUI with visualizations
- **Process Management**: CPU/IO priority optimization and safe termination
- **Export Functionality**: Save generated loops for later use
- **Debug Mode**: Comprehensive logging and file state tracking

## Available Versions

### 1. Terminal Version (`ilterm.py`) - Most Stable ⭐

- Command-line interface with full interactive mode
- Real transition support between loops  
- Advanced audio validation and error recovery
- Debug mode with detailed logging
- Generation-only mode for single loops
- Complete settings configuration
- Lowest resource consumption

**Interactive Commands:**
- `start '<prompt>'` - Start infinite loop
- `stop` - Stop playback
- `status` - Detailed system status
- `save <file.wav>` - Export current loop
- `set duration/driver` - Change settings
- `debug on/off` - Toggle debug mode
- `validate current/next/both` - Check file integrity
- `help` - Show all commands

### 2. Lightweight GUI Version (`il1.py`) - Most Recommended ⭐

- Clean graphical interface with essential controls
- Same robust audio engine as terminal version
- Real transition support with visual feedback
- Preset system for quick generation
- Real-time status monitoring
- Settings persistence
- Loop information display (title, artist, duration)
- Save/load configuration

### 3. Advanced GUI Version (`il2.py`) - Experimental

- Full graphical interface with audio visualizations
- Real-time waveform and spectrum analysis
- Visual loop metrics display
- Configurable generation parameters
- **Note**: Does not support crossfading between tracks
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

## Usage

**Note**: MusicGPT will download the selected model (medium by default) on first startup. The very first generation will be significantly slower than subsequent ones. This applies to all versions of INFINI LOOP.

### Terminal Version (`ilterm.py`)

#### Quick Start

```bash
# Interactive mode (recommended)
python ilterm.py

# Direct generation with prompt
python ilterm.py --prompt "electronic dance loop"

# Custom settings
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
🎛️ > status                              # Show detailed status
🎛️ > save my_favorite_loop.wav          # Export current loop
🎛️ > set duration                        # Change generation length (5-30s)
🎛️ > set driver                          # Change audio driver
🎛️ > debug on                            # Enable debug logging
🎛️ > validate both                       # Check file integrity
🎛️ > help                                # Show all commands
🎛️ > quit                                # Exit program
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

### Lightweight GUI Version (`il1.py`)

```bash
python il1.py
```

**Usage Steps:**
1. **Enter Prompt**: Type your musical description (e.g., "calm acoustic guitar")
2. **Choose Preset**: Click preset buttons for quick setups
3. **Adjust Duration**: Use settings tab to change generation length (10-15s optimal)
4. **Start Generation**: Click "▶️ START LOOP" 
5. **Monitor Progress**: Watch status bar and log tab
6. **Save Loops**: Use "💾 Save Current Loop" to export
7. **Settings**: Configure duration, audio driver, debug mode

**Available Presets:**
- **Ambient**: Ethereal soundscapes
- **Reggae**: Classic reggae rhythms  
- **Electronic**: Synth and dance beats
- **Classical**: Orchestral arrangements
- **Rock**: Guitar-driven loops
- **Lofi Rap**: Melodic hip-hop beats

### Advanced GUI Version (`il2.py`)

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
4. **Audio Normalization**: LUFS standardization to -14 dB (broadcast standard)
5. **Zero-Crossing Optimization**: Fine-tune loop points for seamless transitions
6. **Continuous Playback**: Native infinite looping with background generation

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

### Audio Normalization

Professional LUFS (Loudness Units relative to Full Scale) normalization:
- **Target**: -14 LUFS (Spotify/YouTube standard)
- **Peak Limiting**: Prevent clipping above -0.1 dBFS
- **Dynamic Range**: Preserve musical dynamics
- **Consistency**: Uniform loudness across all generated loops

### Process Management

- **CPU Priority**: Background generation with `nice` and `ionice`
- **CPU Affinity**: Core assignment for optimal performance
- **Memory Management**: Temporary file cleanup and leak prevention  
- **Safe Termination**: Graceful process shutdown with timeout handling
- **Error Recovery**: Automatic retry with exponential backoff

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

**Issue**: "No interesting loop" or low quality output
- **Cause**: AI generated audio not suitable for looping
- **Solutions**:
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
  
  # Try different audio drivers
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
- Error stack traces

### Performance Optimization

#### For Low-End Systems:
```bash
# Reduce generation duration
python ilterm.py --duration 8

# Use ALSA for lower latency
python ilterm.py --driver alsa

# Lower process priority
nice -n 10 python ilterm.py --prompt "ambient"
```

#### For High-End Systems:
```bash
# Use longer durations for better quality
python ilterm.py --duration 20

# Dedicated CPU cores
taskset -c 2,3 python ilterm.py --prompt "complex orchestral"
```

### Log Analysis

Common log patterns and meanings:

```
✅ Perfect loop found?                    # Loop detection succeeded
❌ No interesting loop                    # AI output not suitable for looping
🎚️ Normalized from X to -14 LUFS        # Audio loudness standardization
🎯 Zero-crossing optimization...         # Fine-tuning loop boundaries
🔄 Reinitializing...                     # Recovering from error
⚠️ Zero-crossing rejected               # Optimization would break rhythm
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

## Development and Contributing

### Code Structure

- `ilterm.py`: Core engine with terminal interface
- `il1.py`: Lightweight GUI wrapper around core engine  
- `il2.py`: Advanced GUI with visualizations (experimental)

### Key Classes and Methods

**InfiniLoopTerminal** (main engine):
- `find_perfect_loop_advanced()`: Multi-metric loop detection
- `find_perfect_loop_simple()`: Beat-focused fallback algorithm
- `generate_audio_safe()`: AI generation with LUFS normalization
- `process_loop_detection()`: Complete loop analysis pipeline
- `safe_file_swap()`: Thread-safe file management
- `validate_audio_file()`: Multi-stage file validation

### Bug Reports and Feature Requests

When reporting issues, please include:

- **Version**: Which file (ilterm.py, il1.py, il2.py)
- **System**: OS, Python version, RAM, CPU details
- **Audio**: Driver used, audio hardware
- **Logs**: Complete error messages with debug mode enabled
- **Reproduction**: Exact steps to reproduce the issue
- **Prompt**: The generation prompt that caused issues

### Testing

```bash
# Test basic functionality
python ilterm.py --generate-only "test ambient" test_output.wav

# Test audio system
python ilterm.py --prompt "short test" --duration 5 --verbose

# Test all drivers
for driver in pulse alsa dsp; do
  echo "Testing $driver..."
  timeout 30 python ilterm.py --prompt "test" --driver $driver --duration 5
done
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
- **pyloudnorm** by csteinmetz1 for LUFS normalization: https://github.com/csteinmetz1/pyloudnorm
- Developed with assistance from AI language models
- Audio processing powered by librosa, soundfile, and scipy
- GUI implementation using tkinter and matplotlib

## Changelog

- ➕ LUFS normalization for professional audio quality
- ➕ Enhanced loop detection with beat-focused fallback
- ➕ Zero-crossing optimization for seamless transitions
- ➕ Comprehensive error handling and file validation
- ➕ Process management with CPU priority and affinity
- ➕ Debug mode with detailed logging
- ➕ Interactive terminal with full command set
- ➕ Settings persistence in GUI version
- ➕ Preset system for quick generation
- 🔧 Improved memory management and cleanup
- 🔧 Better cross-platform audio driver support
- 🐛 Fixed race conditions in file swapping
- 🐛 Resolved audio dropout issues

**🎵 Enjoy infinite AI music with INFINI LOOP! 🎵**
---

# INFINI LOOP - Generazione Musicale Infinita con AI Locale

INFINI LOOP è un sistema musicale basato su intelligenza artificiale progettato per generare loop audio infiniti e senza interruzioni.
Crea automaticamente nuovi frammenti musicali, rileva i migliori punti di loop e li riproduce in sequenza continua, mentre prepara il segmento successivo — offrendo un flusso musicale ininterrotto e sempre nuovo.

All'avvio, uno dei due file .wav inclusi verrà riprodotto subito, così potrai ascoltare musica anche mentre l'IA prepara la prima generazione.

Una volta configurato, il tuo computer diventa una stazione musicale AI locale, in grado di produrre nuovi brani con transizioni fluide e rilevamento automatico del loop. Locale, privato, e più personale di qualsiasi playlist di YouTube o Spotify.

**NOVITÀ IN QUESTA VERSIONE:**
- **Normalizzazione**: Standardizzazione volume audio
- **Rilevamento Loop Migliorato**: Analisi multi-metrica avanzata con fallback basato sui ritmi
- **Ottimizzazione Zero-Crossing**: Raffinamento preciso dei punti di loop per transizioni perfette
- **Gestione Errori Robusta**: Validazione file avanzata e recupero da crash
- **Generazione Intelligente**: Sistema di retry adattivo con validazione qualità
- **Ottimizzazioni Performance**: Migliore gestione memoria e processi
- **Modalità Debug**: Logging completo per diagnostica

Versione GUI Avanzata (**sperimentale**):

<img width="1234" height="679" alt="immagine" src="https://github.com/user-attachments/assets/19390959-ba49-476e-8b26-df606c6dad36" />

Versione GUI Leggera (**PIÙ raccomandata**):

<img width="672" height="864" alt="immagine" src="https://github.com/user-attachments/assets/3f99ec86-661b-4a5a-8f0a-f97731340f84" />

Versione Terminale (**raccomandata**):

<img width="849" height="608" alt="immagine" src="https://github.com/user-attachments/assets/9a95d2dd-8690-4d00-8735-530511ef9498" />

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

- **Generazione Musicale AI Locale**: Utilizza MusicGPT per sintesi audio di alta qualità (https://github.com/gabotechs/MusicGPT)
- **Rilevamento Loop Avanzato**: Analisi multi-metrica con algoritmi adattivi
  - Analisi similarità spettrale
  - Misurazione continuità forma d'onda
  - Allineamento ritmi e preservazione beat
  - Ottimizzazione coerenza di fase
  - Raffinamento zero-crossing
- **Riproduzione Continua**: Loop infinito nativo con transizioni
- **Generazione Intelligente**: Sistema retry con validazione qualità e recupero errori
- **Interfacce Multiple**: Terminale, GUI leggera e GUI avanzata con visualizzazioni
- **Gestione Processi**: Ottimizzazione priorità CPU/IO e terminazione sicura
- **Funzione Esportazione**: Salva loop generati per uso futuro
- **Modalità Debug**: Logging completo e tracking stato file

## Versioni Disponibili

### 1. Versione Terminale (`ilterm.py`) - Più Stabile ⭐

- Interfaccia a riga di comando con modalità interattiva completa
- Supporto transizione reale tra i loop
- Validazione audio avanzata e recupero errori
- Modalità debug con logging dettagliato
- Modalità generazione singola per loop individuali
- Configurazione completa delle impostazioni
- Consumo risorse minimo

**Comandi Interattivi:**
- `start '<prompt>'` - Avvia loop infinito
- `stop` - Ferma riproduzione
- `status` - Status sistema dettagliato
- `save <file.wav>` - Esporta loop corrente
- `set duration/driver` - Cambia impostazioni
- `debug on/off` - Attiva/disattiva debug
- `validate current/next/both` - Controlla integrità file
- `help` - Mostra tutti i comandi

### 2. Versione GUI Leggera (`il1.py`) - Più Raccomandata ⭐

- Interfaccia grafica pulita con controlli essenziali
- Stesso motore audio robusto della versione terminale
- Supporto transizione reale con feedback visivo
- Sistema preset per generazione rapida
- Monitoraggio status in tempo reale
- Persistenza impostazioni
- Display informazioni loop (titolo, artista, durata)
- Salvataggio/caricamento configurazione

### 3. Versione GUI Avanzata (`il2.py`) - Sperimentale

- Interfaccia grafica completa con visualizzazioni audio
- Analisi forma d'onda e spettro in tempo reale
- Display visivo delle metriche del loop
- Parametri di generazione configurabili
- **Nota**: Non supporta crossfading tra tracce
- Maggiore utilizzo risorse per le visualizzazioni

## Requisiti di Sistema

### Requisiti Minimi

- **Sistema Operativo**: Linux (Ubuntu 20.04+ o equivalente)
- **Python**: 3.8 o superiore
- **RAM**: 8 GB memoria di sistema
- **CPU**: Processore multi-core (2+ core raccomandati)
- **Audio**: Sottosistema audio funzionante (PulseAudio, ALSA o OSS)
- **Storage**: 500 MB spazio libero per file temporanei

### Requisiti Consigliati

- **Sistema Operativo**: Ubuntu 22.04 LTS
- **Python**: 3.10 o superiore
- **RAM**: 16 GB memoria di sistema
- **CPU**: CPU moderna ad alta frequenza (3+ GHz, 4+ core)
- **Audio**: PulseAudio con configurazione a bassa latenza
- **Storage**: 2 GB spazio libero

### Note sulle Performance

- **MusicGPT (medium)** funziona meglio su CPU moderne ad alta frequenza
- **Supporto GPU** è sperimentale e non richiesto
- **Errori allocazione memoria** si verificano con RAM insufficiente (<8 GB)
- **Inferenza lenta** capita con CPU deboli (<2 core o <2 GHz)
- **Dropout audio** possono verificarsi senza driver audio configurati correttamente

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
chmod +x ilterm.py il1.py il2.py
```

### Passo 4: Scarica e Configura il Binario MusicGPT

1. Visita la [pagina releases di MusicGPT](https://github.com/gabotechs/MusicGPT/releases)
2. Scarica `musicgpt-x86_64-unknown-linux-gnu` (versione più recente)
3. Posizionalo nella stessa directory degli script Python
4. Rendilo eseguibile:

```bash
chmod +x musicgpt-x86_64-unknown-linux-gnu

# Verifica che il binario funzioni
./musicgpt-x86_64-unknown-linux-gnu --help
```

### Passo 5: Configura Sistema Audio (Linux)

```bash
# Assicurati che i servizi audio siano attivi
sudo systemctl --user enable pulseaudio
sudo systemctl --user start pulseaudio

# Testa output audio
speaker-test -t wav -c 2

# Opzionale: Configura audio a bassa latenza
echo "default-sample-rate = 44100" >> ~/.pulse/daemon.conf
echo "alternate-sample-rate = 48000" >> ~/.pulse/daemon.conf
pulseaudio --kill && pulseaudio --start
```

### Passo 6: File di Setup Opzionali

Crea questi file opzionali nella directory INFINI LOOP per funzionalità migliorate:

```bash
# Titoli casuali (nomi.txt, nomi2.txt)
echo -e "AMBIENT\nCHILL\nDREAM\nFLOW\nWAVE" > nomi.txt
echo -e "LOOPS\nBEATS\nSOUNDS\nVIBES\nRHYTHM" > nomi2.txt

# Artisti casuali (artisti.txt)
echo -e "AI COMPOSER\nDIGITAL ORCHESTRA\nSYNTHETIC SOUNDS\nNEURAL BEATS\nALGORITHMIC MUSIC" > artisti.txt
```

## Utilizzo

**Nota**: Al primo avvio, MusicGPT scaricherà il modello selezionato (medium di default). La primissima generazione sarà significativamente più lenta rispetto alle successive. Questo vale per tutte le versioni di INFINI LOOP.

### Versione Terminale (`ilterm.py`)

#### Avvio Rapido

```bash
# Modalità interattiva (consigliata)
python ilterm.py

# Generazione diretta con prompt
python ilterm.py --prompt "electronic dance loop"

# Impostazioni personalizzate
python ilterm.py --prompt "ambient chill" --duration 20 --driver pulse

# Genera singolo loop ed esci
python ilterm.py --generate-only "jazz piano" output.wav

# Modalità debug
python ilterm.py --prompt "test loop" --verbose
```

#### Comandi Interattivi

In modalità interattiva, usa questi comandi:

```bash
🎛️ > start 'ambient electronic loop'     # Avvia loop infinito
🎛️ > stop                                # Ferma riproduzione
🎛️ > status                              # Mostra status dettagliato
🎛️ > save my_favorite_loop.wav          # Esporta loop corrente
🎛️ > set duration                        # Cambia durata generazione (5-30s)
🎛️ > set driver                          # Cambia driver audio
🎛️ > debug on                            # Abilita logging debug
🎛️ > validate both                       # Controlla integrità file
🎛️ > help                                # Mostra tutti i comandi
🎛️ > quit                                # Esci dal programma
```

#### Opzioni Riga di Comando

```bash
python ilterm.py [OPZIONI]

Opzioni:
  -p, --prompt TEXT        Prompt per generazione musicale
  -i, --interactive        Avvia in modalità interattiva
  -g, --generate-only PROMPT OUTPUT  Genera singolo loop e salva
  -d, --duration INTEGER   Durata generazione (5-30 secondi)
  --driver [pulse|alsa|dsp]  Selezione driver audio
  -v, --verbose            Output dettagliato
  -q, --quiet              Output minimale
  --no-debug               Disabilita modalità debug
  -h, --help               Mostra messaggio aiuto
```

### Versione GUI Leggera (`il1.py`)

```bash
python il1.py
```

**Passi di Utilizzo:**
1. **Inserisci Prompt**: Digita la tua descrizione musicale (es. "calm acoustic guitar")
2. **Scegli Preset**: Clicca i pulsanti preset per configurazioni rapide
3. **Regola Durata**: Usa la tab impostazioni per cambiare durata generazione (10-15s ottimali)
4. **Avvia Generazione**: Clicca "▶️ START LOOP"
5. **Monitora Progresso**: Osserva barra stato e tab log
6. **Salva Loop**: Usa "💾 Save Current Loop" per esportare
7. **Impostazioni**: Configura durata, driver audio, modalità debug

**Preset Disponibili:**
- **Ambient**: Paesaggi sonori eterei
- **Reggae**: Ritmi reggae classici
- **Electronic**: Synth e beat da ballo
- **Classical**: Arrangiamenti orchestrali
- **Rock**: Loop guidati da chitarra
- **Lofi Rap**: Beat hip-hop melodici

### Versione GUI Avanzata (`il2.py`)

```bash
python il2.py
```

1. Inserisci il tuo prompt musicale nel campo testo
2. Scegli tipo algoritmo:
   - **Avanzato**: Analisi multi-metrica (consigliato)
   - **Classico**: Solo similarità spettrale
3. Seleziona modello (medium consigliato) e durata
4. Clicca "AVVIA" per iniziare generazione
5. Monitora visualizzazioni in tempo reale per analisi loop
6. Usa pulsante "SALVA" per salvare loop

**Nota**: Questa versione non supporta crossfading tra loop.

## Dettagli Tecnici

### Pipeline Elaborazione Audio

1. **Generazione AI**: MusicGPT crea audio grezzo usando prompt testuali
2. **Validazione Qualità**: Controllo integrità file multi-stadio
3. **Analisi Loop**: Algoritmo multi-metrico avanzato rileva punti loop ottimali
4. **Normalizzazione Audio**: Standardizzazione LUFS a -14 dB (standard broadcast)
5. **Ottimizzazione Zero-Crossing**: Affina punti loop per transizioni perfette
6. **Riproduzione Continua**: Loop infinito nativo con generazione in background

### Algoritmo Rilevamento Loop

INFINI LOOP usa un approccio sofisticato a due stadi:

#### Stadio 1: Analisi Multi-Metrica Avanzata
1. **Similarità Spettrale**: Analisi cepstrale mel-frequency dei confini loop
2. **Continuità Forma d'Onda**: Cross-correlazione e matching RMS
3. **Allineamento Beat**: Preservazione ritmo usando tracking beat
4. **Continuità Fase**: Analisi coerenza fase STFT
5. **Punteggio Composito**: Combinazione pesata di tutte le metriche

#### Stadio 2: Fallback Focalizzato sui Beat
1. **Rilevamento Tempo**: Analisi BPM con misurazione consistenza
2. **Struttura Musicale**: Preferenza per loop da 1, 2, 4, 8 misure
3. **Allineamento Griglia Beat**: Aggancio a posizioni beat rilevate
4. **Preservazione Ritmo**: Mantiene coerenza musicale

### Normalizzazione Audio

Normalizzazione LUFS (Loudness Units relative to Full Scale) professionale:
- **Target**: -14 LUFS (standard Spotify/YouTube)
- **Limitazione Picchi**: Previene clipping sopra -0.1 dBFS
- **Range Dinamico**: Preserva dinamiche musicali
- **Consistenza**: Volume uniforme su tutti i loop generati

### Gestione Processi

- **Priorità CPU**: Generazione in background con `nice` e `ionice`
- **Affinità CPU**: Assegnazione core per performance ottimali
- **Gestione Memoria**: Pulizia file temporanei e prevenzione leak
- **Terminazione Sicura**: Spegnimento processi con gestione timeout
- **Recupero Errori**: Retry automatico con backoff esponenziale

### Raccomandazioni Modelli

- **Modello Small**: Veloce ma spesso bassa qualità - non consigliato per musica
- **Modello Medium**: Miglior bilanciamento qualità/velocità - **consigliato**
- **Modello Large**: Qualità massima ma molto lento e intensivo di risorse

## Risoluzione Problemi

### Problemi Comuni e Soluzioni

#### Problemi di Generazione

**Problema**: "File audio generato con errori dalla AI"
- **Causa**: Binario MusicGPT non trovato o non eseguibile
- **Soluzione**:
  ```bash
  chmod +x musicgpt-x86_64-unknown-linux-gnu
  ./musicgpt-x86_64-unknown-linux-gnu --help  # Testa binario
  ```

**Problema**: Generazione molto lenta (>60 secondi)
- **Causa**: CPU insufficiente o memoria
- **Soluzioni**:
  - Usa durata più breve (8-12 secondi)
  - Chiudi altre applicazioni
  - Assicurati 16+ GB RAM disponibili
  - Controlla che CPU non sia throttling per calore

**Problema**: "No interesting loop" o output bassa qualità
- **Causa**: Audio generato da AI non adatto per loop
- **Soluzioni**:
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
  
  # Prova driver audio diversi
  python ilterm.py --driver alsa    # o pulse, dsp
  
  # Controlla PulseAudio
  pulseaudio --check -v
  systemctl --user restart pulseaudio
  ```

**Problema**: Audio che salta o dropout
- **Causa**: Buffer underrun audio o problemi driver
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
### Modalità Debug

Abilita logging completo per diagnostica:

```bash
# Versione terminale
python ilterm.py --prompt "test loop" --verbose

# Modalità interattiva
python ilterm.py
🎛️ > debug on
🎛️ > start 'test ambient'

# Versione GUI: Tab Impostazioni → Abilita modalità debug
```

Output debug include:
- Tracking stato file (creazione, validazione, eliminazione)
- Passi pipeline elaborazione audio
- Progresso algoritmo rilevamento loop
- Creazione e terminazione processi
- Utilizzo memoria e CPU
- Stack trace errori

### Ottimizzazione Performance

#### Per Sistemi Entry-Level:
```bash
# Riduci durata generazione
python ilterm.py --duration 8

# Usa ALSA per latenza più bassa
python ilterm.py --driver alsa

# Abbassa priorità processo
nice -n 10 python ilterm.py --prompt "ambient"
```

#### Per Sistemi High-End:
```bash
# Usa durate più lunghe per qualità migliore
python ilterm.py --duration 20

# Core CPU dedicati
taskset -c 2,3 python ilterm.py --prompt "complex orchestral"
```

### Analisi Log

Pattern log comuni e significati:

```
✅ Perfect loop found?                    # Rilevamento loop riuscito
❌ No interesting loop                    # Output AI non adatto per loop
🎚️ Normalized from X to -14 LUFS        # Standardizzazione volume audio
🎯 Zero-crossing optimization...         # Affinamento confini loop
🔄 Reinitializing...                     # Recupero da errore
⚠️ Zero-crossing rejected               # Ottimizzazione comprometterebbe ritmo
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

## Sviluppo e Contributi

### Struttura Codice

- `ilterm.py`: Engine core con interfaccia terminale
- `il1.py`: Wrapper GUI leggero attorno all'engine core
- `il2.py`: GUI avanzata con visualizzazioni (sperimentale)

### Classi e Metodi Chiave

**InfiniLoopTerminal** (engine principale):
- `find_perfect_loop_advanced()`: Rilevamento loop multi-metrico
- `find_perfect_loop_simple()`: Algoritmo fallback focalizzato sui beat
- `generate_audio_safe()`: Generazione AI con normalizzazione LUFS
- `process_loop_detection()`: Pipeline completa analisi loop
- `safe_file_swap()`: Gestione file thread-safe
- `validate_audio_file()`: Validazione file multi-stadio

### Segnalazioni Bug e Richieste Funzionalità

Quando segnali problemi, includi:

- **Versione**: Quale file (ilterm.py, il1.py, il2.py)
- **Sistema**: OS, versione Python, RAM, dettagli CPU
- **Audio**: Driver usato, hardware audio
- **Log**: Messaggi errore completi con modalità debug abilitata
- **Riproduzione**: Passi esatti per riprodurre il problema
- **Prompt**: Il prompt di generazione che ha causato problemi

### Testing

```bash
# Testa funzionalità di base
python ilterm.py --generate-only "test ambient" test_output.wav

# Testa sistema audio
python ilterm.py --prompt "short test" --duration 5 --verbose

# Testa tutti i driver
for driver in pulse alsa dsp; do
  echo "Testing $driver..."
  timeout 30 python ilterm.py --prompt "test" --driver $driver --duration 5
done
```

## Licenza

Questo progetto è rilasciato sotto licenza Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).

**Permessi:**
- ✅ Condividi e adatta il materiale
- ✅ Usa per scopi personali ed educativi
- ✅ Modifica e costruisci sopra il codice

**Restrizioni:**
- ❌ Uso commerciale senza permesso
- ❌ Distribuzione senza attribuzione

Per licenze commerciali, contatta gli autori.

## Crediti

- **MusicGPT** di gabotechs per generazione musicale AI: https://github.com/gabotechs/MusicGPT
- **librosa** team per analisi audio: https://librosa.org/
- **pyloudnorm** di csteinmetz1 per normalizzazione LUFS: https://github.com/csteinmetz1/pyloudnorm
- Sviluppato con assistenza di modelli linguistici AI
- Elaborazione audio basata su librosa, soundfile e scipy
- Implementazione GUI usando tkinter e matplotlib

## Changelog

- ➕ Normalizzazione LUFS per qualità audio professionale
- ➕ Rilevamento loop migliorato con fallback basato sui beat
- ➕ Ottimizzazione zero-crossing per transizioni perfette
- ➕ Gestione errori completa e validazione file
- ➕ Gestione processi con priorità CPU e affinità
- ➕ Modalità debug con logging dettagliato
- ➕ Terminale interattivo con set comandi completo
- ➕ Persistenza impostazioni nella versione GUI
- ➕ Sistema preset per generazione rapida
- 🔧 Gestione memoria e pulizia migliorate
- 🔧 Supporto driver audio multi-piattaforma migliorato
- 🐛 Corrette race condition nel file swapping
- 🐛 Risolti problemi dropout audio

---

**🎵 Buona musica con INFINI LOOP! 🎵**
