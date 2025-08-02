# INFINI LOOP - Infinite AI Music Generation

**INFINI LOOP** is an advanced AI-powered music generation tool that creates seamless, infinite loops in real-time. Using sophisticated loop detection algorithms and AI music generation, it provides continuous, high-quality musical experiences perfect for ambient music, background tracks, and creative compositions.

<img width="1234" height="679" alt="immagine" src="https://github.com/user-attachments/assets/19390959-ba49-476e-8b26-df606c6dad36" />

## Available Versions

The project includes two complementary applications:

### il2.py - Full GUI Application
The main application featuring a complete graphical interface with real-time visualization and comprehensive loop analysis tools.

### ilterm.py - Terminal Version (Lightweight)
A streamlined command-line version designed for minimal resource usage, server environments, and advanced file handling with crash recovery.

## Features

### Core Functionality (Both Versions)
- **AI-Powered Generation**: Creates unique music loops using MusicGPT AI models (https://github.com/gabotechs/MusicGPT)
- **Infinite Seamless Playback**: Continuous loop playback with perfect transitions
- **Smart Loop Detection**: Advanced algorithms to find optimal loop points
- **Zero-Crossing Optimization**: Perfect cut points to eliminate audio clicks
- **Beat Tracking & Tempo Detection**: Musical timing awareness for better loops

### Advanced Algorithms (Both Versions)
- **Multi-Metric Analysis**: 
  - Spectral similarity analysis (25% weight)
  - Waveform continuity calculation (35% weight)
  - Beat alignment detection (25% weight)
  - Phase continuity optimization (15% weight)
- **Classic Algorithm**: Mel-spectrogram spectral similarity (faster fallback)
- **Automatic Fallback**: Advanced algorithm with simple algorithm backup

### GUI Version Exclusive Features (il2.py)
- **Modern Dark GUI**: Professional tkinter-based interface
- **Real-Time Visualization**: Live waveform and spectrum analysis with matplotlib
- **Loop Metrics Display**: Visual representation of detection algorithm scores
- **AI Model Selection**: Choose between Small, Medium, or Large models
- **Crossfade Control**: Adjustable overlap timing (1-5000ms)
- **Advanced Settings Panel**: Complete control over all generation parameters
- **Track Information**: Generated random titles and artist names
- **Console Logging**: Real-time process information display
- **PyAudio Support**: Low-latency audio playback (with ffplay fallback)

### Terminal Version Exclusive Features (ilterm.py)
- **Command-Line Interface**: Text-based operation for scripts and servers
- **Interactive Mode**: Rich command system with help and status
- **Generate-Only Mode**: Create single loops without playback
- **Advanced File Handling**: Safe temporary files and corruption recovery
- **Debug Mode**: Detailed file state tracking and validation
- **Crash Recovery**: Robust error handling and process cleanup
- **Fixed Configuration**: Optimized settings (Medium model, 1ms crossfade)

### Audio Engine (Both Versions)
- **Multiple Driver Support**: PulseAudio, ALSA, OSS compatibility
- **Audio Optimization**: Automatic normalization and level control
- **Loop Export**: Save perfect loops as WAV files
- **Process Management**: Automatic cleanup of audio processes

## Requirements

### System Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum (GUI), 2GB RAM minimum (Terminal)
- **Storage**: 2GB free space for temporary files

### Dependencies
```bash
# Core dependencies (both versions)
librosa>=0.9.0
soundfile>=0.10.0
scipy>=1.7.0
numpy>=1.21.0
pydub>=0.25.0

# GUI version additional dependencies
tkinter (usually included with Python)
matplotlib>=3.5.0
Pillow>=8.0.0

# Optional but recommended for GUI
pyaudio>=0.2.11

# Required external binary
musicgpt-x86_64-unknown-linux-gnu (https://github.com/gabotechs/MusicGPT)
```

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/davidegat/infiniloop.git
cd infiniloop
```

### 2. Install Python Dependencies

For **GUI Version** (il2.py):
```bash
# Install all dependencies including GUI components
pip install librosa soundfile scipy numpy matplotlib pillow pydub

# Install PyAudio for low-latency audio (recommended)
# On Ubuntu/Debian:
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio
```

For **Terminal Version** (ilterm.py):
```bash
# Install core dependencies only
pip install librosa soundfile scipy numpy pydub
```

### 3. Install System Audio Dependencies
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install ffmpeg pulseaudio-utils alsa-utils

# Fedora/RHEL
sudo dnf install ffmpeg pulseaudio-utils alsa-utils

# Arch Linux
sudo pacman -S ffmpeg pulseaudio alsa-utils
```

### 4. Download MusicGPT Binary
The software requires the MusicGPT AI model binary from repository (https://github.com/gabotechs/MusicGPT):

```bash
# Download MusicGPT (replace with actual download link)
wget [MUSICGPT_DOWNLOAD_URL] -O musicgpt-x86_64-unknown-linux-gnu
chmod +x musicgpt-x86_64-unknown-linux-gnu
```
## Usage

### GUI Version (il2.py)

#### Basic Usage
1. **Launch the application**:
   ```bash
   python il2.py
   ```

2. **Enter a music prompt**: 
   - Example: "lofi calm rap beat"
   - Example: "ambient electronic soundscape"
   - Example: "jazz piano melody"

3. **Choose algorithm**:
   - **Advanced**: Multi-metric analysis (recommended)
   - **Classic**: Spectral similarity only (faster)

4. **Click "AVVIA" (START)** to begin generation and playback

#### Advanced Settings
Access the settings panel (SETUP) to configure:
- **AI Model**: Small (fast) / Medium (balanced) / Large (quality)
- **Generation Duration**: 5-30 seconds
- **Audio Driver**: PulseAudio/ALSA/OSS
- **Crossfade Timing**: 1-5000ms overlap

#### Controls
- **AVVIA**: Start infinite loop generation
- **FERMA**: Stop playback and generation
- **SALVA**: Export current loop as WAV file
- **SETUP**: Open advanced settings
- **OVERLAP**: Adjust crossfade timing (slider)

### Terminal Version (ilterm.py)

#### Command Line Usage
```bash
# Direct prompt mode
python ilterm.py --prompt "ambient drone loop"

# Interactive mode
python ilterm.py --interactive

# Generate single loop only
python ilterm.py --generate-only "jazz loop" output.wav

# With custom settings
python ilterm.py --prompt "rock loop" --duration 20 --driver alsa
```

#### Interactive Commands
```bash
start <prompt>     # Start infinite loop with prompt
stop               # Stop current playback
status             # Show detailed system status
save <file.wav>    # Save current loop to file
set duration       # Change generation duration (5-30s)
set driver         # Change audio driver (pulse/alsa/dsp)
validate <target>  # Validate file integrity (current/next/both)
debug on/off       # Toggle debug mode
help               # Show all commands
quit               # Exit application
```

#### Examples
```bash
# Interactive session
python ilterm.py -i
> start 'ambient chill loop'
> status
> save my_loop.wav
> quit

# Direct generation
python ilterm.py -g "electronic beat" beat.wav
```

## Algorithm Details

### Advanced Algorithm (Multi-Metric)
Analyzes multiple aspects of audio for optimal loop points:
1. **Spectral Similarity** (25%): Frequency content matching using cosine distance
2. **Waveform Continuity** (35%): Amplitude transition smoothness with correlation analysis
3. **Beat Alignment** (25%): Musical timing synchronization using detected beats
4. **Phase Continuity** (15%): Phase relationship optimization using STFT analysis

### Classic Algorithm (Spectral)
Uses mel-spectrogram analysis for faster processing with good results, focused on frequency domain similarity.

### Loop Optimization
- **Zero-crossing detection** for click-free transitions
- **Automatic tempo and beat detection** using librosa
- **Crossfade optimization** with configurable overlap
- **Audio normalization** and level control

## Troubleshooting

### Common Issues

**Audio Problems**:
```bash
# Check audio system
pulseaudio --check -v
aplay -l

# Restart audio if needed
pulseaudio --kill && pulseaudio --start
```

**PyAudio Installation Issues** (GUI only):
```bash
# Install development headers first
sudo apt-get install portaudio19-dev python3-dev
pip install pyaudio
```

**Permission Issues**:
```bash
# Make MusicGPT executable
chmod +x musicgpt-x86_64-unknown-linux-gnu

# Check audio group membership
groups $USER
sudo usermod -a -G audio $USER  # if needed
```

**Performance Issues**:
- **GUI**: Reduce generation duration, use Small AI model
- **Terminal**: Use `--duration` parameter, check with `status` command
- Close other audio applications
- Monitor system resources

### Version-Specific Issues

**GUI Version (il2.py)**:
- **High memory usage**: Normal due to matplotlib visualization
- **Display issues**: Check X11 forwarding if using SSH
- **Visualization lag**: Reduce crossfade overlap or use smaller windows

**Terminal Version (ilterm.py)**:
- **File corruption**: Use `validate` command to check file integrity
- **Process cleanup**: Terminal version has automatic cleanup and recovery
- **Debug information**: Use `debug on` for detailed file operation logs

## Project Structure
```
infiniloop/
├── il2.py                              # Main GUI application
├── ilterm.py                           # Terminal/CLI application
├── artisti.txt                         # Artist names
├── nomi.txt                           # Title words part 1
├── nomi2.txt                          # Title words part 2
├── music1.wav                         # Temporary audio file 1
├── music2.wav                         # Temporary audio file 2
└── README.md                          # This file
```

## When to Use Each Version

### Choose GUI Version (il2.py) when:
- You need visual feedback and real-time analysis
- You want to experiment with different AI models and settings
- You prefer point-and-click interface
- You need detailed loop metrics and visualization
- You want PyAudio low-latency performance

### Choose Terminal Version (ilterm.py) when:
- You're working on a server or headless system
- You need minimal resource usage and maximum stability
- You want to script or automate loop generation
- You prefer command-line workflows
- You need advanced file validation and crash recovery

## Performance Comparison

| Feature | GUI Version | Terminal Version |
|---------|-------------|------------------|
| CPU Usage | Higher (visualization) | Lower (no GUI) |
| Stability | Good | Excellent (crash recovery) |
| Audio Latency | Lower (PyAudio) | Standard (ffplay) |
| Customization | Full control | Fixed optimized settings |

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Notes
- Both versions share core loop detection algorithms in similar functions
- GUI-specific features are isolated in il2.py
- Terminal version includes additional safety and recovery mechanisms
- Test both versions when modifying core algorithms

## License

This project is licensed under the GPL License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **librosa**: Audio analysis and beat tracking
- **MusicGPT**: AI music generation model
- **PyAudio**: Low-latency audio playback (GUI version)
- **NumPy/SciPy**: Mathematical computing and signal processing
- **matplotlib**: Real-time visualization (GUI version)

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Specify which version (GUI/Terminal) you're using
- Include relevant logs and system information
- Check the troubleshooting section first

---

## ITALIANO

# INFINI LOOP - Generazione Musicale AI Infinita

**INFINI LOOP** è uno strumento avanzato per la generazione musicale AI che crea loop seamless infiniti in tempo reale. Utilizzando algoritmi sofisticati di rilevamento loop e generazione musicale AI, fornisce esperienze musicali continue e di alta qualità, perfette per musica ambient, tracce di sottofondo e composizioni creative.

## Versioni Disponibili

Il progetto include due applicazioni complementari:

### il2.py - Applicazione GUI Completa
L'applicazione principale con interfaccia grafica completa, visualizzazione in tempo reale e strumenti di analisi loop comprensivi.

### ilterm.py - Versione Terminale (Leggera)
Una versione semplificata da riga di comando progettata per uso minimo risorse, ambienti server e gestione file avanzata con recupero da crash.

## Caratteristiche

### Funzionalità Principali (Entrambe le Versioni)
- **Generazione AI**: Crea loop musicali unici usando modelli AI MusicGPT
- **Riproduzione Infinita Seamless**: Riproduzione continua con transizioni perfette
- **Rilevamento Loop Intelligente**: Algoritmi avanzati per trovare punti di loop ottimali
- **Ottimizzazione Zero-Crossing**: Punti di taglio perfetti per eliminare click audio
- **Rilevamento Beat e Tempo**: Consapevolezza del timing musicale per loop migliori

### Algoritmi Avanzati (Entrambe le Versioni)
- **Analisi Multi-Metrica**:
  - Analisi similarità spettrale (peso 25%)
  - Calcolo continuità forma d'onda (peso 35%)
  - Rilevamento allineamento beat (peso 25%)
  - Ottimizzazione continuità fase (peso 15%)
- **Algoritmo Classico**: Similarità spettrale mel-spettrogramma (fallback più veloce)
- **Fallback Automatico**: Algoritmo avanzato con backup su algoritmo semplice

### Funzionalità Esclusive Versione GUI (il2.py)
- **GUI Scura Moderna**: Interfaccia professionale basata su tkinter
- **Visualizzazione Tempo Reale**: Analisi live forma d'onda e spettro con matplotlib
- **Display Metriche Loop**: Rappresentazione visiva dei punteggi algoritmi di rilevamento
- **Selezione Modello AI**: Scelta tra modelli Small, Medium o Large
- **Controllo Crossfade**: Timing sovrapposizione regolabile (1-5000ms)
- **Pannello Impostazioni Avanzate**: Controllo completo su tutti i parametri di generazione
- **Informazioni Traccia**: Titoli e nomi artista casuali generati
- **Logging Console**: Visualizzazione informazioni processo in tempo reale
- **Supporto PyAudio**: Riproduzione audio bassa latenza (con fallback ffplay)

### Funzionalità Esclusive Versione Terminale (ilterm.py)
- **Interfaccia Riga di Comando**: Operazione testuale per script e server
- **Modalità Interattiva**: Sistema comandi ricco con aiuto e stato
- **Modalità Solo-Generazione**: Crea singoli loop senza riproduzione
- **Gestione File Avanzata**: File temporanei sicuri e recupero da corruzione
- **Modalità Debug**: Tracciamento dettagliato stato file e validazione
- **Recupero da Crash**: Gestione errori robusta e pulizia processi
- **Configurazione Fissa**: Impostazioni ottimizzate (modello Medium, crossfade 1ms)

### Motore Audio (Entrambe le Versioni)
- **Supporto Driver Multipli**: Compatibilità PulseAudio, ALSA, OSS
- **Ottimizzazione Audio**: Normalizzazione automatica e controllo livelli
- **Export Loop**: Salvataggio loop perfetti come file WAV
- **Gestione Processi**: Pulizia automatica processi audio

## Requisiti

### Requisiti di Sistema
- **OS**: Linux (Ubuntu 20.04+ raccomandato)
- **Python**: 3.8 o superiore
- **Memoria**: 4GB RAM minimo (GUI), 2GB RAM minimo (Terminale)
- **Storage**: 2GB spazio libero per file temporanei

### Dipendenze
```bash
# Dipendenze core (entrambe le versioni)
librosa>=0.9.0
soundfile>=0.10.0
scipy>=1.7.0
numpy>=1.21.0
pydub>=0.25.0

# Dipendenze aggiuntive versione GUI
tkinter (solitamente incluso con Python)
matplotlib>=3.5.0
Pillow>=8.0.0

# Opzionale ma raccomandato per GUI
pyaudio>=0.2.11

# Binario esterno richiesto
musicgpt-x86_64-unknown-linux-gnu
```

## Installazione

### 1. Clona il Repository
```bash
git clone https://github.com/davidegat/infiniloop.git
cd infiniloop
```

### 2. Installa Dipendenze Python

Per **Versione GUI** (il2.py):
```bash
# Installa tutte le dipendenze inclusi componenti GUI
pip install librosa soundfile scipy numpy matplotlib pillow pydub

# Installa PyAudio per audio bassa latenza (raccomandato)
# Su Ubuntu/Debian:
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio
```

Per **Versione Terminale** (ilterm.py):
```bash
# Installa solo dipendenze core
pip install librosa soundfile scipy numpy pydub
```

### 3. Installa Dipendenze Audio di Sistema
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install ffmpeg pulseaudio-utils alsa-utils

# Fedora/RHEL
sudo dnf install ffmpeg pulseaudio-utils alsa-utils

# Arch Linux
sudo pacman -S ffmpeg pulseaudio alsa-utils
```

### 4. Scarica Binario MusicGPT
Il software richiede il binario del modello AI MusicGPT:

```bash
# Scarica MusicGPT (sostituisci con il link di download effettivo)
wget [MUSICGPT_DOWNLOAD_URL] -O musicgpt-x86_64-unknown-linux-gnu
chmod +x musicgpt-x86_64-unknown-linux-gnu
```

## Utilizzo

### Versione GUI (il2.py)

#### Utilizzo Base
1. **Avvia l'applicazione**:
   ```bash
   python il2.py
   ```

2. **Inserisci un prompt musicale**:
   - Esempio: "lofi calm rap beat"
   - Esempio: "ambient electronic soundscape"
   - Esempio: "jazz piano melody"

3. **Scegli algoritmo**:
   - **Avanzato**: Analisi multi-metrica (raccomandato)
   - **Classico**: Solo similarità spettrale (più veloce)

4. **Clicca "AVVIA"** per iniziare generazione e riproduzione

#### Impostazioni Avanzate
Accedi al pannello impostazioni (SETUP) per configurare:
- **Modello AI**: Small (veloce) / Medium (bilanciato) / Large (qualità)
- **Durata Generazione**: 5-30 secondi
- **Driver Audio**: PulseAudio/ALSA/OSS
- **Timing Crossfade**: 1-5000ms sovrapposizione

#### Controlli
- **AVVIA**: Inizia generazione loop infinito
- **FERMA**: Ferma riproduzione e generazione
- **SALVA**: Esporta loop corrente come file WAV
- **SETUP**: Apri impostazioni avanzate
- **OVERLAP**: Regola timing crossfade (slider)

### Versione Terminale (ilterm.py)

#### Utilizzo Riga di Comando
```bash
# Modalità prompt diretta
python ilterm.py --prompt "ambient drone loop"

# Modalità interattiva
python ilterm.py --interactive

# Genera solo singolo loop
python ilterm.py --generate-only "jazz loop" output.wav

# Con impostazioni personalizzate
python ilterm.py --prompt "rock loop" --duration 20 --driver alsa
```

#### Comandi Interattivi
```bash
start <prompt>     # Inizia loop infinito con prompt
stop               # Ferma riproduzione corrente
status             # Mostra stato sistema dettagliato
save <file.wav>    # Salva loop corrente su file
set duration       # Cambia durata generazione (5-30s)
set driver         # Cambia driver audio (pulse/alsa/dsp)
validate <target>  # Valida integrità file (current/next/both)
debug on/off       # Attiva/disattiva modalità debug
help               # Mostra tutti i comandi
quit               # Esci dall'applicazione
```

#### Esempi
```bash
# Sessione interattiva
python ilterm.py -i
> start 'ambient chill loop'
> status
> save my_loop.wav
> quit

# Generazione diretta
python ilterm.py -g "electronic beat" beat.wav
```

## Dettagli Algoritmi

### Algoritmo Avanzato (Multi-Metrico)
Analizza molteplici aspetti dell'audio per punti di loop ottimali:
1. **Similarità Spettrale** (25%): Corrispondenza contenuto frequenza usando distanza coseno
2. **Continuità Forma d'Onda** (35%): Fluidità transizione ampiezza con analisi correlazione
3. **Allineamento Beat** (25%): Sincronizzazione timing musicale usando beat rilevati
4. **Continuità Fase** (15%): Ottimizzazione relazione fase usando analisi STFT

### Algoritmo Classico (Spettrale)
Usa analisi mel-spettrogramma per elaborazione più veloce con buoni risultati, focalizzato su similarità dominio frequenza.

### Ottimizzazione Loop
- **Rilevamento zero-crossing** per transizioni senza click
- **Rilevamento automatico tempo e beat** usando librosa
- **Ottimizzazione crossfade** con sovrapposizione configurabile
- **Normalizzazione audio** e controllo livelli

## Struttura Progetto
```
infiniloop/
├── il2.py                              # Applicazione GUI principale
├── ilterm.py                           # Applicazione Terminale/CLI
├── artisti.txt                         # Nomi artista
├── nomi.txt                           # Parole titolo parte 1
├── nomi2.txt                          # Parole titolo parte 2
├── music1.wav                         # File audio temporaneo 1
├── music2.wav                         # File audio temporaneo 2
└── README.md                          # Questo file
```

## Quando Usare Ogni Versione

### Scegli Versione GUI (il2.py) quando:
- Hai bisogno di feedback visivo e analisi tempo reale
- Vuoi sperimentare con diversi modelli AI e impostazioni
- Preferisci interfaccia point-and-click
- Hai bisogno di metriche loop dettagliate e visualizzazione
- Vuoi prestazioni PyAudio bassa latenza

### Scegli Versione Terminale (ilterm.py) quando:
- Stai lavorando su un server o sistema headless
- Hai bisogno di uso risorse minimo e stabilità massima
- Vuoi scriptare o automatizzare generazione loop
- Preferisci workflow riga di comando
- Hai bisogno di validazione file avanzata e recupero da crash

## Confronto Prestazioni

| Caratteristica | Versione GUI | Versione Terminale |
|----------------|--------------|-------------------|
| Uso CPU | Maggiore (visualizzazione) | Minore (no GUI) |
| Stabilità | Buona | Eccellente (recupero crash) |
| Latenza Audio | Minore (PyAudio) | Standard (ffplay) |
| Personalizzazione | Controllo completo | Impostazioni ottimizzate fisse |

## Licenza

Questo progetto è rilasciato sotto Licenza GPL - vedi il file [LICENSE](LICENSE) per dettagli.

## Ringraziamenti

- **librosa**: Analisi audio e rilevamento beat
- **MusicGPT**: Modello generazione musicale AI
- **PyAudio**: Riproduzione audio bassa latenza (versione GUI)
- **NumPy/SciPy**: Calcolo matematico e elaborazione segnali
- **matplotlib**: Visualizzazione tempo reale (versione GUI)

## Supporto

Per problemi, domande o contributi:
- Apri un issue su GitHub
- Specifica quale versione (GUI/Terminale) stai usando
- Includi log rilevanti e informazioni sistema
- Controlla prima la sezione risoluzione problemi

---

*Creato con passione per la comunità della musica digitale*
