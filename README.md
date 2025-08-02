# INFINI LOOP - Infinite AI Music Generation

**INFINI LOOP** is an advanced AI-powered music generation tool that creates seamless, infinite loops in real-time. Using sophisticated loop detection algorithms and AI music generation, it provides continuous, high-quality musical experiences perfect for ambient music, background tracks, and creative compositions.

<img width="1234" height="679" alt="immagine" src="https://github.com/user-attachments/assets/19390959-ba49-476e-8b26-df606c6dad36" />

## Features

### Core Functionality
- **AI-Powered Generation**: Creates unique music loops using advanced AI models
- **Infinite Seamless Playback**: Continuous loop playback with perfect transitions
- **Real-Time Processing**: Live generation and playback with minimal latency
- **Smart Loop Detection**: Advanced algorithms to find optimal loop points
- **Crossfade Control**: Adjustable overlap timing for smooth transitions (1-5000ms)

### Advanced Algorithms
- **Multi-Metric Analysis**: 
  - Spectral similarity analysis
  - Waveform continuity calculation
  - Beat alignment detection
  - Phase continuity optimization
- **Zero-Crossing Optimization**: Perfect cut points to eliminate audio clicks
- **Tempo & Beat Detection**: Musical timing awareness for better loops
- **Mel-Spectrogram Analysis**: Frequency-domain processing for optimal results

### Audio Engine
- **Low-Latency Playback**: PyAudio integration for professional audio performance
- **Multiple Driver Support**: PulseAudio, ALSA, OSS compatibility
- **Fallback Systems**: Automatic ffplay fallback for maximum compatibility
- **Audio Optimization**: Automatic normalization and level control

### User Interface
- **Modern Dark GUI**: Professional, eye-friendly interface
- **Real-Time Visualization**: Live waveform and spectrum analysis
- **Advanced Settings Panel**: Full control over generation parameters
- **Console Logging**: Detailed real-time process information
- **Track Information**: Generated titles and artist names
- **Export Functionality**: Save perfect loops as WAV files

## Requirements

### System Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB free space for temporary files

### Dependencies
```bash
# Audio processing
librosa>=0.9.0
soundfile>=0.10.0
scipy>=1.7.0
numpy>=1.21.0

# GUI and visualization
tkinter (usually included with Python)
matplotlib>=3.5.0
Pillow>=8.0.0

# Audio playback (optional but recommended)
pyaudio>=0.2.11
pydub>=0.25.0

# AI Model (required)
# MusicGPT binary (see installation instructions)
```

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/davidegat/infiniloop.git
cd infiniloop
```

### 2. Install Python Dependencies
```bash
# Install required packages
pip install librosa soundfile scipy numpy matplotlib pillow pydub

# Install PyAudio for low-latency audio (recommended)
# On Ubuntu/Debian:
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio

# On other distributions, install portaudio development packages first
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
The software requires the MusicGPT AI model binary. Download the appropriate version for your system:

```bash
# Download MusicGPT (replace with actual download link)
wget [MUSICGPT_DOWNLOAD_URL] -O musicgpt-x86_64-unknown-linux-gnu
chmod +x musicgpt-x86_64-unknown-linux-gnu
```

### 5. Create Resource Files (Optional)
For enhanced experience, create these optional text files in the project directory:

```bash
# Artist names for random generation
echo -e "AMBIENT COLLECTIVE\nDIGITAL DREAMS\nCYBER SOUNDSCAPE" > artisti.txt

# Title word lists for random generation
echo -e "INFINITE\nCOSMIC\nETHEREAL" > nomi.txt
echo -e "JOURNEY\nVIBES\nWAVES" > nomi2.txt
```

## Usage

### Basic Usage
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

### Advanced Settings
Access the settings panel (SETUP) to configure:

- **AI Model**: 
  - Small (fast generation)
  - Medium (balanced quality/speed)
  - Large (highest quality)
- **Generation Duration**: 5-30 seconds
- **Audio Driver**: PulseAudio/ALSA/OSS
- **Crossfade Timing**: 1-5000ms overlap

### Controls
- **AVVIA**: Start infinite loop generation
- **FERMA**: Stop playback and generation
- **SALVA**: Export current loop as WAV file
- **SETUP**: Open advanced settings
- **OVERLAP**: Adjust crossfade timing

## Troubleshooting

### Common Issues

**Audio Problems**:
```bash
# Check audio system
pulseaudio --check -v
# or
aplay -l

# Restart audio if needed
pulseaudio --kill && pulseaudio --start
```

**PyAudio Installation Issues**:
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
# Add to audio group if needed:
sudo usermod -a -G audio $USER
```

**Performance Issues**:
- Reduce generation duration in settings
- Use "Small" AI model for faster generation
- Close other audio applications
- Check system resources (CPU/RAM usage)

### Logs and Debugging
The application provides detailed logging in the console panel. Common log messages:
- `Modalità bassa latenza attiva!`: PyAudio working correctly
- `PyAudio non trovato`: Using ffplay fallback
- `Loop perfetto!`: Successful loop detection
- `Errore`: Check system configuration

## Algorithm Details

### Advanced Algorithm (Multi-Metric)
The advanced algorithm analyzes multiple aspects:
1. **Spectral Similarity** (25%): Frequency content matching
2. **Waveform Continuity** (35%): Amplitude transition smoothness  
3. **Beat Alignment** (25%): Musical timing synchronization
4. **Phase Continuity** (15%): Phase relationship optimization

### Classic Algorithm (Spectral)
Uses mel-spectrogram analysis for faster processing with good results.

### Loop Optimization
- Zero-crossing detection for click-free transitions
- Automatic tempo and beat detection
- Crossfade optimization
- Audio normalization and level control

## Project Structure
```
infiniloop/
├── il2.py                              # Main application
├── musicgpt-x86_64-unknown-linux-gnu   # AI model binary
├── artisti.txt                         # Artist names (optional)
├── nomi.txt                           # Title words part 1 (optional)  
├── nomi2.txt                          # Title words part 2 (optional)
├── music1.wav                         # Temporary audio file
├── music2.wav                         # Temporary audio file
└── README.md                          # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the GPL License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **librosa**: Audio analysis library
- **MusicGPT**: AI music generation model
- **PyAudio**: Low-latency audio playback
- **NumPy/SciPy**: Mathematical computing

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check the troubleshooting section
- Review the console logs for detailed error information

---

## ITALIANO

# INFINI LOOP - Generazione Musicale AI Infinita

**INFINI LOOP** è uno strumento avanzato per la generazione musicale AI che crea loop seamless infiniti in tempo reale. Utilizzando algoritmi sofisticati di rilevamento loop e generazione musicale AI, fornisce esperienze musicali continue e di alta qualità, perfette per musica ambient, tracce di sottofondo e composizioni creative.

## Caratteristiche

### Funzionalità Principali
- **Generazione AI**: Crea loop musicali unici usando modelli AI avanzati
- **Riproduzione Infinita Seamless**: Riproduzione continua con transizioni perfette
- **Elaborazione in Tempo Reale**: Generazione e riproduzione live con latenza minima
- **Rilevamento Loop Intelligente**: Algoritmi avanzati per trovare punti di loop ottimali
- **Controllo Crossfade**: Timing di sovrapposizione regolabile per transizioni fluide (1-5000ms)

### Algoritmi Avanzati
- **Analisi Multi-Metrica**:
  - Analisi similarità spettrale
  - Calcolo continuità della forma d'onda
  - Rilevamento allineamento beat
  - Ottimizzazione continuità di fase
- **Ottimizzazione Zero-Crossing**: Punti di taglio perfetti per eliminare click audio
- **Rilevamento Tempo e Beat**: Consapevolezza del timing musicale per loop migliori
- **Analisi Mel-Spettrogramma**: Elaborazione dominio frequenza per risultati ottimali

### Motore Audio
- **Riproduzione Bassa Latenza**: Integrazione PyAudio per prestazioni audio professionali
- **Supporto Driver Multipli**: Compatibilità PulseAudio, ALSA, OSS
- **Sistemi di Fallback**: Fallback automatico ffplay per massima compatibilità
- **Ottimizzazione Audio**: Normalizzazione automatica e controllo livelli

### Interfaccia Utente
- **GUI Scura Moderna**: Interfaccia professionale e rilassante per gli occhi
- **Visualizzazione Tempo Reale**: Analisi live forma d'onda e spettro
- **Pannello Impostazioni Avanzate**: Controllo completo sui parametri di generazione
- **Logging Console**: Informazioni dettagliate del processo in tempo reale
- **Informazioni Traccia**: Titoli e nomi artista generati
- **Funzionalità Export**: Salva loop perfetti come file WAV

## Requisiti

### Requisiti di Sistema
- **OS**: Linux (Ubuntu 20.04+ raccomandato)
- **Python**: 3.8 o superiore
- **Memoria**: 4GB RAM minimo, 8GB raccomandato
- **Storage**: 2GB spazio libero per file temporanei

### Dipendenze
```bash
# Elaborazione audio
librosa>=0.9.0
soundfile>=0.10.0
scipy>=1.7.0
numpy>=1.21.0

# GUI e visualizzazione
tkinter (solitamente incluso con Python)
matplotlib>=3.5.0
Pillow>=8.0.0

# Riproduzione audio (opzionale ma raccomandato)
pyaudio>=0.2.11
pydub>=0.25.0

# Modello AI (richiesto)
# Binario MusicGPT (vedi istruzioni installazione)
```

## Installazione

### 1. Clona il Repository
```bash
git clone https://github.com/davidegat/infiniloop.git
cd infiniloop
```

### 2. Installa Dipendenze Python
```bash
# Installa pacchetti richiesti
pip install librosa soundfile scipy numpy matplotlib pillow pydub

# Installa PyAudio per audio bassa latenza (raccomandato)
# Su Ubuntu/Debian:
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio

# Su altre distribuzioni, installa prima i pacchetti di sviluppo portaudio
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
Il software richiede il binario del modello AI MusicGPT. Scarica la versione appropriata per il tuo sistema:

```bash
# Scarica MusicGPT (sostituisci con il link di download effettivo)
wget [MUSICGPT_DOWNLOAD_URL] -O musicgpt-x86_64-unknown-linux-gnu
chmod +x musicgpt-x86_64-unknown-linux-gnu
```

### 5. Crea File Risorse (Opzionale)
Per un'esperienza migliorata, crea questi file di testo opzionali nella directory del progetto:

```bash
# Nomi artista per generazione casuale
echo -e "AMBIENT COLLECTIVE\nDIGITAL DREAMS\nCYBER SOUNDSCAPE" > artisti.txt

# Liste parole titolo per generazione casuale
echo -e "INFINITE\nCOSMIC\nETHEREAL" > nomi.txt
echo -e "JOURNEY\nVIBES\nWAVES" > nomi2.txt
```

## Utilizzo

### Utilizzo Base
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

### Impostazioni Avanzate
Accedi al pannello impostazioni (SETUP) per configurare:

- **Modello AI**:
  - Small (generazione veloce)
  - Medium (qualità/velocità bilanciata)
  - Large (qualità massima)
- **Durata Generazione**: 5-30 secondi
- **Driver Audio**: PulseAudio/ALSA/OSS
- **Timing Crossfade**: 1-5000ms sovrapposizione

### Controlli
- **AVVIA**: Inizia generazione loop infinito
- **FERMA**: Ferma riproduzione e generazione
- **SALVA**: Esporta loop corrente come file WAV
- **SETUP**: Apri impostazioni avanzate
- **OVERLAP**: Regola timing crossfade

## Risoluzione Problemi

### Problemi Comuni

**Problemi Audio**:
```bash
# Controlla sistema audio
pulseaudio --check -v
# oppure
aplay -l

# Riavvia audio se necessario
pulseaudio --kill && pulseaudio --start
```

**Problemi Installazione PyAudio**:
```bash
# Installa prima gli header di sviluppo
sudo apt-get install portaudio19-dev python3-dev
pip install pyaudio
```

**Problemi Permessi**:
```bash
# Rendi eseguibile MusicGPT
chmod +x musicgpt-x86_64-unknown-linux-gnu

# Controlla appartenenza gruppo audio
groups $USER
# Aggiungi al gruppo audio se necessario:
sudo usermod -a -G audio $USER
```

**Problemi Prestazioni**:
- Riduci durata generazione nelle impostazioni
- Usa modello AI "Small" per generazione più veloce
- Chiudi altre applicazioni audio
- Controlla risorse sistema (uso CPU/RAM)

### Log e Debug
L'applicazione fornisce logging dettagliato nel pannello console. Messaggi log comuni:
- `Modalità bassa latenza attiva!`: PyAudio funziona correttamente
- `PyAudio non trovato`: Uso fallback ffplay
- `Loop perfetto!`: Rilevamento loop riuscito
- `Errore`: Controlla configurazione sistema

## Dettagli Algoritmi

### Algoritmo Avanzato (Multi-Metrico)
L'algoritmo avanzato analizza molteplici aspetti:
1. **Similarità Spettrale** (25%): Corrispondenza contenuto frequenza
2. **Continuità Forma d'Onda** (35%): Fluidità transizione ampiezza
3. **Allineamento Beat** (25%): Sincronizzazione timing musicale
4. **Continuità Fase** (15%): Ottimizzazione relazione fase

### Algoritmo Classico (Spettrale)
Usa analisi mel-spettrogramma per elaborazione più veloce con buoni risultati.

### Ottimizzazione Loop
- Rilevamento zero-crossing per transizioni senza click
- Rilevamento automatico tempo e beat
- Ottimizzazione crossfade
- Normalizzazione audio e controllo livelli

## Struttura Progetto
```
infiniloop/
├── il2.py                              # Applicazione principale
├── musicgpt-x86_64-unknown-linux-gnu   # Binario modello AI
├── artisti.txt                         # Nomi artista (opzionale)
├── nomi.txt                           # Parole titolo parte 1 (opzionale)  
├── nomi2.txt                          # Parole titolo parte 2 (opzionale)
├── music1.wav                         # File audio temporaneo
├── music2.wav                         # File audio temporaneo
└── README.md                          # Questo file
```

## Contribuire

1. Forka il repository
2. Crea un branch feature (`git checkout -b feature/amazing-feature`)
3. Committa le tue modifiche (`git commit -m 'Add amazing feature'`)
4. Pusha al branch (`git push origin feature/amazing-feature`)
5. Apri una Pull Request

## Licenza

Questo progetto è rilasciato sotto Licenza GPL - vedi il file [LICENSE](LICENSE) per dettagli.

## Ringraziamenti

- **librosa**: Libreria analisi audio
- **MusicGPT**: Modello generazione musicale AI
- **PyAudio**: Riproduzione audio bassa latenza
- **NumPy/SciPy**: Calcolo matematico

## Supporto

Per problemi, domande o contributi:
- Apri un issue su GitHub
- Controlla la sezione risoluzione problemi
- Rivedi i log della console per informazioni dettagliate sugli errori

---
