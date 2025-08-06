# INFINI LOOP - Infinite Local AI Music Generation

INFINI LOOP is an AI-powered music system designed to generate seamless, infinite audio loops. It automatically creates new musical fragments using AI, detects optimal loop points, and plays them continuously while preparing the next segment—resulting in a smooth, never-ending stream of always-fresh instrumental music.

At startup, one of two pre-included .wav files will play, so you can enjoy music immediately while the first AI generation is being prepared.

Once set up and running, your machine becomes a local AI music station, continuously producing new tracks with smooth transitions. It's local, private, and more personal than any YouTube or Spotify playlist.

INFINI LOOP is powered by MusicGPT (https://github.com/gabotechs/MusicGPT) and MusicGen models by Meta (https://huggingface.co/facebook/musicgen-medium). According to Meta, the model was trained on licensed data from the following sources: the Meta Music Initiative Sound Collection, Shutterstock music collection, and the Pond5 music collection. See the paper for more details about the training set and corresponding preprocessing.

All audio generated with INFINI LOOP is produced locally and never sent to any external service. The resulting music is entirely owned by the user, who retains full rights to use, distribute, modify, and commercialize it without restrictions.

GUI version:

<img width="500" height="881" alt="immagine" src="https://github.com/user-attachments/assets/1bcbf69c-a16e-47ec-afe1-2749a3fc2228" />

Terminal version:

<img width="500" alt="immagine" src="https://github.com/user-attachments/assets/9a95d2dd-8690-4d00-8735-530511ef9498" />

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

- **Local AI Music Generation**: Powered by MusicGPT for high-quality audio synthesis
- **Model Selection**: Choose between small (fast), medium (balanced), and large (slow/high quality) MusicGen models
- **Advanced Loop Detection**: Multi-metric analysis with configurable minimum loop length
- **Seamless Playback**: Native infinite looping with crossfade transitions
- **Flexible Generation**: Configurable AI sample duration (5-30s) and minimum loop requirements
- **Smart Loop Management**: Configurable minimum song duration before switching (10-300s)
- **Multiple Interfaces**: Terminal and GUI versions with real-time monitoring
- **Preset System**: Quick generation with pre-configured musical styles
- **Benchmark System**: Performance monitoring and generation time statistics
- **Export Functionality**: Save generated loops for later use
- **Settings Persistence**: Automatic save/restore of user preferences
- **Debug Mode**: Comprehensive logging and file state tracking

## Available Versions

### 1. GUI Version (`il1.py`) - Recommended ⭐

- Clean graphical interface with tabbed organization
- Same robust audio engine as terminal version
- Real crossfade transition support with visual feedback
- Model Selection: Choose between small, medium, and large models directly from the interface
- Sample Duration Control: Set AI generation length from 5-30 seconds
- Minimum Loop Length: Configure required loop length for acceptance
- Preset System for quick generation
- Statistics Tab: Benchmark data and generation time tracking
- Settings Tab: Configure all generation parameters and audio driver
- Loop Information: Display random titles, artists, duration, and genre
- Real-time status monitoring with timing information
- Settings persistence across sessions
- Save current loop functionality

### 2. Terminal Version (`ilterm.py`) - Most Stable ⭐

- Command-line interface with full interactive mode
- Real transition support between loops  
- Advanced audio validation and error recovery
- Debug mode with detailed logging
- Generation-only mode for single loops
- Complete settings configuration including model selection
- Lowest resource consumption
- Benchmark system for tracking generation performance

**Interactive Commands:**
- `start '<prompt>'` - Start infinite loop
- `stop` - Stop playback
- `status` - Show detailed system status with timing information
- `save <file.wav>` - Export current loop
- `set duration/minduration/minlength/model/driver` - Change settings
- `debug on/off` - Toggle debug mode
- `validate current/next/both` - Check file integrity
- `help` - Show all commands

## Recommended Requirements

- **Operating System**: Ubuntu 22.04 LTS
- **Python**: 3.10 or higher
- **RAM**: 16 GB system memory
- **CPU**: Modern high-frequency CPU (3+ GHz, 4+ cores)
- **Audio**: PulseAudio with low-latency configuration
- **Storage**: 8 GB or more, depending on the model 

### Performance Notes

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

## Usage

### Important: Model Download

MusicGPT will download the selected model on first use. The very first generation will be significantly slower than subsequent ones. **It is strongly recommended to download and test your desired model from the terminal first:**

```bash
# For small model (fast, good quality, less variety)
./musicgpt-x86_64-unknown-linux-gnu "Create a relaxing LoFi song" --model small

# For medium model (balanced quality and speed - recommended)
./musicgpt-x86_64-unknown-linux-gnu "Create a relaxing LoFi song" --model medium

# For large model (highest quality but very slow - not recommended)
./musicgpt-x86_64-unknown-linux-gnu "Create a relaxing LoFi song" --model large
```

This will download the required model and give you an idea of generation times before using INFINI LOOP.

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
python ilterm.py --prompt "electronic dance loop" --model medium

# Custom settings with all parameters
python ilterm.py --prompt "ambient chill" --duration 20 --model small --minlength 5

# Generate single loop and exit
python ilterm.py --generate-only "jazz piano" output.wav --model medium

# Debug mode
python ilterm.py --prompt "test loop" --verbose
```

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

## Technical Details

### Model Comparison & Recommendations

- **Small Model** ⭐ **RECOMMENDED**: 
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

For commercial licensing, please contact the authors.

## Credits

- **MusicGPT** by gabotechs for AI music generation: https://github.com/gabotechs/MusicGPT
- **librosa** team for audio analysis: https://librosa.org/
- **pyloudnorm** by csteinmetz1 for audio processing: https://github.com/csteinmetz1/pyloudnorm
- Developed with assistance from AI language models

---
**🎵 Enjoy infinite AI music with INFINI LOOP! 🎵**

---

# INFINI LOOP - Generazione Musicale AI Infinita e Locale

INFINI LOOP è un sistema musicale basato su AI progettato per generare loop audio infiniti e senza soluzione di continuità. Crea automaticamente nuovi frammenti musicali utilizzando l'AI, rileva i punti di loop ottimali e li riproduce continuamente mentre prepara il segmento successivo—risultando in un flusso fluido e infinito di musica strumentale sempre nuova.

All'avvio, uno dei due file .wav pre-inclusi verrà riprodotto, così potrai goderti la musica immediatamente mentre la prima generazione AI viene preparata.

Una volta configurato e avviato, la tua macchina diventa una stazione musicale AI locale, che produce continuamente nuove tracce con transizioni fluide. È locale, privata e più personale di qualsiasi playlist di YouTube o Spotify.

INFINI LOOP è alimentato da MusicGPT (https://github.com/gabotechs/MusicGPT) e dai modelli MusicGen di Meta (https://huggingface.co/facebook/musicgen-medium). Secondo Meta, il modello è stato addestrato su dati autorizzati dalle seguenti fonti: la Meta Music Initiative Sound Collection, la collezione musicale Shutterstock e la collezione musicale Pond5. Consulta il documento per maggiori dettagli sul set di addestramento e la relativa elaborazione.

Tutto l'audio generato con INFINI LOOP è prodotto localmente e non viene mai inviato a servizi esterni. La musica risultante è interamente di proprietà dell'utente, che mantiene tutti i diritti per utilizzare, distribuire, modificare e commercializzare senza restrizioni.

Versione GUI:

<img width="500" height="881" alt="immagine" src="https://github.com/user-attachments/assets/1bcbf69c-a16e-47ec-afe1-2749a3fc2228" />

Versione terminale:

<img width="500" alt="immagine" src="https://github.com/user-attachments/assets/9a95d2dd-8690-4d00-8735-530511ef9498" />

## Indice

- [Caratteristiche](#caratteristiche)
- [Versioni Disponibili](#versioni-disponibili)
- [Requisiti di Sistema](#requisiti-di-sistema)
- [Installazione](#installazione)
- [Utilizzo](#utilizzo)
- [Dettagli Tecnici](#dettagli-tecnici)
- [Risoluzione Problemi](#risoluzione-problemi)
- [Licenza](#licenza)

## Caratteristiche

- **Generazione Musicale AI Locale**: Alimentato da MusicGPT per sintesi audio di alta qualità
- **Selezione Modello**: Scegli tra modelli MusicGen piccolo (veloce), medio (bilanciato) e grande (lento/alta qualità)
- **Rilevamento Loop Avanzato**: Analisi multi-metrica con lunghezza minima loop configurabile
- **Riproduzione Fluida**: Loop infinito nativo con transizioni crossfade
- **Generazione Flessibile**: Durata campione AI configurabile (5-30s) e requisiti loop minimi
- **Gestione Loop Intelligente**: Durata minima brano configurabile prima del cambio (10-300s)
- **Interfacce Multiple**: Versioni terminale e GUI con monitoraggio in tempo reale
- **Sistema Preset**: Generazione rapida con stili musicali pre-configurati
- **Sistema Benchmark**: Monitoraggio prestazioni e statistiche tempi di generazione
- **Funzionalità Esportazione**: Salva i loop generati per uso successivo
- **Persistenza Impostazioni**: Salvataggio/ripristino automatico delle preferenze utente
- **Modalità Debug**: Logging completo e tracciamento stato file

## Versioni Disponibili

### 1. Versione GUI (`il1.py`) - Consigliata ⭐

- Interfaccia grafica pulita con organizzazione a schede
- Stesso motore audio robusto della versione terminale
- Supporto reale transizioni crossfade con feedback visuale
- Selezione Modello: Scegli tra modelli piccolo, medio e grande direttamente dall'interfaccia
- Controllo Durata Campione: Imposta la lunghezza generazione AI da 5-30 secondi
- Lunghezza Loop Minima: Configura la lunghezza loop richiesta per l'accettazione
- Sistema Preset per generazione rapida
- Scheda Statistiche: Dati benchmark e tracciamento tempi di generazione
- Scheda Impostazioni: Configura tutti i parametri di generazione e driver audio
- Informazioni Loop: Mostra titoli casuali, artisti, durata e genere
- Monitoraggio stato in tempo reale con informazioni temporali
- Persistenza impostazioni tra sessioni
- Funzionalità salvataggio loop corrente

### 2. Versione Terminale (`ilterm.py`) - Più Stabile ⭐

- Interfaccia riga di comando con modalità interattiva completa
- Supporto transizioni reali tra loop
- Validazione audio avanzata e recupero errori
- Modalità debug con logging dettagliato
- Modalità solo-generazione per loop singoli
- Configurazione impostazioni completa inclusa selezione modello
- Consumo risorse più basso
- Sistema benchmark per tracciamento prestazioni generazione

**Comandi Interattivi:**
- `start '<prompt>'` - Avvia loop infinito
- `stop` - Ferma riproduzione
- `status` - Mostra stato sistema dettagliato con informazioni temporali
- `save <file.wav>` - Esporta loop corrente
- `set duration/minduration/minlength/model/driver` - Cambia impostazioni
- `debug on/off` - Attiva/disattiva modalità debug
- `validate current/next/both` - Controlla integrità file
- `help` - Mostra tutti i comandi

## Requisiti di Sistema

### Requisiti Consigliati

- **Sistema Operativo**: Ubuntu 22.04 LTS
- **Python**: 3.10 o superiore
- **RAM**: 16 GB memoria sistema
- **CPU**: CPU moderna alta frequenza (3+ GHz, 4+ core)
- **Audio**: PulseAudio con configurazione bassa latenza
- **Archiviazione**: 8 GB spazio libero o più a seconda del modello

### Note Prestazioni

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
# Installa librerie elaborazione audio principali
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

# Rendi gli script eseguibili
chmod +x *.py
```

### Passo 4: Scarica e Configura Binario MusicGPT

1. Visita la [pagina release MusicGPT](https://github.com/gabotechs/MusicGPT/releases)
2. Scarica `musicgpt-x86_64-unknown-linux-gnu` (versione più recente)
3. Posizionalo nella stessa directory degli script Python
4. Rendilo eseguibile:

```bash
chmod +x musicgpt-x86_64-unknown-linux-gnu

# Verifica che il binario funzioni
./musicgpt-x86_64-unknown-linux-gnu --help
```

### Passo 5: Opzionale: Configura Audio Bassa Latenza

```bash
echo "default-sample-rate = 44100" >> ~/.pulse/daemon.conf
echo "alternate-sample-rate = 48000" >> ~/.pulse/daemon.conf
pulseaudio --kill && pulseaudio --start
```

## Utilizzo

### Importante: Download Modello

MusicGPT scaricherà il modello selezionato al primo utilizzo. La primissima generazione sarà significativamente più lenta delle successive. **È fortemente consigliato scaricare e testare il modello desiderato dal terminale prima:**

```bash
# Per modello piccolo (veloce, buona qualità, meno varietà)
./musicgpt-x86_64-unknown-linux-gnu "Create a relaxing LoFi song" --model small

# Per modello medio (qualità e velocità bilanciate - consigliato)
./musicgpt-x86_64-unknown-linux-gnu "Create a relaxing LoFi song" --model medium

# Per modello grande (qualità massima ma molto lento - non consigliato)
./musicgpt-x86_64-unknown-linux-gnu "Create a relaxing LoFi song" --model large
```

Questo scaricherà il modello richiesto e ti darà un'idea dei tempi di generazione prima di usare INFINI LOOP.

### Versione GUI (`il1.py`) - Consigliata

```bash
python il1.py
```

**Passi Utilizzo:**
1. **Inserisci Prompt**: Digita la tua descrizione musicale (es. "chitarra acustica calma")
2. **Scegli Modello**: Seleziona piccolo (veloce), medio (bilanciato), o grande (lento ma alta qualità)
3. **Imposta Durata Campione**: Configura lunghezza generazione AI (5-30 secondi)
4. **Imposta Lunghezza Loop Minima**: Imposta durata loop richiesta per accettazione
5. **Scegli Preset**: Clicca pulsanti preset per configurazioni rapide
6. **Configura Impostazioni**: 
   - **Durata Minima Brano**: Quanto ogni loop suona prima del cambio (10-300 secondi)
   - **Driver Audio**: pulse/alsa/dsp
7. **Avvia Generazione**: Clicca "▶️ AVVIA LOOP" 
8. **Monitora Progresso**: Osserva barra stato, informazioni loop e temporizzazione
9. **Visualizza Statistiche**: Controlla benchmark tempi generazione nella scheda Statistiche
10. **Salva Loop**: Usa "💾 Salva Loop Corrente" per esportare

### Versione Terminale (`ilterm.py`)

#### Avvio Rapido

```bash
# Modalità interattiva (consigliata)
python ilterm.py

# Generazione diretta con modello specifico
python ilterm.py --prompt "electronic dance loop" --model medium

# Impostazioni personalizzate con tutti i parametri
python ilterm.py --prompt "ambient chill" --duration 20 --model small --minlength 5

# Genera loop singolo ed esci
python ilterm.py --generate-only "jazz piano" output.wav --model medium

# Modalità debug
python ilterm.py --prompt "test loop" --verbose
```

#### Comandi Interattivi

```bash
🎛️ > start 'ambient electronic loop'     # Avvia loop infinito
🎛️ > stop                                # Ferma riproduzione
🎛️ > status                              # Mostra stato dettagliato
🎛️ > save my_loop.wav                   # Esporta loop corrente
🎛️ > set model                          # Cambia modello AI (small/medium/large)
🎛️ > set duration                        # Cambia lunghezza generazione campione (5-30s)
🎛️ > set minlength                       # Cambia lunghezza loop minima per accettazione
🎛️ > set minduration                     # Cambia durata minima brano (10-300s)
🎛️ > set driver                          # Cambia driver audio
🎛️ > debug on/off                        # Attiva/disattiva modalità debug
🎛️ > help                                # Mostra tutti i comandi
🎛️ > quit                                # Esci dal programma
```

#### Opzioni Riga di Comando

```bash
python ilterm.py [OPZIONI]

Opzioni:
  -p, --prompt TEXT        Prompt generazione musicale
  -m, --model [small|medium|large]  Selezione modello AI
  -d, --duration INTEGER   Durata generazione campione (5-30 secondi)
  --minlength INTEGER      Lunghezza loop minima per accettazione (secondi)
  --driver [pulse|alsa|dsp]  Selezione driver audio
  -v, --verbose            Output dettagliato
  -h, --help               Mostra messaggio aiuto
```

## Dettagli Tecnici

### Confronto Modelli e Raccomandazioni

- **Modello Piccolo** ⭐ **CONSIGLIATO**: 
  - Tempo generazione: Veloce (tipicamente 15-30 secondi)
  - Qualità: Buona fedeltà audio
  - Varietà: Meno complessità musicale
  - Migliore per: Test rapidi, loop più lunghi, iterazione veloce

- **Modello Medio** ⭐⭐ **CONSIGLIATO**:
  - Tempo generazione: Moderato (tipicamente 30-60 secondi)
  - Qualità: Migliore fedeltà audio con buona struttura musicale
  - Varietà: Eccellente equilibrio di creatività e coerenza
  - Migliore per: Uso generale, prestazioni bilanciate

- **Modello Grande**:
  - Tempo generazione: Molto lento (tipicamente 2-5 minuti)
  - Qualità: Alta fedeltà e complessità musicale
  - Varietà: Massima diversità creativa
  - Migliore per: Solo progetti speciali - **non consigliato per uso regolare**

## Risoluzione Problemi

### Problemi Comuni e Soluzioni

#### Problemi Generazione

**Problema**: Generazione molto lenta
- **Causa**: Selezione modello sbagliata o risorse sistema insufficienti
- **Soluzioni**:
  - Prova modello piccolo per generazione più veloce
  - Testa tempi generazione dal terminale prima usando il comando fornito nella sezione Utilizzo
  - Usa durata più breve (8-12 secondi)
  - Chiudi altre applicazioni

**Problema**: "No interesting loop" o qualità scadente
- **Causa**: Audio generato AI non adatto per loop o lunghezza loop minima troppo restrittiva
- **Soluzioni**:
  - Aumenta durata campione AI a 15-20 secondi
  - Abbassa requisito lunghezza loop minima (prova 2.0-3.0 secondi)
  - Prova modelli diversi (medio spesso funziona meglio del piccolo per prompt complessi)
  - Usa pulsanti preset per prompt provati
  - Aggiungi parole chiave: "seamless", "loopable", "continuous"

#### Problemi Riproduzione Audio

**Problema**: Nessun output audio
- **Soluzioni**:
  ```bash
  # Testa sistema audio
  speaker-test -t wav -c 2
  
  # Prova driver diversi nelle impostazioni
  # GUI: scheda Impostazioni → Driver Audio
  # Terminale: set driver
  ```

**Problema**: Audio balbettante
- **Causa**: Problemi buffer audio o driver
- **Soluzioni**:
  - Prova driver audio diversi (pulse → alsa → dsp)
  - Chiudi applicazioni non necessarie
  - Controlla uso risorse sistema

### Ottimizzazione Prestazioni

#### Per Generazione Rapida:
- Usa **modello piccolo** con durata 10-20 secondi
- Imposta lunghezza loop minima a 3-5 secondi
- Usa durata minima brano più breve (20-30 secondi)

#### Per Più Varietà:
- Usa **modello medio** con durata 10-15 secondi
- Imposta lunghezza loop minima a 2.5-4 secondi
- Usa durata minima brano più lunga (60-120 secondi)

Testa sempre il modello scelto dal terminale prima per capire i tempi di generazione:
```bash
./musicgpt-x86_64-unknown-linux-gnu "test prompt" --model your_chosen_model
```
Oppure usa la funzionalità statistiche per registrare i tempi di generazione.

### Creazione Preset Personalizzati

Modifica preset in `il1.py`:

```python
presets = {
    "Jazz Personalizzato": "smooth jazz piano trio seamless loop",
    "Cinema Epico": "cinematic orchestral epic soundtrack loop",
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

Per licenze commerciali, contatta gli autori.

## Crediti

- **MusicGPT** di gabotechs per generazione musicale AI: https://github.com/gabotechs/MusicGPT
- Team **librosa** per analisi audio: https://librosa.org/
- **pyloudnorm** di csteinmetz1 per elaborazione audio: https://github.com/csteinmetz1/pyloudnorm
- Sviluppato con assistenza di modelli linguistici AI

---
**🎵 Goditi la musica AI infinita con INFINI LOOP! 🎵**
