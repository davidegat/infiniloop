# INFINILOOP

Infinite local music generation with AI + loop analysis and audio visualization
**Designed for seamless creative performance.**

---

## Description

**INFINILOOP** is a Python application that combines **local AI-based music generation** with **advanced loop detection** to create seamless musical loops.
It uses [MusicGPT](https://github.com/gabotechs/MusicGPT) to generate WAV files from text prompts and analyzes them to extract high-quality audio loops with rhythmic and spectral continuity.

Includes a fullscreen graphical interface with:

* Compact and intuitive controls
* Real-time waveform, spectrum, and loop metrics
* Continuous audio playback with smart crossfade
* Manual export of current loop
* Advanced settings: model selection, duration, audio backend, and analysis mode

---

## Requirements

Python **>= 3.8**
Install the required libraries:

```bash
pip install numpy scipy librosa soundfile matplotlib pydub pillow
```

**System dependencies (Linux):**

```bash
sudo apt install ffmpeg
sudo apt install libasound2-dev portaudio19-dev
```

> Requires `ffplay` (included in FFmpeg) for playback.

---

## MusicGPT

This software integrates with [MusicGPT](https://github.com/gabotechs/MusicGPT) for AI music generation.
For this project, I used a precompiled binary, but you are free to build or run it in any way supported by the original project.

Please refer to the official MusicGPT repository for instructions:

* [https://github.com/gabotechs/MusicGPT](https://github.com/gabotechs/MusicGPT)

---

## Running the App

```bash
python il2.py
```

---

## Using the GUI

### Start generation

1. Enter a text prompt (e.g., `lofi hiphop nointro loop`)
2. Select the algorithm:

   * Advanced (multi-metric)
   * Classic (spectral only)
3. Click "AVVIA" to start

### Infinite loop

* Two audio fragments are generated and alternated
* Loops are automatically analyzed and trimmed
* Second loop overlaps slightly with the end of the first to ensure seamless playback

### Visualization

* Waveform, spectrum, and analysis metrics are shown
* Loop start/end markers are displayed
* Loop score and measures are evaluated

### Info and export

* Random title and artist are generated from local wordlists
* Click "SALVA" to export the current loop

### Advanced settings

* AI model: small, medium, large
* Duration: from 5 to 30 seconds
* Audio driver: pulse, alsa, oss

---

## Loop Detection Engine

Advanced mode evaluates:

* Spectral similarity
* Waveform continuity
* Beat alignment
* Phase alignment (STFT)

The best loop is trimmed, faded, and saved.

---

## File Structure

* `il2.py` – main GUI and processing logic
* `nomi.txt`, `nomi2.txt` – wordlists for title generation
* `artisti.txt` – list of fictional artist names

---

## Notes

* Press `ESC` to exit fullscreen
* Crossfade duration is adjustable (1–5000 ms)
* Playback uses `ffplay` with selectable audio backend

---

## License

This project is licensed under the **GNU GPL v3.0 – Non-Commercial Use Only**.
See the full license text here: [https://www.gnu.org/licenses/gpl-3.0.html](https://www.gnu.org/licenses/gpl-3.0.html)

---

# INFINILOOP

Generazione musicale infinita in locale con AI + analisi dei loop + visualizzazione audio
**Pensato per performance creative senza interruzioni.**

---

## Descrizione

**INFINILOOP** è un'applicazione Python che unisce la **generazione musicale AI locale** con l'**analisi avanzata dei loop audio** per ottenere frammenti musicali perfettamente riproducibili in ciclo continuo.

Utilizza [MusicGPT](https://github.com/gabotechs/MusicGPT) per generare musica a partire da prompt testuali.
Nel mio caso ho usato un binario precompilato, ma l'utente è libero di utilizzare qualsiasi metodo previsto dal progetto ufficiale.

---

## Requisiti

Python **>= 3.8**
Installa le librerie necessarie:

```bash
pip install numpy scipy librosa soundfile matplotlib pydub pillow
```

**Dipendenze di sistema (Linux):**

```bash
sudo apt install ffmpeg
sudo apt install libasound2-dev portaudio19-dev
```

> Richiede `ffplay` per la riproduzione (incluso in `ffmpeg`).

---

## Avvio dell'applicazione

```bash
python il2.py
```

---

## Uso dell'interfaccia

### Avvia generazione

1. Inserisci un prompt musicale (es. `lofi hiphop nointro loop`)
2. Seleziona l'algoritmo:

   * Avanzato (multi-metrico)
   * Classico (solo spettrale)
3. Premi "AVVIA"

### Loop infinito

* Vengono generati due brani in alternanza
* I loop vengono analizzati e rifilati automaticamente
* Il secondo brano parte leggermente prima della fine del primo per una transizione fluida

### Visualizzazioni

* Forma d'onda e spettro medio
* Indicatori di inizio/fine loop
* Punteggio qualitativo del loop mostrato a video

### Info e salvataggio

* Titolo e artista sono generati da file `nomi.txt`, `nomi2.txt`, `artisti.txt`
* Premi "SALVA" per esportare il file WAV corrente

### Impostazioni avanzate

* Modello AI: small, medium, large
* Durata generazione: da 5 a 30 secondi
* Driver audio: pulse, alsa, oss

---

## Motore di analisi loop

La modalità avanzata valuta:

* Similarità spettrale
* Continuità della forma d'onda
* Allineamento ai battiti
* Continuità di fase (STFT)

Il miglior loop viene rifinito con fade-in/fade-out e salvato.

---

## File del progetto

* `il2.py` – logica principale e interfaccia
* `nomi.txt`, `nomi2.txt` – liste per nomi casuali
* `artisti.txt` – lista nomi artista improbabili

---

## Note aggiuntive

* Premi `ESC` per uscire dalla modalità schermo intero
* Durata del crossfade regolabile da 1 a 5000 ms
* Riproduzione con `ffplay` e driver selezionabile

---

## Licenza

Questo progetto è rilasciato sotto **GNU GPL v3.0 - Solo uso non commerciale**.
Testo completo della licenza: [https://www.gnu.org/licenses/gpl-3.0.html](https://www.gnu.org/licenses/gpl-3.0.html)
