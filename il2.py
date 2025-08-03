import os
import time
import subprocess
import shutil
import threading
import queue
import random
from datetime import datetime
import numpy as np
import soundfile as sf
import librosa
import librosa.display
from pydub.utils import mediainfo
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class InfiniLoopGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("INFINI LOOP - by gat")
        self.root.configure(bg='#0a0a0a')
        self.root.resizable(True, True)

        self.file_lock = threading.Lock()
        self.swap_lock = threading.Lock()
        self.FILE1, self.FILE2 = "./music1.wav", "./music2.wav"
        self.CURRENT, self.NEXT = self.FILE1, self.FILE2

        self.CROSSFADE_MS = 2000
        self.CROSSFADE_SEC = self.CROSSFADE_MS / 1000

        self.PROMPT = ""

        self.use_advanced_algorithm = tk.BooleanVar(value=True)
        self.model_var = tk.StringVar(value="medium")
        self.duration_var = tk.IntVar(value=15)
        self.audio_driver_var = tk.StringVar(value="pulse")

        self.is_playing = False
        self.stop_event = threading.Event()
        self.loop_thread = None
        self.message_queue = queue.Queue()
        self.setup_styles()
        self.create_gui()
        self.process_messages()
        self._check_audio_setup()


    def _check_audio_setup(self):
        try:
            import pyaudio
            self._has_pyaudio = True
            self.log_message("✅ Modalità bassa latenza attiva!")
        except ImportError:
            self._has_pyaudio = False
            self.log_message("⚠️ PyAudio non trovato - fallback su ffplay\n💡 Suggerito: pip install pyaudio")


    def _loop_with_pyaudio(self, audio_data, sr, stop_event):
        import pyaudio, time, numpy as np
        chunk = 512
        dur = chunk / sr
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32, channels=1, rate=sr, output=True, frames_per_buffer=chunk)
        chunks = [np.pad(audio_data[i:i+chunk], (0, chunk - len(audio_data[i:i+chunk])), 'constant').astype(np.float32).tobytes()
                for i in range(0, len(audio_data), chunk)]
        t0, i = time.perf_counter(), 0
        try:
            while not stop_event.is_set() and self.is_playing:
                stream.write(chunks[i % len(chunks)])
                i = (i + 1) % len(chunks)
                sleep = t0 + i * dur - time.perf_counter()
                if sleep > 0: time.sleep(min(sleep, dur * 0.1))
        finally:
            stream.stop_stream(), stream.close(), p.terminate()


    def _loop_with_ffplay_optimized(self, audio_data, sr, stop_event, original_filepath):
        import os, soundfile as sf
        tmp = "/tmp/infini_loop_optimized.wav"
        sf.write(tmp, audio_data, sr)
        try:
            while not stop_event.is_set() and self.is_playing:
                self.play_with_ffplay(tmp)
                if stop_event.is_set(): break
        finally:
            try: os.remove(tmp)
            except: pass


    def setup_styles(self):

        style = ttk.Style()
        style.theme_use('clam')


        style.configure('Title.TLabel',
                       background='#0a0a0a',
                       foreground='#00ff88',
                       font=('Courier New', 18, 'bold'))

        style.configure('Subtitle.TLabel',
                       background='#0a0a0a',
                       foreground='#ffffff',
                       font=('Courier New', 10))

        style.configure('Status.TLabel',
                       background='#1a1a1a',
                       foreground='#00ff88',
                       font=('Courier New', 9, 'bold'),
                       relief='solid',
                       borderwidth=1)

        style.configure('Custom.TButton',
                       background='#2a2a2a',
                       foreground='#00ff88',
                       font=('Courier New', 9, 'bold'),
                       borderwidth=2,
                       focuscolor='none')

        style.map('Custom.TButton',
                 background=[('active', '#3a3a3a'), ('pressed', '#1a1a1a')])

        style.configure('Custom.TEntry',
                       fieldbackground='#2a2a2a',
                       foreground='#ffffff',
                       font=('Courier New', 10),
                       borderwidth=2,
                       insertcolor='#00ff88')


    def create_gui(self):

        main_frame = tk.Frame(self.root, bg='#0a0a0a')
        main_frame.pack(fill='both', expand=True, padx=15, pady=10)

        self.create_compact_header(main_frame)

        main_horizontal_frame = tk.Frame(main_frame, bg='#0a0a0a')
        main_horizontal_frame.pack(fill='both', expand=True, pady=(0, 10))

        left_panel = tk.Frame(main_horizontal_frame, bg='#0a0a0a')
        left_panel.pack(side='left', fill='y', padx=(0, 10))

        right_panel = tk.Frame(main_horizontal_frame, bg='#0a0a0a')
        right_panel.pack(side='right', fill='both', expand=True)

        self.create_compact_control_panel(left_panel)
        self.create_compact_visualization_area(right_panel)
        self.create_compact_status_area(left_panel)
        self.create_compact_log_area(right_panel)


    def create_compact_header(self, parent):

        header_frame = tk.Frame(parent, bg='#0a0a0a')
        header_frame.pack(fill='x', pady=(0, 10))

        title_label = ttk.Label(header_frame,
                               text="I N F I N I L O O P",
                               style='Title.TLabel')
        title_label.pack()

        subtitle_label = ttk.Label(header_frame,
                                  text="🎵 Infinite AI Music Generation • Loop Detection • by gat 🎵",
                                  style='Subtitle.TLabel')
        subtitle_label.pack(pady=(2, 0))


        separator = tk.Frame(header_frame, height=1, bg='#00ff88')
        separator.pack(fill='x', pady=(5, 0))


    def create_compact_control_panel(self, parent):

        control_frame = tk.LabelFrame(parent,
                                     text=" 🎛️ CONTROLLI ",
                                     bg='#1a1a1a',
                                     fg='#00ff88',
                                     font=('Courier New', 10, 'bold'),
                                     bd=2,
                                     relief='solid')
        control_frame.pack(fill='both', pady=(0, 10), ipady=5)


        prompt_frame = tk.Frame(control_frame, bg='#1a1a1a')
        prompt_frame.pack(fill='x', padx=10, pady=8)

        tk.Label(prompt_frame,
                text="🎼 PROMPT:",
                bg='#1a1a1a',
                fg='#ffffff',
                font=('Courier New', 9, 'bold')).pack(anchor='w')

        self.prompt_entry = ttk.Entry(prompt_frame,
                                     style='Custom.TEntry',
                                     font=('Courier New', 10),
                                     width=40)
        self.prompt_entry.pack(fill='x', pady=(3, 0))
        self.prompt_entry.insert(0, "lofi calm rap beat")


        algorithm_frame = tk.Frame(control_frame, bg='#1a1a1a')
        algorithm_frame.pack(fill='x', padx=10, pady=(0, 8))

        tk.Label(algorithm_frame,
                text="🧠 ALGORITMO:",
                bg='#1a1a1a',
                fg='#ffffff',
                font=('Courier New', 9, 'bold')).pack(anchor='w')

        algo_options_frame = tk.Frame(algorithm_frame, bg='#1a1a1a')
        algo_options_frame.pack(fill='x', pady=(3, 0))

        tk.Radiobutton(algo_options_frame,
                      text="🎯 Avanzato",
                      variable=self.use_advanced_algorithm,
                      value=True,
                      bg='#1a1a1a',
                      fg='#00ff88',
                      selectcolor='#2a2a2a',
                      activebackground='#2a2a2a',
                      activeforeground='#00ff88',
                      font=('Courier New', 8, 'bold')).pack(anchor='w')

        tk.Radiobutton(algo_options_frame,
                      text="📊 Classico",
                      variable=self.use_advanced_algorithm,
                      value=False,
                      bg='#1a1a1a',
                      fg='#ffffff',
                      selectcolor='#2a2a2a',
                      activebackground='#2a2a2a',
                      activeforeground='#ffffff',
                      font=('Courier New', 8)).pack(anchor='w', pady=(2, 0))

        buttons_frame = tk.Frame(control_frame, bg='#1a1a1a')
        buttons_frame.pack(fill='x', padx=10, pady=(0, 8))

        self.start_button = ttk.Button(buttons_frame,
                                      text="🚀 AVVIA",
                                      style='Custom.TButton',
                                      command=self.toggle_loop)
        self.start_button.pack(fill='x', pady=(0, 3))

        button_row = tk.Frame(buttons_frame, bg='#1a1a1a')
        button_row.pack(fill='x')

        settings_button = ttk.Button(button_row,
                                   text="⚙️ SETUP",
                                   style='Custom.TButton',
                                   command=self.open_settings)
        settings_button.pack(side='left', fill='x', expand=True, padx=(0, 3))

        save_button = ttk.Button(button_row,
                                text="💾 SALVA",
                                style='Custom.TButton',
                                command=self.save_current_loop)
        save_button.pack(side='right', fill='x', expand=True)

        crossfade_frame = tk.Frame(control_frame, bg='#1a1a1a')
        crossfade_frame.pack(fill='x', padx=10, pady=(0, 8))

        tk.Label(crossfade_frame,
                text="🔀 OVERLAP (ms):",
                bg='#1a1a1a',
                fg='#ffffff',
                font=('Courier New', 8)).pack(anchor='w')

        self.crossfade_var = tk.IntVar(value=self.CROSSFADE_MS)
        self.crossfade_scale = tk.Scale(crossfade_frame,
                                       from_=1, to=5000,
                                       orient='horizontal',
                                       variable=self.crossfade_var,
                                       bg='#2a2a2a',
                                       fg='#00ff88',
                                       activebackground='#3a3a3a',
                                       highlightbackground='#1a1a1a',
                                       font=('Courier New', 8),
                                       length=200,
                                       command=self.update_crossfade)

        self.crossfade_scale.pack(fill='x', pady=(2, 0))

    def create_compact_visualization_area(self, parent):

        viz_frame = tk.LabelFrame(parent,
                                 text=" 📊 VISUALIZZAZIONE ",
                                 bg='#1a1a1a',
                                 fg='#00ff88',
                                 font=('Courier New', 10, 'bold'),
                                 bd=2,
                                 relief='solid')
        viz_frame.pack(fill='both', expand=True, pady=(0, 10))

        self.fig = Figure(figsize=(8, 4), facecolor='#1a1a1a')

        gs = self.fig.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.4, wspace=0.3)
        self.ax_wave = self.fig.add_subplot(gs[0, :], facecolor='#0a0a0a')
        self.ax_spectrum = self.fig.add_subplot(gs[1, 0], facecolor='#0a0a0a')
        self.ax_analysis = self.fig.add_subplot(gs[1, 1], facecolor='#0a0a0a')

        for ax in [self.ax_wave, self.ax_spectrum, self.ax_analysis]:
            ax.set_facecolor('#0a0a0a')
            ax.tick_params(colors='#00ff88', labelsize=6)
            for spine in ax.spines.values():
                spine.set_color('#00ff88')

        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)

        self.plot_welcome_screen()


    def create_compact_status_area(self, parent):

        status_frame = tk.Frame(parent, bg='#0a0a0a')
        status_frame.pack(fill='x', pady=(0, 10))

        self.current_track_frame = tk.LabelFrame(status_frame,
                                               text=" 🎧 BRANO ",
                                               bg='#1a1a1a',
                                               fg='#00ff88',
                                               font=('Courier New', 9, 'bold'),
                                               bd=1,
                                               relief='solid')
        self.current_track_frame.pack(fill='x', pady=(0, 5))

        self.track_info_label = tk.Label(self.current_track_frame,
                                        text="Nessun brano",
                                        bg='#1a1a1a',
                                        fg='#ffffff',
                                        font=('Courier New', 8),
                                        justify='left',
                                        wraplength=250)
        self.track_info_label.pack(padx=8, pady=5, anchor='w')


        self.generation_frame = tk.LabelFrame(status_frame,
                                            text=" 🛠️ STATO ",
                                            bg='#1a1a1a',
                                            fg='#00ff88',
                                            font=('Courier New', 9, 'bold'),
                                            bd=1,
                                            relief='solid')
        self.generation_frame.pack(fill='x')

        self.generation_status_label = tk.Label(self.generation_frame,
                                              text="In attesa...",
                                              bg='#1a1a1a',
                                              fg='#ffffff',
                                              font=('Courier New', 8),
                                              wraplength=250)
        self.generation_status_label.pack(padx=8, pady=5)


    def create_compact_log_area(self, parent):

        log_frame = tk.LabelFrame(parent,
                                 text=" 📝 CONSOLE ",
                                 bg='#1a1a1a',
                                 fg='#00ff88',
                                 font=('Courier New', 9, 'bold'),
                                 bd=2,
                                 relief='solid')
        log_frame.pack(fill='both', expand=True)


        log_inner_frame = tk.Frame(log_frame, bg='#1a1a1a')
        log_inner_frame.pack(fill='both', expand=True, padx=5, pady=5)

        self.log_text = tk.Text(log_inner_frame,
                               height=6,
                               bg='#0a0a0a',
                               fg='#00ff88',
                               font=('Courier New', 8),
                               wrap='word',
                               state='disabled')

        log_scrollbar = tk.Scrollbar(log_inner_frame, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=log_scrollbar.set)

        self.log_text.pack(side='left', fill='both', expand=True)
        log_scrollbar.pack(side='right', fill='y')

        self.log_message("🎵 INFINI LOOP ADVANCED inizializzato")
        self.log_message("💡 Inserisci prompt e premi AVVIA")


    def plot_welcome_screen(self):

        for ax in [self.ax_wave, self.ax_spectrum, self.ax_analysis]:
            ax.clear()
            ax.set_facecolor('#0a0a0a')

        x = np.linspace(0, 10, 1000)
        y1 = np.sin(x) * np.exp(-x/10)
        y2 = np.sin(x * 2) * np.exp(-x/10) * 0.5

        self.ax_wave.set_title('Taglio campione', color='#00ff88', fontsize=9, fontweight='bold')
        self.ax_wave.set_xlabel('Tempo (s)', color='#00ff88', fontsize=8)
        self.ax_wave.set_ylabel('Ampiezza', color='#00ff88', fontsize=8)
        self.ax_wave.legend(facecolor='#1a1a1a', edgecolor='#00ff88', labelcolor='#ffffff', fontsize=7)
        self.ax_wave.grid(True, alpha=0.3, color='#00ff88')

        freqs = np.linspace(0, 22050, 100)
        spectrum = np.exp(-freqs/5000) * np.random.normal(1, 0.1, 100)
        self.ax_spectrum.plot(freqs, spectrum, color='#00ff88', linewidth=1)
        self.ax_spectrum.set_title('Spettro', color='#00ff88', fontsize=8, fontweight='bold')
        self.ax_spectrum.set_xlabel('Frequenza (Hz)', color='#00ff88', fontsize=7)
        self.ax_spectrum.set_ylabel('Volume (dB)', color='#00ff88', fontsize=7)
        self.ax_spectrum.grid(True, alpha=0.3, color='#00ff88')

        metrics = ['Spett', 'Wave', 'Beat', 'Fase']
        scores = [0.85, 0.72, 0.91, 0.68]
        colors = ['#00ff88', '#ff8800', '#0088ff', '#ff0088']

        bars = self.ax_analysis.bar(metrics, scores, color=colors, alpha=0.7)
        self.ax_analysis.set_title('Metriche', color='#00ff88', fontsize=8, fontweight='bold')
        self.ax_analysis.set_ylabel('Score', color='#00ff88', fontsize=7)
        self.ax_analysis.set_ylim(0, 1)
        self.ax_analysis.grid(True, alpha=0.3, color='#00ff88')

        #self.fig.tight_layout(pad=1.0)
        self.canvas.draw()


    def plot_waveform_advanced(self, filepath, loop_info=None):
        try:
            y, sr = librosa.load(filepath, sr=None, mono=True)

            for ax in [self.ax_wave, self.ax_spectrum, self.ax_analysis]:
                ax.clear()

            # WAVEFORM
            time_axis = np.linspace(0, len(y)/sr, len(y))
            self.ax_wave.plot(time_axis, y, color='#00ff88', linewidth=1, alpha=0.8)

            if loop_info:
                start_time = loop_info['start_sample'] / sr
                end_time = loop_info['end_sample'] / sr
                self.ax_wave.axvspan(start_time, end_time, alpha=0.3, color='#ff0088')
                self.ax_wave.axvline(start_time, color='#00ff88', linestyle='--', alpha=0.8)
                self.ax_wave.axvline(end_time, color='#ff0088', linestyle='--', alpha=0.8)

            measures = loop_info.get('measures', '?') if loop_info else '?'
            self.ax_wave.set_title(f'Analisi sample - {measures}mis',
                                color='#00ff88', fontsize=9, fontweight='bold')
            self.ax_wave.set_xlabel('Tempo (s)', color='#00ff88', fontsize=8)
            self.ax_wave.set_ylabel('Ampiezza', color='#00ff88', fontsize=8)
            self.ax_wave.legend(facecolor='#1a1a1a', edgecolor='#00ff88', labelcolor='#ffffff', fontsize=6)
            self.ax_wave.grid(True, alpha=0.3, color='#00ff88')

            # SPETTRO
            S = librosa.feature.melspectrogram(y=y, sr=sr)
            S_dB = librosa.power_to_db(S, ref=np.max)
            mean_spectrum = np.mean(S_dB, axis=1)
            mel_freqs = librosa.mel_frequencies(n_mels=len(mean_spectrum))

            self.ax_spectrum.plot(mel_freqs, mean_spectrum, color='#00ff88', linewidth=1)
            self.ax_spectrum.set_title('Spettro Mel', color='#00ff88', fontsize=8, fontweight='bold')
            self.ax_spectrum.set_xlabel('Frequenza (Hz)', color='#00ff88', fontsize=7)
            self.ax_spectrum.set_ylabel('Volume (dB)', color='#00ff88', fontsize=7)
            self.ax_spectrum.grid(True, alpha=0.3, color='#00ff88')

            # METRICHE
            if loop_info and 'metrics' in loop_info:
                metric_names = {'spettrale': 'Spett', 'waveform': 'Wave', 'beat align': 'Beat', 'fase': 'Fase'}
                metrics = [metric_names.get(k.lower(), k) for k in loop_info['metrics'].keys()]
                scores = list(loop_info['metrics'].values())
                colors = ['#00ff88', '#ff8800', '#0088ff', '#ff0088'][:len(metrics)]

                bars = self.ax_analysis.bar(metrics, scores, color=colors, alpha=0.7)
                score_text = f"Punteggio: {loop_info.get('total_score', 0):.2f}"
                self.ax_analysis.set_title(score_text, color='#00ff88', fontsize=8, fontweight='bold')
                self.ax_analysis.set_ylabel('Score', color='#00ff88', fontsize=7)
                self.ax_analysis.set_ylim(0, 1)
                self.ax_analysis.grid(True, alpha=0.3, color='#00ff88')

                for bar, score in zip(bars, scores):
                    height = bar.get_height()
                    self.ax_analysis.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                        f'{score:.1f}', ha='center', va='bottom',
                                        color='#ffffff', fontsize=6)
            self.canvas.draw()

        except Exception as e:
            self.log_message(f"❌ Errore visualizzazione: {str(e)}")


    def find_optimal_zero_crossing(self, y, sample, window_size=256):
        s, e = max(0, sample - window_size // 2), min(len(y), sample + window_size // 2)
        slope = np.sign(np.mean(np.diff(y[max(0, sample - window_size//4):sample + window_size//4])))
        best, score = sample, float('inf')
        for i in range(s, e - 1):
            if y[i] == 0 or np.sign(y[i]) != np.sign(y[i+1]):
                amp = abs(y[i]) + abs(y[i+1])
                der = abs(y[i+1] - y[i])
                penalty = 0 if np.sign(y[i+1] - y[i]) == slope else 5.0
                sc = amp + 1.2 * der + penalty
                if sc < score: best, score = i, sc
        return best if best != sample else s + np.argmin(np.abs(y[s:e]))


    def calculate_waveform_continuity(self, y, start, end, sr):
        import numpy as np
        t = max(64, min(sr // 40, (end - start) // 20))
        a, b = y[max(0, end - t):end], y[start:start + t]
        if len(a) == 0 or len(b) == 0: return 0.0
        m = min(len(a), len(b)); a, b = a[-m:], b[:m]
        c = np.corrcoef(a, b)[0,1]; c = 0.0 if np.isnan(c) else abs(c)
        r1, r2 = np.sqrt(np.mean(a**2)), np.sqrt(np.mean(b**2))
        rms_sim = 1 - min(1, np.sqrt(np.mean((a - b)**2)) / max(r1, r2, 1e-8))
        if len(a) > 1:
            d1, d2 = np.diff(a)[-1], np.diff(b)[0]
            d_cont = 1 - min(1, abs(d1 - d2) / max(abs(d1), abs(d2), 1e-8))
        else: d_cont = 1.0
        return c * 0.4 + rms_sim * 0.4 + d_cont * 0.2


    def calculate_beat_alignment(self, start, end, beats, sr):
        if not len(beats): return 0.5
        d = np.abs(beats - start), np.abs(beats - end)
        c = np.min(d[0]), np.min(d[1])
        if len(beats) > 1:
            b = np.mean(np.diff(beats)) * 0.5
            a = [1 - min(1, x / b) for x in c]
        else: a = [0.5, 0.5]
        return sum(a) / 2


    def calculate_phase_continuity(self, S, start, end, w=3):
        if start < w or end >= S.shape[1] - w: return 0.5
        a, b = np.angle(S[:, start - w:start + w]), np.angle(S[:, end - w:end + w])
        d = np.abs(np.mean(a, axis=1) - np.mean(b, axis=1))
        d = np.minimum(d, 2*np.pi - d)
        return max(0, 1 - np.mean(d) / np.pi)


    def trim_with_improved_loop_detection(self, filepath):
        try:
            self.message_queue.put(("log", "🔍 Analisi avanzata loop detection..."))
            y, sr = librosa.load(filepath, sr=None, mono=True)
            y_loop, loop_info = None, None

            try:
                if self.use_advanced_algorithm.get():
                    loop_info = self.find_perfect_loop_advanced(y, sr)
                    s, e = loop_info['start_sample'], loop_info['end_sample']
                    self.message_queue.put(("log", "🎯 Ottimizzazione zero-crossing..."))
                    s, e = self.find_optimal_zero_crossing(y, s), self.find_optimal_zero_crossing(y, e)
                    loop_info.update({'start_sample': s, 'end_sample': e})
                    y_loop = y[s:e]
                else:
                    self.message_queue.put(("log", "📊 Uso algoritmo classico"))
                    loop_info = self.trim_with_spectral_similarity_classic(y, sr)
                    y_loop = y[loop_info['start_sample']:loop_info['end_sample']]
            except Exception as err:
                self.message_queue.put(("log", f"⚠️ Algoritmi falliti: {err}, uso fallback sicuro"))
                s, e = len(y)//4, len(y) - len(y)//4
                y_loop = y[s:e]
                loop_info = {'start_sample': s, 'end_sample': e, 'score': 0.5, 'measures': 'fallback', 'metrics': {'Fallback': 0.5}, 'total_score': 0.5}

            if y_loop is None or len(y_loop) < sr * 2:
                self.message_queue.put(("log", "⚠️ Loop troppo corto, estendo"))
                q = len(y) // 4
                y_loop = y[q:len(y)-q] if len(y) > sr * 2 else y

            if y_loop is None or len(y_loop) == 0 or np.all(np.abs(y_loop) < 1e-8):
                self.message_queue.put(("log", "❌ Loop vuoto o silenzioso, uso file completo"))
                y_loop = y
                loop_info = loop_info or {'start_sample': 0, 'end_sample': len(y), 'score': 0.3, 'measures': 'full', 'metrics': {'Full': 0.3}, 'total_score': 0.3}
                loop_info.update({'start_sample': 0, 'end_sample': len(y)})

            f = min(64, len(y_loop)//200)
            if f > 0 and len(y_loop) > f * 4:
                w = np.linspace(0, np.pi/2, f)
                fade_in, fade_out = np.sin(w)**2, np.cos(w)**2
                a, b = y_loop[:f].copy(), y_loop[-f:].copy()
                y_loop[:f] = a * fade_in + b * (1 - fade_in)
                y_loop[-f:] = b * fade_out + a * (1 - fade_out)

            sf.write(filepath + ".original.wav", y, sr)
            self.message_queue.put(("waveform_advanced", (filepath + ".original.wav", loop_info)))
            sf.write(filepath, y_loop, sr)

            d = len(y_loop) / sr
            self.message_queue.put(("log", f"🧬 Loop perfetto! {loop_info.get('measures','?')} mis, {d:.1f}s"))

        except Exception as e:
            self.message_queue.put(("error", f"❌ Errore loop detection: {e}"))


    def trim_with_spectral_similarity_classic(self, y, sr):
        try:
            S_dB = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512), ref=np.max)
            min_frames = int(7 * sr / 512)
            best_score, best_start, best_end = -np.inf, 0, min_frames

            for i in range(S_dB.shape[1] - min_frames):
                for j in range(i + min_frames, S_dB.shape[1]):
                    try:
                        a, b = S_dB[:, i], S_dB[:, j]
                        n1, n2 = np.linalg.norm(a), np.linalg.norm(b)
                        if n1 > 1e-8 and n2 > 1e-8:
                            score = np.dot(a, b) / (n1 * n2)
                            if score > best_score: best_score, best_start, best_end = score, i, j
                    except: continue

            if best_end <= best_start or best_score < 0.1 or best_start < 0 or best_end >= S_dB.shape[1]:
                c, h = S_dB.shape[1] // 2, min_frames // 2
                best_start, best_end, best_score = max(0, c - h), min(S_dB.shape[1], c + h), 0.5

            s = self.find_nearest_zero_crossing(y, best_start * 512)
            e = self.find_nearest_zero_crossing(y, best_end * 512)
            s, e = max(0, min(s, len(y) - 1)), max(s + sr, min(e, len(y)))
            y_loop = y[s:e]

            try:
                tempo, _ = librosa.beat.beat_track(y=y_loop, sr=sr)
                tempo = tempo.item() if isinstance(tempo, np.ndarray) else tempo
                if tempo > 0:
                    bl = 60.0 / tempo
                    m = (len(y_loop) / sr) / bl // 4
                    if m >= 1:
                        samples = int(m * 4 * bl * sr)
                        y_loop = y_loop[:samples] if 0 < samples <= len(y_loop) else y_loop
                        measures = int(m)
                    else: measures = "< 1"
                else: measures = "?"
            except: measures = "?"

            return {
                'start_sample': s,
                'end_sample': s + len(y_loop),
                'score': best_score,
                'measures': measures,
                'metrics': {'Spettrale': best_score},
                'total_score': best_score
            }

        except Exception as e:
            self.message_queue.put(("log", f"⚠️ Errore algoritmo classico: {e}, uso fallback"))
            s, e = len(y) // 4, len(y) - len(y) // 4
            return {
                'start_sample': s,
                'end_sample': e,
                'score': 0.3,
                'measures': 'fallback',
                'metrics': {'Fallback': 0.3},
                'total_score': 0.3
            }


    def find_nearest_zero_crossing(self, y, sample):

        while sample < len(y) - 1 and np.sign(y[sample]) == np.sign(y[sample + 1]):
            sample += 1
        return sample


    def log_message(self, message):

        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"

        self.log_text.config(state='normal')
        self.log_text.insert('end', formatted_message)
        self.log_text.see('end')
        self.log_text.config(state='disabled')


    def update_crossfade(self, value):

        self.CROSSFADE_MS = int(value)
        self.CROSSFADE_SEC = self.CROSSFADE_MS / 1000.0


    def toggle_loop(self):

        if not self.is_playing:
            self.start_loop()
        else:
            self.stop_loop()


    def start_loop(self):

        self.PROMPT = self.prompt_entry.get().strip() + " 8-bar repeating seamless loop"
        if not self.PROMPT:
            messagebox.showerror("Errore", "Inserisci un prompt musicale!")
            return

        self.is_playing = True
        self.start_button.config(text="⏹️ FERMA")
        algorithm_type = "Avanzato" if self.use_advanced_algorithm.get() else "Classico"
        self.log_message(f"🚀 Invio prompt alla AI: '{self.PROMPT}'\n")


        self.loop_thread = threading.Thread(target=self.main_loop, daemon=True)
        self.loop_thread.start()


    def stop_loop(self):

        self.is_playing = False
        self.stop_event.set()
        self.start_button.config(text="🚀 AVVIA")
        self.log_message("⏹️ Loop fermato\n")
        self.generation_status_label.config(text="Fermato")


        try:
            result = subprocess.run(["pgrep", "-f", "musicgpt-x86_64-unknown-linux-gnu"], capture_output=True, text=True)
            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    subprocess.run(["kill", "-9", pid], check=False)
                self.log_message(f"🛑 MusicGPT terminato.\n")
        except:
            pass


    def main_loop(self):

        try:

            if not os.path.exists(self.CURRENT):
                self.message_queue.put(("status", "📁 Generazione primo sample..."))
                self.generate_audio(self.CURRENT)

            if not os.path.exists(self.NEXT):
                self.message_queue.put(("status", "📡 Generazione secondo loop..."))
                self.generate_audio(self.NEXT)


            self.run_loop()

        except Exception as e:
            self.message_queue.put(("error", f"❌ Errore nel loop: {str(e)}"))
            self.is_playing = False


    def generate_audio(self, outfile):
        try:
            self.message_queue.put(("status", "🎼 Generazione sample..."))
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f: temp = f.name
            try:
                subprocess.run([
                    "./musicgpt-x86_64-unknown-linux-gnu",
                    self.PROMPT,
                    "--model", self.model_var.get(),
                    "--secs", str(self.duration_var.get()),
                    "--output", temp,
                    "--no-playback", "--no-interactive", "--ui-no-open"
                ], check=True, capture_output=True, text=True)
                self.message_queue.put(("log", f"🎼 Sample generato ({self.duration_var.get()}s)!\n"))
                self.trim_with_improved_loop_detection(temp)
                with self.file_lock: shutil.move(temp, outfile)
            except Exception as e:
                if os.path.exists(temp): os.remove(temp)
                raise e
        except subprocess.CalledProcessError as e:
            self.message_queue.put(("log", f"❌ Errore generazione: {e}"))


    def safe_file_swap(self):
        with self.swap_lock:
            try:
                self.stop_event.set()
                if self.loop_thread and self.loop_thread.is_alive():
                    self.loop_thread.join(timeout=min(self.CROSSFADE_SEC + 1, 5))
                    if self.loop_thread.is_alive():
                        self.message_queue.put(("log", "⚠️ Timeout attesa - forzo terminazione"))
                        self.kill_all_ffplay_processes()
                        self.loop_thread.join(timeout=2)

                if not os.path.exists(self.NEXT) or os.path.getsize(self.NEXT) < 1024:
                    raise Exception(f"File NEXT non valido: {self.NEXT}")

                with self.file_lock:
                    self.CURRENT, self.NEXT = self.NEXT, self.CURRENT

                self.stop_event = threading.Event()
                return True

            except Exception as e:
                self.message_queue.put(("log", f"❌ Errore durante swap: {e}"))
                self.stop_event = threading.Event()
                return False


    def get_duration(self, filepath):

        try:
            info = mediainfo(filepath)
            return float(info['duration'])
        except:
            return 0.0


    def get_random_title(self):

        try:
            with open("nomi.txt", "r") as f1, open("nomi2.txt", "r") as f2:
                list1 = [line.strip().upper() for line in f1 if line.strip()]
                list2 = [line.strip().upper() for line in f2 if line.strip()]
            if list1 and list2:
                word1 = ''.join(c for c in random.choice(list1) if c.isalnum())
                word2 = ''.join(c for c in random.choice(list2) if c.isalnum())
                return f"{word1} {word2}"
        except Exception:
            pass
        return "SENZA TITOLO"


    def get_random_artist(self):

        try:
            with open("artisti.txt", "r") as f:
                artists = [line.strip() for line in f if line.strip()]
            return random.choice(artists).upper() if artists else "ARTISTA SCONOSCIUTO"
        except Exception:
            return "ARTISTA SCONOSCIUTO"


    def play_with_ffplay(self, filepath):
        """Play audio file with ffplay - versione robusta"""
        proc = None
        try:
            if not os.path.exists(filepath) or os.path.getsize(filepath) < 1024:
                self.message_queue.put(("log", f"⚠️ File non valido per riproduzione: {filepath}"))
                return

            env = os.environ.copy()
            env["SDL_AUDIODRIVER"] = self.audio_driver_var.get()
            proc = subprocess.Popen([
                "ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", filepath
            ], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            if proc.wait() != 0:
                self.message_queue.put(("log", f"⚠️ ffplay terminato con codice {proc.returncode}"))

        except subprocess.TimeoutExpired:
            self.message_queue.put(("log", "⚠️ ffplay timeout - terminazione forzata"))
            if proc: _kill(proc)
        except Exception as e:
            self.message_queue.put(("log", f"⚠️ ffplay error: {e}"))
            if proc: _kill(proc)
        finally:
            if proc and proc.poll() is None:
                _kill(proc)


    def _kill(proc):
        for f in (proc.terminate, proc.kill):
            try:
                f(); proc.wait(timeout=1); break
            except: continue


    def loop_current_crossfade_blocking(self, filepath, crossfade_sec, stop_event):
        """Loop con VERO crossfade audio - implementazione completa"""
        try:
            audio_data, sr = sf.read(filepath)
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)

            dur = len(audio_data) / sr
            delay = max(0, dur - crossfade_sec)

            title, artist = self.get_random_title(), self.get_random_artist()
            algo = "🧠 Avanzato" if self.use_advanced_algorithm.get() else "📊 Classico"
            self.message_queue.put(("track_info", f"Titolo:   {title}\nArtista:  {artist}\n\n⏱️ {dur:.1f}s | 🔀 {crossfade_sec:.3f}s\n{algo}"))

            audio_data = audio_data.astype(np.float32)
            peak = np.max(np.abs(audio_data))
            if peak > 0: audio_data *= 0.95 / peak

            if getattr(self, '_has_pyaudio', False):
                self._loop_with_pyaudio_crossfade(audio_data, sr, stop_event, delay, crossfade_sec)
            else:
                self._loop_with_ffplay_timing(filepath, stop_event, delay)

        except Exception as e:
            self.message_queue.put(("error", f"❌ Errore nel loop: {e}"))


    def _loop_with_pyaudio_crossfade(self, audio_data, sr, stop_event, delay, crossfade_sec):
        import pyaudio, time, numpy as np
        chunk = 512
        stream = pyaudio.PyAudio().open(format=pyaudio.paFloat32, channels=1, rate=sr, output=True, frames_per_buffer=chunk)
        chunks = [np.pad(audio_data[i:i+chunk], (0, max(0, chunk - len(audio_data[i:i+chunk]))), 'constant').astype(np.float32)
                for i in range(0, len(audio_data), chunk)]

        dur = chunk / sr
        delay_chunks = int(delay / dur)
        crossfade_chunks = int(crossfade_sec / dur)
        t0, i = time.perf_counter(), 0

        try:
            while not stop_event.is_set() and self.is_playing:
                c = chunks[i % len(chunks)]
                if i >= delay_chunks and hasattr(self, '_crossfade_next_audio'):
                    prog = min(1.0, (i - delay_chunks) / max(1, crossfade_chunks))
                    mix = c * (1 - prog) + self._crossfade_next_chunks[(i - delay_chunks) % len(self._crossfade_next_chunks)] * prog
                    stream.write(mix.tobytes())
                    if prog >= 1.0:
                        self._switch_to_next_audio()
                        break
                else:
                    stream.write(c.tobytes())

                i += 1
                if i >= len(chunks) and not hasattr(self, '_crossfade_next_audio'):
                    i, t0 = 0, time.perf_counter()

                sleep = t0 + i * dur - time.perf_counter()
                if sleep > 0: time.sleep(min(sleep, dur * 0.1))
        finally:
            stream.stop_stream(); stream.close(); stream._parent.terminate()


    def _loop_with_ffplay_timing(self, filepath, stop_event, delay):
        """Fallback: timing preciso senza vero crossfade"""
        while not stop_event.is_set() and self.is_playing:
            start_time = time.perf_counter()
            self.play_with_ffplay(filepath)

            if stop_event.is_set():
                break


            elapsed = time.perf_counter() - start_time
            remaining_delay = max(0, delay - elapsed)

            if remaining_delay > 0:
                if stop_event.wait(remaining_delay):
                    break

    def run_loop(self):
        self.loop_thread = threading.Thread(
            target=self.loop_current_crossfade_blocking,
            args=(self.CURRENT, self.CROSSFADE_SEC, self.stop_event),
            daemon=True
        )
        self.loop_thread.start()

        errors, max_errors = 0, 3

        while self.is_playing:
            try:
                self.generate_audio(self.NEXT)
                if not self.is_playing: break

                if not self.safe_file_swap():
                    self.message_queue.put(("log", "❌ Swap fallito, rigenerazione..."))
                    errors += 1
                    if errors >= max_errors:
                        self.message_queue.put(("log", "❌ Troppi errori, arresto"))
                        self.is_playing = False
                        break
                    continue

                if not self.is_playing: break

                self.loop_thread = threading.Thread(
                    target=self.loop_current_crossfade_blocking,
                    args=(self.CURRENT, self.CROSSFADE_SEC, self.stop_event),
                    daemon=True
                )
                self.loop_thread.start()
                errors = 0

            except Exception as e:
                errors += 1
                self.message_queue.put(("log", f"❌ Errore nel ciclo ({errors}/{max_errors}): {e}"))
                if errors >= max_errors:
                    self.message_queue.put(("log", "❌ Troppi errori consecutivi, arresto"))
                    self.is_playing = False
                    break

                self.kill_all_ffplay_processes()
                if self.stop_event.wait(2): break


    def kill_all_ffplay_processes(self):

        try:
            result = subprocess.run(["pgrep", "-f", "ffplay"], capture_output=True, text=True)
            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    try:
                        subprocess.run(["kill", "-9", pid], check=False, timeout=2)
                    except:
                        pass
        except Exception:

            pass

    def process_messages(self):

        try:
            while True:
                msg_type, content = self.message_queue.get_nowait()

                if msg_type == "log":
                    self.log_message(content)
                elif msg_type == "error":
                    self.log_message(content)
                    messagebox.showerror("Errore", content)
                elif msg_type == "status":
                    self.generation_status_label.config(text=content)
                    self.log_message(content)
                elif msg_type == "track_info":
                    self.track_info_label.config(text=content)
                elif msg_type == "waveform":
                    self.plot_waveform_advanced(content)
                elif msg_type == "waveform_advanced":
                    filepath, loop_info = content
                    self.plot_waveform_advanced(filepath, loop_info)

        except queue.Empty:
            pass

        self.root.after(100, self.process_messages)

    def open_settings(self):
        win = tk.Toplevel(self.root)
        win.title("⚙️ Impostazioni Avanzate")
        win.geometry("450x600")
        win.configure(bg='#1a1a1a')
        win.transient(self.root)
        win.grab_set()

        frame = tk.Frame(win, bg='#1a1a1a')
        frame.pack(fill='both', expand=True, padx=15, pady=15)

        tk.Label(frame, text="🛠️ IMPOSTAZIONI AVANZATE", bg='#1a1a1a', fg='#00ff88',
                font=('Courier New', 12, 'bold')).pack(pady=(0, 15))

        def section(label, parent=frame): return tk.Label(parent, text=label, bg='#1a1a1a', fg='#ffffff',
                                                        font=('Courier New', 10, 'bold'))

        section("🧠 Algoritmo Loop Detection:").pack(anchor='w', pady=(0, 3))
        algo = tk.Frame(frame, bg='#1a1a1a'); algo.pack(fill='x', pady=(0, 10))
        for text, val, color in [("🎯 Avanzato (Multi-metrico)", True, '#00ff88'),
                                ("📊 Classico (Solo spettrale)", False, '#ffffff')]:
            tk.Radiobutton(algo, text=text, variable=self.use_advanced_algorithm, value=val,
                        bg='#1a1a1a', fg=color, selectcolor='#2a2a2a',
                        activebackground='#2a2a2a', activeforeground=color,
                        font=('Courier New', 9, 'bold' if val else '')).pack(anchor='w', pady=(0, 3 if not val else 0))

        section("🎼 Modello AI:").pack(anchor='w', pady=(10, 3))
        model = tk.Frame(frame, bg='#1a1a1a'); model.pack(fill='x', pady=(0, 10))
        for text, val in [("Small (veloce)", "small"), ("Medium (bilanciato)", "medium"), ("Large (qualità)", "large")]:
            tk.Radiobutton(model, text=text, variable=self.model_var, value=val,
                        bg='#1a1a1a', fg='#ffffff', selectcolor='#2a2a2a',
                        activebackground='#2a2a2a', activeforeground='#00ff88',
                        font=('Courier New', 9)).pack(anchor='w')

        section("⏱️ Durata generazione (sec):").pack(anchor='w', pady=(10, 3))
        tk.Scale(tk.Frame(frame, bg='#1a1a1a'), from_=5, to=30, orient='horizontal',
                variable=self.duration_var, bg='#2a2a2a', fg='#00ff88',
                activebackground='#3a3a3a', highlightbackground='#1a1a1a',
                font=('Courier New', 8)).pack(fill='x', pady=(0, 10))

        section("🔊 Driver Audio:").pack(anchor='w', pady=(10, 3))
        drv = tk.Frame(frame, bg='#1a1a1a'); drv.pack(fill='x', pady=(0, 15))
        for text, val in [("PulseAudio", "pulse"), ("ALSA", "alsa"), ("OSS", "dsp")]:
            tk.Radiobutton(drv, text=text, variable=self.audio_driver_var, value=val,
                        bg='#1a1a1a', fg='#ffffff', selectcolor='#2a2a2a',
                        activebackground='#2a2a2a', activeforeground='#00ff88',
                        font=('Courier New', 9)).pack(anchor='w')

        info = tk.LabelFrame(frame, text=" ℹ️ Info Algoritmi ", bg='#1a1a1a', fg='#00ff88',
                            font=('Courier New', 9, 'bold'))
        info.pack(fill='x', pady=(5, 15))
        tk.Label(info, text="🎯 AVANZATO: Multi-metrico (spettrale, waveform, battiti, fasi)\n"
                            "📊 CLASSICO: Solo similarità spettrale mel (più veloce)",
                bg='#1a1a1a', fg='#ffffff', font=('Courier New', 8),
                justify='left', wraplength=400).pack(padx=8, pady=8)

        btns = tk.Frame(frame, bg='#1a1a1a'); btns.pack(fill='x')
        ttk.Button(btns, text="✅ SALVA", style='Custom.TButton',
                command=lambda: self.save_settings(win)).pack(side='left', padx=(0, 10))
        ttk.Button(btns, text="❌ ANNULLA", style='Custom.TButton',
                command=win.destroy).pack(side='left')


    def save_settings(self, window):

        algorithm_type = "Avanzato" if self.use_advanced_algorithm.get() else "Classico"
        self.log_message(f"⚙️ Impostazioni salvate! Algoritmo: {algorithm_type}")
        window.destroy()


    def save_current_loop(self):

        if not os.path.exists(self.CURRENT):
            messagebox.showwarning("Attenzione", "Nessun loop da salvare!")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("File WAV", "*.wav"), ("Tutti i file", "*.*")],
            title="Salva Loop Corrente"
        )

        if filename:
            try:
                shutil.copy2(self.CURRENT, filename)
                self.log_message(f"💾 Loop salvato: {os.path.basename(filename)}")
                messagebox.showinfo("Successo", f"Loop salvato come:\n{os.path.basename(filename)}")
            except Exception as e:
                self.log_message(f"❌ Errore salvataggio: {str(e)}")
                messagebox.showerror("Errore", f"Impossibile salvare:\n{str(e)}")


def main():

    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    os.environ["SDL_AUDIODRIVER"] = "dsp"
    os.environ["ALSA_CARD"] = "default"


    root = tk.Tk()
    app = InfiniLoopGUI(root)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\n🛑 Applicazione terminata dall'utente")
        app.stop_loop()

if __name__ == "__main__":
    main()
