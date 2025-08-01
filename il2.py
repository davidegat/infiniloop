import subprocess
import os
import time
import soundfile as sf
import numpy as np
import librosa
import librosa.display
import threading
from pydub.utils import mediainfo
import sys
import random
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.patches as patches
from datetime import datetime
import queue
from scipy.signal import correlate
from scipy.spatial.distance import cosine

class InfiniLoopGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("INFINI LOOP - by gat")
        self.root.attributes('-fullscreen', True)
        self.root.configure(bg='#0a0a0a')
        self.root.resizable(True, True)
        self.root.bind('<Escape>', lambda e: self.root.attributes('-fullscreen', False))


        self.FILE1 = "./music1.wav"
        self.FILE2 = "./music2.wav"
        self.CURRENT = self.FILE1
        self.NEXT = self.FILE2
        self.CROSSFADE_MS = 1
        self.CROSSFADE_SEC = self.CROSSFADE_MS / 1000.0
        self.PROMPT = ""


        self.use_advanced_algorithm = tk.BooleanVar(value=True)
        self.model_var = tk.StringVar(value="medium")
        self.duration_var = tk.IntVar(value=15)
        self.audio_driver_var = tk.StringVar(value="pulse")


        self.is_playing = False
        self.stop_event = threading.Event()
        self.loop_thread = None
        self.generation_thread = None


        self.message_queue = queue.Queue()


        self.setup_styles()


        self.create_gui()


        self.process_messages()

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
                               text="I N F I N I   L O O P   A D V A N C E D",
                               style='Title.TLabel')
        title_label.pack()

        subtitle_label = ttk.Label(header_frame,
                                  text="🎵 Infinite AI Music Generation • Advanced Loop Detection • by gat 🎵",
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
        self.prompt_entry.insert(0, "chill loop melody")


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
        self.log_message("🧠 Algoritmo avanzato attivato")
        self.log_message("💡 Inserisci prompt e premi AVVIA\n")

    def plot_welcome_screen(self):

        for ax in [self.ax_wave, self.ax_spectrum, self.ax_analysis]:
            ax.clear()
            ax.set_facecolor('#0a0a0a')


        x = np.linspace(0, 10, 1000)
        y1 = np.sin(x) * np.exp(-x/10)
        y2 = np.sin(x * 2) * np.exp(-x/10) * 0.5

        self.ax_wave.plot(x, y1, color='#00ff88', linewidth=1.5, alpha=0.8, label='Loop A')
        self.ax_wave.plot(x, y2, color='#ff0088', linewidth=1.5, alpha=0.8, label='Loop B')
        self.ax_wave.set_title('Analisi del campione', color='#00ff88', fontsize=9, fontweight='bold')
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


        self.fig.tight_layout(pad=1.0)
        self.canvas.draw()

    def plot_waveform_advanced(self, filepath, loop_info=None):

        try:
            y, sr = librosa.load(filepath, sr=None, mono=True)

            for ax in [self.ax_wave, self.ax_spectrum, self.ax_analysis]:
                ax.clear()
                ax.set_facecolor('#0a0a0a')


            time_axis = np.linspace(0, len(y)/sr, len(y))
            self.ax_wave.plot(time_axis, y, color='#00ff88', linewidth=1, alpha=0.8)

            if loop_info:
                start_time = loop_info['start_sample'] / sr
                end_time = loop_info['end_sample'] / sr
                self.ax_wave.axvspan(start_time, end_time, alpha=0.3, color='#ff0088', label='Loop')
                self.ax_wave.axvline(start_time, color='#00ff88', linestyle='--', alpha=0.8, label='Start')
                self.ax_wave.axvline(end_time, color='#ff0088', linestyle='--', alpha=0.8, label='End')


            measures_text = f"{loop_info.get('measures', '?')}mis" if loop_info else "?"
            self.ax_wave.set_title(f'Analisi sample - {measures_text}',
                                  color='#00ff88', fontsize=9, fontweight='bold')
            self.ax_wave.set_xlabel('Tempo (s)', color='#00ff88', fontsize=8)
            self.ax_wave.set_ylabel('Ampiezza', color='#00ff88', fontsize=8)
            self.ax_wave.legend(facecolor='#1a1a1a', edgecolor='#00ff88', labelcolor='#ffffff', fontsize=6)
            self.ax_wave.grid(True, alpha=0.3, color='#00ff88')


            S = librosa.feature.melspectrogram(y=y, sr=sr)
            S_dB = librosa.power_to_db(S, ref=np.max)
            mean_spectrum = np.mean(S_dB, axis=1)

            mel_freqs = librosa.mel_frequencies(n_mels=S.shape[0])
            self.ax_spectrum.plot(mel_freqs, mean_spectrum, color='#00ff88', linewidth=1)
            self.ax_spectrum.set_title('Spettro Mel', color='#00ff88', fontsize=8, fontweight='bold')
            self.ax_spectrum.set_xlabel('Frequenza (Hz)', color='#00ff88', fontsize=7)
            self.ax_spectrum.set_ylabel('Volume (dB)', color='#00ff88', fontsize=7)
            self.ax_spectrum.grid(True, alpha=0.3, color='#00ff88')


            if loop_info and 'metrics' in loop_info:

                metric_names = {'Spettrale': 'Spett', 'Waveform': 'Wave', 'Beat Align': 'Beat', 'Fase': 'Fase'}
                metrics = [metric_names.get(k, k) for k in loop_info['metrics'].keys()]
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


            self.fig.tight_layout(pad=1.0)
            self.canvas.draw()

        except Exception as e:
            self.log_message(f"❌ Errore visualizzazione: {str(e)}")



    def find_optimal_zero_crossing(self, y, sample, window_size=256):

        start = max(0, sample - window_size // 2)
        end = min(len(y), sample + window_size // 2)

        best_sample = sample
        min_amplitude = float('inf')

        for i in range(start, end - 1):
            if np.sign(y[i]) != np.sign(y[i + 1]):
                amplitude = abs(y[i]) + abs(y[i + 1])
                if amplitude < min_amplitude:
                    min_amplitude = amplitude
                    best_sample = i

        return best_sample

    def calculate_waveform_continuity(self, y, start_sample, end_sample, sr):

        transition_length = min(sr // 40, (end_sample - start_sample) // 20)

        if transition_length < 64:
            transition_length = 64

        end_segment = y[max(0, end_sample - transition_length):end_sample]
        start_segment = y[start_sample:min(len(y), start_sample + transition_length)]

        if len(end_segment) == 0 or len(start_segment) == 0:
            return 0.0

        min_len = min(len(end_segment), len(start_segment))
        end_segment = end_segment[-min_len:]
        start_segment = start_segment[:min_len]


        correlation = np.corrcoef(end_segment, start_segment)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0


        rms_diff = np.sqrt(np.mean((end_segment - start_segment) ** 2))
        max_rms = max(np.sqrt(np.mean(end_segment ** 2)), np.sqrt(np.mean(start_segment ** 2)))

        if max_rms > 1e-8:
            rms_similarity = 1 - min(1.0, rms_diff / max_rms)
        else:
            rms_similarity = 1.0


        if len(end_segment) > 1 and len(start_segment) > 1:
            end_derivative = np.diff(end_segment)
            start_derivative = np.diff(start_segment)
            derivative_diff = abs(end_derivative[-1] - start_derivative[0])
            max_derivative = max(abs(end_derivative[-1]), abs(start_derivative[0]), 1e-8)
            derivative_continuity = 1 - min(1.0, derivative_diff / max_derivative)
        else:
            derivative_continuity = 1.0

        return (abs(correlation) * 0.4 + rms_similarity * 0.4 + derivative_continuity * 0.2)

    def calculate_beat_alignment(self, start_sample, end_sample, beats, sr):

        if len(beats) == 0:
            return 0.5

        start_distances = np.abs(beats - start_sample)
        end_distances = np.abs(beats - end_sample)

        closest_start_distance = np.min(start_distances)
        closest_end_distance = np.min(end_distances)

        if len(beats) > 1:
            avg_beat_distance = np.mean(np.diff(beats))
            start_alignment = 1 - min(1.0, closest_start_distance / (avg_beat_distance * 0.5))
            end_alignment = 1 - min(1.0, closest_end_distance / (avg_beat_distance * 0.5))
        else:
            start_alignment = 0.5
            end_alignment = 0.5

        return (start_alignment + end_alignment) / 2

    def calculate_phase_continuity(self, S, start_frame, end_frame, window=3):

        if start_frame < window or end_frame >= S.shape[1] - window:
            return 0.5

        start_window = S[:, max(0, start_frame - window):start_frame + window]
        end_window = S[:, max(0, end_frame - window):min(S.shape[1], end_frame + window)]

        start_phases = np.angle(start_window)
        end_phases = np.angle(end_window)

        start_mean_phase = np.mean(start_phases, axis=1)
        end_mean_phase = np.mean(end_phases, axis=1)

        phase_diff = np.abs(start_mean_phase - end_mean_phase)
        phase_diff = np.minimum(phase_diff, 2 * np.pi - phase_diff)

        phase_continuity = 1 - np.mean(phase_diff) / np.pi
        return max(0.0, phase_continuity)

    def find_perfect_loop_advanced(self, y, sr):

        self.message_queue.put(("log", "🧠 Uso algoritmo avanzato multi-metrico"))

        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='samples')
        if isinstance(tempo, np.ndarray):
            tempo = tempo.item()

        self.message_queue.put(("log", f"🥁 Tempo: {tempo:.1f} BPM"))

        hop_length = 512
        S = librosa.stft(y, n_fft=2048, hop_length=hop_length)
        S_mag = np.abs(S)

        beat_length = 60.0 / tempo
        possible_measures = [4, 8]

        best_score = -np.inf
        best_start = 0
        best_end = 0
        best_measure_count = 4
        best_metrics = {}

        for measures in possible_measures:
            target_beats = measures * 4
            target_duration = target_beats * beat_length
            target_samples = int(target_duration * sr)

            if target_samples > len(y) * 0.8 or target_duration < 3.0:
                continue

            search_start = int(len(y) * 0.1)
            search_end = len(y) - target_samples - int(len(y) * 0.1)
            step_size = hop_length * 2

            for start_sample in range(search_start, search_end, step_size):
                end_sample = start_sample + target_samples

                if end_sample >= len(y):
                    break

                start_frame = start_sample // hop_length
                end_frame = end_sample // hop_length

                if end_frame >= S_mag.shape[1]:
                    continue


                window = 5
                start_spectrum = np.mean(S_mag[:, max(0, start_frame-window):start_frame+window], axis=1)
                end_spectrum = np.mean(S_mag[:, max(0, end_frame-window):end_frame+window], axis=1)

                if np.linalg.norm(start_spectrum) > 1e-8 and np.linalg.norm(end_spectrum) > 1e-8:
                    spectral_similarity = 1 - cosine(start_spectrum, end_spectrum)
                else:
                    spectral_similarity = 0.0

                waveform_continuity = self.calculate_waveform_continuity(y, start_sample, end_sample, sr)
                beat_alignment = self.calculate_beat_alignment(start_sample, end_sample, beats, sr)
                phase_continuity = self.calculate_phase_continuity(S, start_frame, end_frame)


                combined_score = (
                    spectral_similarity * 0.25 +
                    waveform_continuity * 0.35 +
                    beat_alignment * 0.25 +
                    phase_continuity * 0.15
                )

                if combined_score > best_score:
                    best_score = combined_score
                    best_start = start_sample
                    best_end = end_sample
                    best_measure_count = measures
                    best_metrics = {
                        'Spettrale': spectral_similarity,
                        'Waveform': waveform_continuity,
                        'Beat Align': beat_alignment,
                        'Fase': phase_continuity
                    }

        self.message_queue.put(("log", f"✅ Miglior loop trovato! {best_measure_count} mis, Score: {best_score:.3f}\n"))

        return {
            'start_sample': best_start,
            'end_sample': best_end,
            'score': best_score,
            'measures': best_measure_count,
            'metrics': best_metrics,
            'total_score': best_score
        }

    def trim_with_improved_loop_detection(self, filepath):

        try:
            self.message_queue.put(("log", f"🔍 Analisi avanzata loop detection..."))

            y, sr = librosa.load(filepath, sr=None, mono=True)

            if self.use_advanced_algorithm.get():

                loop_info = self.find_perfect_loop_advanced(y, sr)
                start_sample = loop_info['start_sample']
                end_sample = loop_info['end_sample']
                measures = loop_info['measures']
                score = loop_info['score']

                self.message_queue.put(("log", f"🎯 Ottimizzazione zero-crossing..."))
                start_sample = self.find_optimal_zero_crossing(y, start_sample)
                end_sample = self.find_optimal_zero_crossing(y, end_sample)


                loop_info['start_sample'] = start_sample
                loop_info['end_sample'] = end_sample

            else:

                self.message_queue.put(("log", "📊 Uso algoritmo classico"))
                loop_info = self.trim_with_spectral_similarity_classic(y, sr)
                start_sample = loop_info['start_sample']
                end_sample = loop_info['end_sample']
                measures = loop_info.get('measures', '?')
                score = loop_info['score']


            y_loop = y[start_sample:end_sample]


            fade_samples = min(256, len(y_loop) // 100)
            if fade_samples > 0:
                fade_in = np.linspace(0, 1, fade_samples)
                fade_out = np.linspace(1, 0, fade_samples)
                y_loop[:fade_samples] *= fade_in
                y_loop[-fade_samples:] *= fade_out


            original_path = filepath + ".original.wav"
            sf.write(original_path, y, sr)


            self.message_queue.put(("waveform_advanced", (original_path, loop_info)))


            sf.write(filepath, y_loop, sr)


            duration = len(y_loop) / sr
            self.message_queue.put(("log", f"🧬 Rendo perfetto il loop! {measures} mis, {duration:.1f}s\n"))




        except Exception as e:
            self.message_queue.put(("error", f"❌ Errore loop detection: {str(e)}"))

    def trim_with_spectral_similarity_classic(self, y, sr):

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512)
        S_dB = librosa.power_to_db(S, ref=np.max)

        min_duration_sec = 7
        min_frames = int(min_duration_sec * sr / 512)

        best_score = -np.inf
        best_start = 0
        best_end = 0

        for i in range(S_dB.shape[1] - min_frames):
            for j in range(i + min_frames, S_dB.shape[1]):
                segment1 = S_dB[:, i]
                segment2 = S_dB[:, j]
                score = np.dot(segment1, segment2) / (np.linalg.norm(segment1) * np.linalg.norm(segment2))
                if score > best_score:
                    best_score = score
                    best_start = i
                    best_end = j

        start_sample = self.find_nearest_zero_crossing(y, best_start * 512)
        end_sample = self.find_nearest_zero_crossing(y, best_end * 512)

        y_loop = y[start_sample:end_sample]

        tempo, _ = librosa.beat.beat_track(y=y_loop, sr=sr)
        if isinstance(tempo, np.ndarray):
            tempo = tempo.item()

        beat_len = 60.0 / tempo
        num_beats = int((len(y_loop) / sr) / beat_len)
        num_measures = num_beats // 4

        if num_measures >= 1:
            final_beats = num_measures * 4
            loop_duration = final_beats * beat_len
            loop_samples = int(loop_duration * sr)
            y_loop = y_loop[:loop_samples]
            measures = num_measures
        else:
            measures = "< 1"

        return {
            'start_sample': start_sample,
            'end_sample': end_sample,
            'score': best_score,
            'measures': measures,
            'metrics': {'Spettrale': best_score}
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

        self.PROMPT = self.prompt_entry.get().strip()
        if not self.PROMPT:
            messagebox.showerror("Errore", "Inserisci un prompt musicale!")
            return

        self.is_playing = True
        self.start_button.config(text="⏹️ FERMA")
        algorithm_type = "Avanzato" if self.use_advanced_algorithm.get() else "Classico"
        self.log_message(f"🚀 Avvio {algorithm_type}: '{self.PROMPT}'")


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
            self.message_queue.put(("status", f"🎼 Generazione sample..."))

            result = subprocess.run([
                "./musicgpt-x86_64-unknown-linux-gnu",
                self.PROMPT,
                "--model", self.model_var.get(),
                "--secs", str(self.duration_var.get()),
                "--output", outfile,
                "--no-playback",
                "--no-interactive",
                "--ui-no-open"
            ], check=True, capture_output=True, text=True)

            self.message_queue.put(("log", f"🎼 Sample generato ({self.duration_var.get()}s)!\n"))
            self.trim_with_improved_loop_detection(outfile)

        except subprocess.CalledProcessError as e:
            print("")
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

        try:
            env = os.environ.copy()
            env["SDL_AUDIODRIVER"] = self.audio_driver_var.get()
            subprocess.run([
                "ffplay",
                "-nodisp",
                "-autoexit",
                "-loglevel", "quiet",
                filepath
            ], env=env, check=True)
        except Exception as e:
            self.message_queue.put(("error", f"❌ Errore riproduzione: {str(e)}"))

    def loop_current_crossfade_blocking(self, filepath, crossfade_sec, stop_event):

        try:
            duration = self.get_duration(filepath)
            delay = max(0, duration - crossfade_sec)
            title = self.get_random_title()
            artist = self.get_random_artist()

            algorithm_type = "🧠 Avanzato" if self.use_advanced_algorithm.get() else "📊 Classico"
            info_text = f"Titolo:   {title}\nArtista:  {artist}\n\n⏱️ {duration:.1f}s | 🔀 {crossfade_sec:.1f}s\n{algorithm_type}"
            self.message_queue.put(("track_info", info_text))

            while not stop_event.is_set() and self.is_playing:
                play_thread = threading.Thread(target=self.play_with_ffplay, args=(filepath,), daemon=True)
                play_thread.start()


                if stop_event.wait(delay):
                    break

        except Exception as e:
            self.message_queue.put(("error", f"❌ Errore nel loop: {str(e)}"))

    def run_loop(self):

        stop_event = threading.Event()
        self.stop_event = stop_event


        loop_thread = threading.Thread(
            target=self.loop_current_crossfade_blocking,
            args=(self.CURRENT, self.CROSSFADE_SEC, stop_event),
            daemon=True
        )
        loop_thread.start()

        while self.is_playing and not stop_event.is_set():

            self.generate_audio(self.NEXT)

            if not self.is_playing:
                break

            self.message_queue.put(("log", "🔁 Nuovo loop! Mixing...\n"))


            stop_event.set()
            loop_thread.join(timeout=1.0)


            self.CURRENT, self.NEXT = self.NEXT, self.CURRENT

            if not self.is_playing:
                break


            stop_event = threading.Event()
            self.stop_event = stop_event
            loop_thread = threading.Thread(
                target=self.loop_current_crossfade_blocking,
                args=(self.CURRENT, self.CROSSFADE_SEC, stop_event),
                daemon=True
            )
            loop_thread.start()

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

        settings_window = tk.Toplevel(self.root)
        settings_window.title("⚙️ Impostazioni Avanzate")
        settings_window.geometry("450x600")
        settings_window.configure(bg='#1a1a1a')
        settings_window.transient(self.root)
        settings_window.grab_set()


        settings_frame = tk.Frame(settings_window, bg='#1a1a1a')
        settings_frame.pack(fill='both', expand=True, padx=15, pady=15)

        tk.Label(settings_frame, text="🛠️ IMPOSTAZIONI AVANZATE",
                bg='#1a1a1a', fg='#00ff88',
                font=('Courier New', 12, 'bold')).pack(pady=(0, 15))


        tk.Label(settings_frame, text="🧠 Algoritmo Loop Detection:",
                bg='#1a1a1a', fg='#ffffff',
                font=('Courier New', 10, 'bold')).pack(anchor='w', pady=(0, 3))

        algo_frame = tk.Frame(settings_frame, bg='#1a1a1a')
        algo_frame.pack(fill='x', pady=(0, 10))

        tk.Radiobutton(algo_frame, text="🎯 Avanzato (Multi-metrico)",
                      variable=self.use_advanced_algorithm, value=True,
                      bg='#1a1a1a', fg='#00ff88', selectcolor='#2a2a2a',
                      activebackground='#2a2a2a', activeforeground='#00ff88',
                      font=('Courier New', 9, 'bold')).pack(anchor='w')

        tk.Radiobutton(algo_frame, text="📊 Classico (Solo spettrale)",
                      variable=self.use_advanced_algorithm, value=False,
                      bg='#1a1a1a', fg='#ffffff', selectcolor='#2a2a2a',
                      activebackground='#2a2a2a', activeforeground='#ffffff',
                      font=('Courier New', 9)).pack(anchor='w', pady=(3, 0))


        tk.Label(settings_frame, text="🎼 Modello AI:",
                bg='#1a1a1a', fg='#ffffff',
                font=('Courier New', 10, 'bold')).pack(anchor='w', pady=(10, 3))

        model_frame = tk.Frame(settings_frame, bg='#1a1a1a')
        model_frame.pack(fill='x', pady=(0, 10))

        models = [("Small (veloce)", "small"), ("Medium (bilanciato)", "medium"), ("Large (qualità)", "large")]
        for text, value in models:
            tk.Radiobutton(model_frame, text=text, variable=self.model_var, value=value,
                          bg='#1a1a1a', fg='#ffffff', selectcolor='#2a2a2a',
                          activebackground='#2a2a2a', activeforeground='#00ff88',
                          font=('Courier New', 9)).pack(anchor='w')


        tk.Label(settings_frame, text="⏱️ Durata generazione (sec):",
                bg='#1a1a1a', fg='#ffffff',
                font=('Courier New', 10, 'bold')).pack(anchor='w', pady=(10, 3))

        duration_frame = tk.Frame(settings_frame, bg='#1a1a1a')
        duration_frame.pack(fill='x', pady=(0, 10))

        duration_scale = tk.Scale(duration_frame, from_=5, to=30, orient='horizontal',
                                 variable=self.duration_var, bg='#2a2a2a', fg='#00ff88',
                                 activebackground='#3a3a3a', highlightbackground='#1a1a1a',
                                 font=('Courier New', 8))
        duration_scale.pack(fill='x')


        tk.Label(settings_frame, text="🔊 Driver Audio:",
                bg='#1a1a1a', fg='#ffffff',
                font=('Courier New', 10, 'bold')).pack(anchor='w', pady=(10, 3))

        driver_frame = tk.Frame(settings_frame, bg='#1a1a1a')
        driver_frame.pack(fill='x', pady=(0, 15))

        drivers = [("PulseAudio", "pulse"), ("ALSA", "alsa"), ("OSS", "dsp")]
        for text, value in drivers:
            tk.Radiobutton(driver_frame, text=text, variable=self.audio_driver_var, value=value,
                          bg='#1a1a1a', fg='#ffffff', selectcolor='#2a2a2a',
                          activebackground='#2a2a2a', activeforeground='#00ff88',
                          font=('Courier New', 9)).pack(anchor='w')


        info_frame = tk.LabelFrame(settings_frame, text=" ℹ️ Info Algoritmi ",
                                  bg='#1a1a1a', fg='#00ff88', font=('Courier New', 9, 'bold'))
        info_frame.pack(fill='x', pady=(5, 15))

        info_text = """🎯 AVANZATO: Multi-metrico (spettrale, waveform, battiti, fasi)
📊 CLASSICO: Solo similarità spettrale mel (più veloce)"""

        tk.Label(info_frame, text=info_text, bg='#1a1a1a', fg='#ffffff',
                font=('Courier New', 8), justify='left', wraplength=400).pack(padx=8, pady=8)


        button_frame = tk.Frame(settings_frame, bg='#1a1a1a')
        button_frame.pack(fill='x')

        ttk.Button(button_frame, text="✅ SALVA", style='Custom.TButton',
                  command=lambda: self.save_settings(settings_window)).pack(side='left', padx=(0, 10))

        ttk.Button(button_frame, text="❌ ANNULLA", style='Custom.TButton',
                  command=settings_window.destroy).pack(side='left')

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
                import shutil
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
