"""
INFINI LOOP by gat
https://github.com/davidegat/infiniloop

For non commercial use only
"""


#!/usr/bin/env python3
import sys
import os
import threading
import time
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, font
import json
from queue import Queue
import random


from ilterm import InfiniLoopTerminal

class InfiniLoopGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎵 InfiniLoop - Local AI Infinite Music Generator")
        self.root.geometry("600x900")
        self.root.minsize(500, 700)
        self.root.maxsize(700, 950)

        self.app = InfiniLoopTerminal()
        self.app.on_generation_complete = self.on_generation_complete
        self.is_running = False
        self.log_queue = Queue()

        self.current_loop_file = None
        self.last_title = None
        self.last_artist = None

        self.original_log = self.app.logging_system
        self.app.logging_system = self.capture_log

        self.colors = {
            'bg': '#1a1a2e',
            'bg_secondary': '#16213e',
            'bg_card': '#0f3460',
            'accent': '#3282b8',
            'accent_hover': '#45a0e6',
            'success': '#00d084',
            'danger': '#e74c3c',
            'text': '#e0e0e0',
            'text_secondary': '#a0a0b0',
            'border': '#0f4c75'
        }

        self.setup_styles()
        self.create_ui()
        self.load_settings()

        self.update_model_ui()
        self.update_loop()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def get_model_path(self, model_name):
        home = os.path.expanduser("~")
        model_dirs = {
            "small": os.path.join(home, ".local/share/musicgpt/v1/small_fp32/"),
            "medium": os.path.join(home, ".local/share/musicgpt/v1/medium_fp32/"),
            "large": os.path.join(home, ".local/share/musicgpt/v1/large_fp32/")
        }

        model_dir = model_dirs.get(model_name)
        if not model_dir or not os.path.exists(model_dir):
            return None


        largest_file = None
        largest_size = 0

        try:
            for file in os.listdir(model_dir):
                file_path = os.path.join(model_dir, file)
                if os.path.isfile(file_path):
                    file_size = os.path.getsize(file_path)
                    if file_size > largest_size:
                        largest_size = file_size
                        largest_file = file_path
        except Exception:
            return None

        return largest_file


    def is_model_installed(self, model_name):

        model_path = self.get_model_path(model_name)
        return model_path and os.path.exists(model_path)


    def get_model_size(self, model_name):
        model_path = self.get_model_path(model_name)
        if model_path and os.path.exists(model_path):
            size_bytes = os.path.getsize(model_path)
            return round(size_bytes / (1024 * 1024), 1)
        return 0


    def delete_model(self, model_name):
        if not self.is_model_installed(model_name):
            messagebox.showwarning("Warning", f"Model {model_name} is not installed!")
            return

        home = os.path.expanduser("~")
        model_dir = os.path.join(home, f".local/share/musicgpt/v1/{model_name}_fp32/")
        size_mb = self.get_model_size(model_name)

        result = messagebox.askyesno(
            "Confirm Model Deletion",
            f"Are you sure you want to delete the {model_name} model?\n\n"
            f"Directory: {model_dir}\n"
            f"Size: {size_mb} MB\n\n"
            f"This action cannot be undone."
        )

        if result:
            try:

                shutil.rmtree(model_dir)
                self.capture_log(f"✅ Model {model_name} deleted successfully ({size_mb} MB freed)")
                messagebox.showinfo("Success", f"Model {model_name} deleted successfully!")

                if self.app.model == model_name:
                    available_models = [m for m in ["small", "medium", "large"] if self.is_model_installed(m)]
                    if available_models:
                        self.app.model = available_models[0]
                        self.model_var.set(self.app.model)
                        self.capture_log(f"🔄 Switched to model: {self.app.model}")
                    else:
                        self.capture_log("⚠️ No models available! Please install a model to continue.")

                self.update_model_ui()
                self.save_settings()

            except Exception as e:
                self.capture_log(f"❌ Failed to delete model {model_name}: {e}")
                messagebox.showerror("Error", f"Failed to delete model {model_name}:\n{e}")

    def update_model_ui(self):

        available_models = []
        model_labels = []

        for model in ["small", "medium", "large"]:
            if self.is_model_installed(model):
                size_mb = self.get_model_size(model)
                available_models.append(model)
                model_labels.append(f"{model} ({size_mb} MB) ✅")
            else:
                available_models.append(model)
                model_labels.append(f"{model} (not installed)")


        self.model_menu['values'] = model_labels


        if hasattr(self, 'model_delete_buttons'):
            for model, button in self.model_delete_buttons.items():
                if self.is_model_installed(model):
                    button.config(state='normal')
                else:
                    button.config(state='disabled')


        self.status_label.config(text=f"🔴 Ready - Model: {self.app.model}")

    def get_model_display_text(self, model):

        if self.is_model_installed(model):
            size_mb = self.get_model_size(model)
            return f"{model} ({size_mb} MB) ✅"
        else:
            return f"{model} (not installed)"

    def on_generation_complete(self):

        if hasattr(self, "update_benchmark_table"):
            self.update_benchmark_table()


    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')


        style.configure(".",
            background=self.colors['bg'],
            foreground=self.colors['text'],
            bordercolor=self.colors['border'],
            focuscolor='none',
            relief='flat')


        style.configure("Card.TFrame",
            background=self.colors['bg_card'],
            relief='raised',
            borderwidth=2)


        style.configure("Heading.TLabel",
            background=self.colors['bg'],
            foreground=self.colors['accent'],
            font=('Segoe UI', 14, 'bold'))

        style.configure("Title.TLabel",
            background=self.colors['bg'],
            foreground=self.colors['accent'],
            font=('Segoe UI', 24, 'bold'))


        style.configure("Accent.TButton",
            background=self.colors['accent'],
            foreground='white',
            borderwidth=0,
            focuscolor='none',
            font=('Segoe UI', 11, 'bold'))
        style.map("Accent.TButton",
            background=[('active', self.colors['accent_hover'])])

        style.configure("Success.TButton",
            background=self.colors['success'],
            foreground='white',
            borderwidth=0,
            font=('Segoe UI', 12, 'bold'))
        style.map("Success.TButton",
            background=[('active', '#00f094')])

        style.configure("Danger.TButton",
            background=self.colors['danger'],
            foreground='white',
            borderwidth=0,
            font=('Segoe UI', 12, 'bold'))
        style.map("Danger.TButton",
            background=[('active', '#ff5c4c')])


        style.configure("TNotebook",
            background=self.colors['bg'],
            borderwidth=0)

        style.configure("TNotebook.Tab",
            background=self.colors['bg_secondary'],
            foreground=self.colors['text_secondary'],
            padding=[10, 5],
            font=('Segoe UI', 10))

        style.map("TNotebook.Tab",
            background=[('selected', self.colors['bg_card'])],
            foreground=[('selected', self.colors['text'])],
            padding=[('selected', [20, 12])])


        style.configure("TCombobox",
            fieldbackground=self.colors['bg_secondary'],
            background=self.colors['bg_secondary'],
            foreground=self.colors['text'],
            bordercolor=self.colors['border'],
            arrowcolor=self.colors['text'],
            focuscolor='none',
            selectbackground=self.colors['accent'],
            selectforeground='white')

        style.map("TCombobox",
            fieldbackground=[('readonly', self.colors['bg_secondary']),
                            ('focus', self.colors['bg_secondary'])],
            background=[('readonly', self.colors['bg_secondary'])],
            foreground=[('readonly', self.colors['text'])],
            arrowcolor=[('readonly', self.colors['text'])],
            selectbackground=[('readonly', self.colors['accent'])],
            selectforeground=[('readonly', 'white')])


        self.root.option_add('*TCombobox*Listbox.background', self.colors['bg_secondary'])
        self.root.option_add('*TCombobox*Listbox.foreground', self.colors['text'])
        self.root.option_add('*TCombobox*Listbox.selectBackground', self.colors['accent'])
        self.root.option_add('*TCombobox*Listbox.selectForeground', 'white')

    def create_ui(self):

        self.root.configure(bg=self.colors['bg'])

        main_container = tk.Frame(self.root, bg=self.colors['bg'])
        main_container.pack(fill='both', expand=True, padx=20, pady=20)

        self.create_header(main_container)

        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill='both', expand=True, pady=(20, 0))

        self.create_controls_tab()
        self.create_log_tab()
        self.app.load_benchmark_data()
        self.create_benchmark_tab()
        self.create_settings_tab()

        self.create_status_bar(main_container)


    def create_header(self, parent):

        header_frame = tk.Frame(parent, bg=self.colors['accent'], height=80)
        header_frame.pack(fill='x', pady=(0, 10))
        header_frame.pack_propagate(False)

        gradient_frame = tk.Frame(header_frame, bg=self.colors['accent'])
        gradient_frame.pack(fill='both', expand=True, padx=2, pady=2)

        content_frame = tk.Frame(gradient_frame, bg=self.colors['accent'])
        content_frame.pack(fill='both', expand=True, padx=20)

        title_frame = tk.Frame(content_frame, bg=self.colors['accent'])
        title_frame.pack(side='left', fill='y')

        logo_label = tk.Label(title_frame, text="🎵", font=('Segoe UI', 32),
                             bg=self.colors['accent'], fg='white')
        logo_label.pack(side='left', padx=(0, 15))

        title_label = tk.Label(title_frame, text="InfiniLoop",
                              font=('Segoe UI', 24, 'bold'),
                              bg=self.colors['accent'], fg='white')
        title_label.pack(side='left')

        subtitle_label = tk.Label(title_frame, text="AI Music Generator - Loop finder",
                                 font=('Segoe UI', 11),
                                 bg=self.colors['accent'], fg='#e0e0e0')
        subtitle_label.pack(side='left', padx=(10, 0))

        self.status_indicator = tk.Label(content_frame, text="⚫",
                                        font=('Segoe UI', 20),
                                        bg=self.colors['accent'], fg='white')
        self.status_indicator.pack(side='right', padx=10)


    def create_controls_tab(self):
        controls_frame = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.notebook.add(controls_frame, text="🎛️ Controls")

        canvas = tk.Canvas(controls_frame, bg=self.colors['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(controls_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['bg'])

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        prompt_frame = self.create_card(scrollable_frame, "🎼 Generation")

        tk.Label(prompt_frame, text="Music prompt:",
                font=('Segoe UI', 11),
                bg=self.colors['bg_card'], fg=self.colors['text']).pack(anchor='w', pady=(0, 5))

        self.prompt_entry = tk.Entry(prompt_frame,
                                    font=('Segoe UI', 11),
                                    bg=self.colors['bg_secondary'],
                                    fg=self.colors['text'],
                                    insertbackground=self.colors['accent'],
                                    relief='flat',
                                    bd=10)
        self.prompt_entry.pack(fill='x', pady=(0, 10))
        self.prompt_entry.bind('<Return>', lambda e: self.start_loop())
        self.prompt_entry.insert(0, "e.g. ambient chill loop, jazz piano...")
        self.prompt_entry.bind('<FocusIn>', self.on_entry_focus_in)
        self.prompt_entry.bind('<FocusOut>', self.on_entry_focus_out)

        preset_frame = tk.Frame(prompt_frame, bg=self.colors['bg_card'])
        preset_frame.pack(fill='x', pady=(0, 10))

        tk.Label(preset_frame, text="Presets:",
                font=('Segoe UI', 10),
                bg=self.colors['bg_card'],
                fg=self.colors['text_secondary']).pack(side='left', padx=(0, 10))

        presets = {
            "Ambient":   "ambient, electronic, stargazing on a calm night, 4/4 nointro seamless loop",
            "Reggae":    "tropical, slow reggae-infused track, relaxing, laid-back rhythm, 4/4 nointro seamless loop",
            "EDM":       "disco, driving drum machine, synthesizer, bass, piano, clubby, 4/4, nointro seamless loop",
            "Rock":      "rock, guitars, drum, bass, up-lifting, moody, 4/4, nointro seamless loop",
            "Lofi Rap":  "slow calm lofi hiphop, music for studying, 4/4, punchy drums, tight bass, nointro seamless loop",
            "Synthwave": "80s retro synthwave, punchy drums, tight bass, nointro seamless loop"
        }

        for name, prompt in presets.items():
            btn = tk.Button(preset_frame,
                        text=name,
                        font=('Segoe UI', 9),
                        bg=self.colors['bg_secondary'],
                        fg=self.colors['text'],
                        activebackground=self.colors['accent'],
                        activeforeground='white',
                        relief='flat',
                        bd=0,
                        padx=15,
                        pady=5,
                        cursor='hand2',
                        command=lambda p=prompt: self.set_preset(p))
            btn.pack(side='left', padx=2)
            self.bind_hover(btn)

        button_frame = tk.Frame(prompt_frame, bg=self.colors['bg_card'])
        button_frame.pack(fill='x', pady=10)
        self.start_button = tk.Button(button_frame,
                                    text="▶️  START",
                                    font=('Segoe UI', 14, 'bold'),
                                    bg=self.colors['success'],
                                    fg='white',
                                    activebackground='#00f094',
                                    activeforeground='white',
                                    relief='flat',
                                    bd=0,
                                    padx=30,
                                    pady=15,
                                    cursor='hand2',
                                    command=self.start_loop)
        self.start_button.pack(side='left', fill='x', expand=True, padx=(0, 5))

        self.stop_button = tk.Button(button_frame,
                                    text="⏹️  STOP",
                                    font=('Segoe UI', 14, 'bold'),
                                    bg=self.colors['danger'],
                                    fg='white',
                                    activebackground='#ff5c4c',
                                    activeforeground='white',
                                    relief='flat',
                                    bd=0,
                                    padx=30,
                                    pady=15,
                                    cursor='hand2',
                                    state='disabled',
                                    command=self.stop_loop)
        self.stop_button.pack(side='left', fill='x', expand=True, padx=(5, 0))

        save_button = tk.Button(prompt_frame,
                            text="💾  Save Current Loop",
                            font=('Segoe UI', 11, 'bold'),
                            bg=self.colors['accent'],
                            fg='white',
                            activebackground=self.colors['accent_hover'],
                            activeforeground='white',
                            relief='flat',
                            bd=0,
                            padx=20,
                            pady=10,
                            cursor='hand2',
                            command=self.save_loop)
        save_button.pack(fill='x', pady=(5, 0))

        np_frame = self.create_card(scrollable_frame, "🎧 Loop info")

        self.np_info = {
            'title': tk.StringVar(value="---"),
            'artist': tk.StringVar(value="---"),
            'duration': tk.StringVar(value="---"),
            'genre': tk.StringVar(value="---")
        }


        for label, var in [("Title:", self.np_info['title']),
                        ("Artist:", self.np_info['artist']),
                        ("Duration:", self.np_info['duration']),
                        ("Prompt:", self.np_info['genre'])]:
            row = tk.Frame(np_frame, bg=self.colors['bg_card'])
            row.pack(fill='x', pady=5)
            tk.Label(row, text=label,
                    font=('Segoe UI', 10),
                    bg=self.colors['bg_card'],
                    fg=self.colors['text_secondary'],
                    width=10,
                    anchor='w').pack(side='left')
            label_value = tk.Label(row,
                textvariable=var,
                font=('Segoe UI', 11, 'bold'),
                bg=self.colors['bg_card'],
                fg=self.colors['text'],
                wraplength=300,
                justify='left',
                anchor='w'
            )
            label_value.pack(side='left', fill='x', expand=True)


    def update_min_duration(self):
        old_min_duration = getattr(self.app, 'min_song_duration', 30)
        self.app.min_song_duration = self.min_duration_var.get()

        if hasattr(self.app, 'update_generation_params'):
            self.app.update_generation_params({'min_song_duration': self.app.min_song_duration})

        self.save_settings()


    def update_model(self, event=None):

        selected_text = self.model_var.get()


        for model in ["small", "medium", "large"]:
            if selected_text.startswith(model):
                if self.is_model_installed(model):
                    self.app.model = model
                    self.save_settings()
                    self.status_label.config(text=f"🔴 Ready - Model: {self.app.model}")
                    self.capture_log(f"🔄 Model changed to: {model}")
                    break
                else:
                    self.app.model = model
                    self.save_settings()
                    self.status_label.config(text=f"🔴 Ready - Model: {self.app.model}")
                    self.capture_log(f"🔄 Model changed to: {model}")
                    messagebox.showwarning(
                        "Model Not Installed",
                        f"Model '{model}' will be downloaded at next play.\nExpect longer times for the first loop!\n\nYou can choose an installed model instead."
                    )
                    break


    def update_min_sample_duration(self):
        """Aggiorna la durata minima del loop e notifica il backend"""
        old_min_sample = getattr(self.app, 'min_sample_duration', 2.6)
        self.app.min_sample_duration = self.min_sample_var.get()

        # Notifica il backend del cambiamento
        if hasattr(self.app, 'update_generation_params'):
            self.app.update_generation_params({'min_sample_duration': self.app.min_sample_duration})

        self.save_settings()


    def update_duration_estimate(self):

        try:
            if not hasattr(self, 'duration_estimate_label'):
                return

            current_duration = self.duration_var.get()
            estimated_time = self.get_estimated_generation_time(current_duration)

            if estimated_time is not None:

                if estimated_time < 60:
                    time_text = f"~{int(round(estimated_time))}s"
                else:
                    mins = int(estimated_time // 60)
                    secs = int(estimated_time % 60)
                    if secs == 0:
                        time_text = f"~{mins}m"
                    else:
                        time_text = f"~{mins}m {secs}s"

                self.duration_estimate_label.config(text=f"({time_text} estimated generation time)")
            else:
                self.duration_estimate_label.config(text="")

        except Exception as e:

            if hasattr(self, 'duration_estimate_label'):
                self.duration_estimate_label.config(text="")


    def get_estimated_generation_time(self, target_duration):

        try:
            if not hasattr(self.app, 'benchmark_data') or not self.app.benchmark_data:
                return None


            duration_averages = {}
            duration_counts = {}

            for entry in self.app.benchmark_data:
                dur = entry["duration_requested"]
                gen_time = entry["generation_time"]

                if dur not in duration_averages:
                    duration_averages[dur] = 0
                    duration_counts[dur] = 0

                duration_averages[dur] += gen_time
                duration_counts[dur] += 1


            for dur in duration_averages:
                duration_averages[dur] = duration_averages[dur] / duration_counts[dur]


            if len(duration_averages) < 2:

                if target_duration in duration_averages:
                    return duration_averages[target_duration]
                return None


            if target_duration in duration_averages:
                return duration_averages[target_duration]


            sorted_durations = sorted(duration_averages.keys())


            if target_duration < sorted_durations[0]:

                x1, x2 = sorted_durations[0], sorted_durations[1]
            elif target_duration > sorted_durations[-1]:

                x1, x2 = sorted_durations[-2], sorted_durations[-1]
            else:

                x1 = x2 = None
                for i in range(len(sorted_durations) - 1):
                    if sorted_durations[i] <= target_duration <= sorted_durations[i + 1]:
                        x1, x2 = sorted_durations[i], sorted_durations[i + 1]
                        break


                if x1 is None:
                    x1, x2 = sorted_durations[0], sorted_durations[1]

            y1, y2 = duration_averages[x1], duration_averages[x2]


            if x2 == x1:
                return y1

            estimated_time = y1 + (y2 - y1) * (target_duration - x1) / (x2 - x1)
            return max(0, estimated_time)

        except Exception as e:
            return None


    def create_settings_tab(self):
        settings_frame = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.notebook.add(settings_frame, text="⚙️ Settings")

        settings_card = self.create_card(settings_frame, "⚙️ Configuration")

        # AI Model section
        model_frame = tk.Frame(settings_card, bg=self.colors['bg_card'])
        model_frame.pack(fill='x', pady=10)

        tk.Label(model_frame, text="AI Model:",
                font=('Segoe UI', 11),
                bg=self.colors['bg_card'],
                fg=self.colors['text']).pack(side='left')

        model_control_frame = tk.Frame(model_frame, bg=self.colors['bg_card'])
        model_control_frame.pack(side='left', padx=10, fill='x', expand=True)

        self.model_var = tk.StringVar(value=self.get_model_display_text(self.app.model))
        self.model_menu = ttk.Combobox(model_control_frame,
                                textvariable=self.model_var,
                                values=[self.get_model_display_text(m) for m in ["small", "medium", "large"]],
                                state="readonly",
                                width=25)
        self.model_menu.pack(side='left')
        self.model_menu.bind('<<ComboboxSelected>>', self.update_model)

        delete_frame = tk.Frame(model_control_frame, bg=self.colors['bg_card'])
        delete_frame.pack(side='left', padx=(10, 0))

        self.model_delete_buttons = {}
        for model in ["small", "medium", "large"]:
            btn = tk.Button(delete_frame,
                        text=f"🗑️ {model}",
                        font=('Segoe UI', 8),
                        bg=self.colors['danger'],
                        fg='white',
                        activebackground='#ff5c4c',
                        activeforeground='white',
                        relief='flat',
                        bd=0,
                        padx=8,
                        pady=4,
                        cursor='hand2',
                        state='normal' if self.is_model_installed(model) else 'disabled',
                        command=lambda m=model: self.delete_model(m))
            btn.pack(side='left', padx=1)
            self.model_delete_buttons[model] = btn

        model_help = tk.Label(model_frame,
                            text="(delete)",
                            font=('Segoe UI', 9),
                            bg=self.colors['bg_card'],
                            fg=self.colors['text_secondary'])
        model_help.pack(side='left', padx=(10, 0))

        # Sample length section
        duration_frame = tk.Frame(settings_card, bg=self.colors['bg_card'])
        duration_frame.pack(fill='x', pady=10)

        tk.Label(duration_frame, text="Sample length:",
                font=('Segoe UI', 11),
                bg=self.colors['bg_card'],
                fg=self.colors['text']).pack(side='left')

        self.duration_var = tk.IntVar(value=self.app.duration)
        duration_spin = tk.Spinbox(duration_frame,
                                from_=5, to=30,
                                textvariable=self.duration_var,
                                font=('Segoe UI', 11),
                                bg=self.colors['bg_secondary'],
                                fg=self.colors['text'],
                                buttonbackground=self.colors['accent'],
                                width=10,
                                command=self.update_duration)
        duration_spin.pack(side='left', padx=10)

        # Binding più reattivi per Sample length
        self.duration_var.trace('w', lambda *args: self.update_duration())
        duration_spin.bind('<Return>', lambda e: self.update_duration())
        duration_spin.bind('<FocusOut>', lambda e: self.update_duration())

        tk.Label(duration_frame, text="seconds",
                font=('Segoe UI', 10),
                bg=self.colors['bg_card'],
                fg=self.colors['text_secondary']).pack(side='left')

        self.duration_estimate_label = tk.Label(duration_frame, text="",
                                            font=('Segoe UI', 9),
                                            bg=self.colors['bg_card'],
                                            fg=self.colors['text_secondary'])
        self.duration_estimate_label.pack(side='left', padx=(10, 0))

        # Song duration section
        min_duration_frame = tk.Frame(settings_card, bg=self.colors['bg_card'])
        min_duration_frame.pack(fill='x', pady=10)

        tk.Label(min_duration_frame, text="Song duration:",
                font=('Segoe UI', 11),
                bg=self.colors['bg_card'],
                fg=self.colors['text']).pack(side='left')

        self.min_duration_var = tk.IntVar(value=getattr(self.app, 'min_song_duration', 30))
        min_duration_spin = tk.Spinbox(min_duration_frame,
                                    from_=10, to=300,
                                    textvariable=self.min_duration_var,
                                    font=('Segoe UI', 11),
                                    bg=self.colors['bg_secondary'],
                                    fg=self.colors['text'],
                                    buttonbackground=self.colors['accent'],
                                    width=10,
                                    command=self.update_min_duration)
        min_duration_spin.pack(side='left', padx=10)

        # Binding più reattivi per Song duration
        self.min_duration_var.trace('w', lambda *args: self.update_min_duration())
        min_duration_spin.bind('<Return>', lambda e: self.update_min_duration())
        min_duration_spin.bind('<FocusOut>', lambda e: self.update_min_duration())

        tk.Label(min_duration_frame, text="seconds",
                font=('Segoe UI', 10),
                bg=self.colors['bg_card'],
                fg=self.colors['text_secondary']).pack(side='left')

        help_label = tk.Label(min_duration_frame,
                            text="(minimum length of one song)",
                            font=('Segoe UI', 9),
                            bg=self.colors['bg_card'],
                            fg=self.colors['text_secondary'])
        help_label.pack(side='left', padx=(10, 0))

        # Loop length section
        min_sample_frame = tk.Frame(settings_card, bg=self.colors['bg_card'])
        min_sample_frame.pack(fill='x', pady=10)

        tk.Label(min_sample_frame, text="Loop length:",
                font=('Segoe UI', 11),
                bg=self.colors['bg_card'],
                fg=self.colors['text']).pack(side='left')

        self.min_sample_var = tk.DoubleVar(value=getattr(self.app, 'min_sample_duration', 2.6))
        min_sample_spin = tk.Spinbox(min_sample_frame,
                                    from_=1.0, to=10.0,
                                    increment=0.1,
                                    textvariable=self.min_sample_var,
                                    font=('Segoe UI', 11),
                                    bg=self.colors['bg_secondary'],
                                    fg=self.colors['text'],
                                    buttonbackground=self.colors['accent'],
                                    width=10,
                                    format="%.1f",
                                    command=self.update_min_sample_duration)
        min_sample_spin.pack(side='left', padx=10)

        # Binding più reattivi per Loop length
        self.min_sample_var.trace('w', lambda *args: self.update_min_sample_duration())
        min_sample_spin.bind('<Return>', lambda e: self.update_min_sample_duration())
        min_sample_spin.bind('<FocusOut>', lambda e: self.update_min_sample_duration())

        tk.Label(min_sample_frame, text="seconds",
                font=('Segoe UI', 10),
                bg=self.colors['bg_card'],
                fg=self.colors['text_secondary']).pack(side='left')

        sample_help_label = tk.Label(min_sample_frame,
                                    text="(minimum valid loop length)",
                                    font=('Segoe UI', 9),
                                    bg=self.colors['bg_card'],
                                    fg=self.colors['text_secondary'])
        sample_help_label.pack(side='left', padx=(10, 0))

        # Audio driver section
        driver_frame = tk.Frame(settings_card, bg=self.colors['bg_card'])
        driver_frame.pack(fill='x', pady=10)

        tk.Label(driver_frame, text="Audio driver:",
                font=('Segoe UI', 11),
                bg=self.colors['bg_card'],
                fg=self.colors['text']).pack(side='left')

        self.driver_var = tk.StringVar(value=self.app.audio_driver)
        driver_menu = ttk.Combobox(driver_frame,
                                textvariable=self.driver_var,
                                values=["pulse", "alsa", "dsp"],
                                state="readonly",
                                width=15)
        driver_menu.pack(side='left', padx=10)
        driver_menu.bind('<<ComboboxSelected>>', self.update_driver)

        # Debug section
        debug_card = self.create_card(settings_frame, "🐛 Debug")

        self.debug_var = tk.BooleanVar(value=self.app.debug_mode)
        debug_check = tk.Checkbutton(debug_card,
                                    text="Enable debug mode",
                                    variable=self.debug_var,
                                    font=('Segoe UI', 11),
                                    bg=self.colors['bg_card'],
                                    fg=self.colors['text'],
                                    selectcolor=self.colors['bg_secondary'],
                                    activebackground=self.colors['bg_card'],
                                    command=self.toggle_debug)
        debug_check.pack(anchor='w', pady=5)

        validate_btn = tk.Button(debug_card,
                                text="Validate Audio Files",
                                font=('Segoe UI', 10),
                                bg=self.colors['accent'],
                                fg='white',
                                activebackground=self.colors['accent_hover'],
                                relief='flat',
                                bd=0,
                                padx=15,
                                pady=8,
                                cursor='hand2',
                                command=self.validate_files)
        validate_btn.pack(anchor='w', pady=10)

        # Aggiorna la stima del tempo di durata
        self.update_duration_estimate()



    def create_log_tab(self):
        log_frame = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.notebook.add(log_frame, text="📋 Console")

        log_container = tk.Frame(log_frame, bg=self.colors['border'])
        log_container.pack(fill='both', expand=True, padx=10, pady=10)

        self.log_text = tk.Text(log_container,
                                font=('Consolas', 13),
                                bg=self.colors['bg_secondary'],
                                fg=self.colors['text_secondary'],
                                wrap='word',
                                relief='flat',
                                bd=0)
        self.log_text.pack(side='left', fill='both', expand=True, padx=2, pady=2)

        log_scrollbar = ttk.Scrollbar(log_container, command=self.log_text.yview)
        log_scrollbar.pack(side='right', fill='y')
        self.log_text.config(yscrollcommand=log_scrollbar.set)


        button_frame = tk.Frame(log_frame, bg=self.colors['bg'])
        button_frame.pack(pady=5)


        self.log_start_button = tk.Button(button_frame,
                                        text="▶️ START",
                                        font=('Segoe UI', 10, 'bold'),
                                        bg=self.colors['success'],
                                        fg='white',
                                        activebackground='#00f094',
                                        activeforeground='white',
                                        relief='flat',
                                        bd=0,
                                        padx=15,
                                        pady=8,
                                        cursor='hand2',
                                        command=self.start_loop)
        self.log_start_button.pack(side='left', padx=5)


        self.log_stop_button = tk.Button(button_frame,
                                        text="⏹️ STOP",
                                        font=('Segoe UI', 10, 'bold'),
                                        bg=self.colors['danger'],
                                        fg='white',
                                        activebackground='#ff5c4c',
                                        activeforeground='white',
                                        relief='flat',
                                        bd=0,
                                        padx=15,
                                        pady=8,
                                        cursor='hand2',
                                        state='disabled',
                                        command=self.stop_loop)
        self.log_stop_button.pack(side='left', padx=5)


        clear_btn = tk.Button(button_frame,
                            text="🗑️ Clear Log",
                            font=('Segoe UI', 10),
                            bg=self.colors['accent'],
                            fg='white',
                            activebackground=self.colors['accent_hover'],
                            relief='flat',
                            bd=0,
                            padx=15,
                            pady=8,
                            cursor='hand2',
                            command=self.clear_log)
        clear_btn.pack(side='left', padx=5)


    def create_benchmark_tab(self):

        bench_frame = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.notebook.add(bench_frame, text="📈 Statistics")

        card = self.create_card(bench_frame, "📊 Song generation statistics")

        self.benchmark_var = tk.BooleanVar(value=self.app.benchmark_enabled)
        bench_check = tk.Checkbutton(
            card,
            text="Enable statistics (local only)",
            variable=self.benchmark_var,
            font=('Segoe UI', 11),
            bg=self.colors['bg_card'],
            fg=self.colors['text'],
            selectcolor=self.colors['bg_secondary'],
            activebackground=self.colors['bg_card'],
            command=self.toggle_benchmark
        )
        bench_check.pack(anchor='w', pady=(0, 10))

        style = ttk.Style()
        style.configure("Benchmark.Treeview",
            background="white",
            foreground="black",
            fieldbackground="white",
            rowheight=24,
            font=('Segoe UI', 10)
        )
        style.configure("Benchmark.Treeview.Heading",
            font=('Segoe UI', 10, 'bold'),
            foreground='black',
            background='lightgray'
        )
        style.map("Benchmark.Treeview",
            background=[("selected", "black")],
            foreground=[("selected", "white")]
        )

        self.average_tree = ttk.Treeview(
            card,
            columns=("dur", "avg"),
            show="headings",
            height=8,
            style="Benchmark.Treeview"
        )
        self.average_tree.heading("dur", text="Sample duration")
        self.average_tree.heading("avg", text="Average generation time")
        self.average_tree.column("dur", anchor='center', width=180)
        self.average_tree.column("avg", anchor='center', width=220)
        self.average_tree.pack(fill='x', expand=False, pady=(0, 10))

        self.app.load_benchmark_data()
        self.update_benchmark_table()

        reset_btn = tk.Button(
            card,
            text="🗑️  Reset Data",
            font=('Segoe UI', 10),
            bg=self.colors['danger'],
            fg='white',
            activebackground='#ff5c4c',
            relief='flat',
            bd=0,
            padx=10,
            pady=8,
            cursor='hand2',
            command=self.reset_benchmark_data
        )
        reset_btn.pack(pady=10)


    def toggle_benchmark(self):

        self.app.benchmark_enabled = self.benchmark_var.get()
        self.save_settings()


    def update_benchmark_table(self):
        try:
            if not hasattr(self, "average_tree"):
                return

            for row in self.average_tree.get_children():
                self.average_tree.delete(row)

            data = self.app.benchmark_data


            buckets = {}
            for entry in data:
                dur = entry["duration_requested"]
                buckets.setdefault(dur, []).append(entry["generation_time"])

            for dur, times in sorted(buckets.items()):
                avg_time = sum(times) / len(times)
                avg_secs = int(round(avg_time))
                avg_mins = round(avg_time / 60, 1)
                formatted = f"{avg_secs}s ({avg_mins} min)"
                self.average_tree.insert("", "end", values=(f"{dur}s", formatted))


            self.update_duration_estimate()

        except Exception as e:
            self.capture_log(f"⚠️ Failed to update benchmark table: {e}")


    def reset_benchmark_data(self):

        self.app.reset_benchmark_data()
        self.update_benchmark_table()


    def create_status_bar(self, parent):

        status_frame = tk.Frame(parent, bg=self.colors['bg_secondary'], height=40)
        status_frame.pack(fill='x', pady=(10, 0))
        status_frame.pack_propagate(False)


        self.status_label = tk.Label(status_frame,
                                     text=f"🔴 Ready - Model: {self.app.model}",
                                     font=('Segoe UI', 10, 'bold'),
                                     bg=self.colors['bg_secondary'],
                                     fg=self.colors['text'])
        self.status_label.pack(side='left', padx=15, pady=10)

        self.generation_label = tk.Label(status_frame,
                                         text="",
                                         font=('Segoe UI', 10),
                                         bg=self.colors['bg_secondary'],
                                         fg=self.colors['text_secondary'])
        self.generation_label.pack(side='left', padx=15)

        self.progress_label = tk.Label(status_frame,
                                       text="",
                                       font=('Segoe UI', 10),
                                       bg=self.colors['bg_secondary'],
                                       fg=self.colors['accent'])
        self.progress_label.pack(side='right', padx=15)


    def create_card(self, parent, title):

        card = tk.Frame(parent, bg=self.colors['bg_card'], relief='raised', bd=2)
        card.pack(fill='x', padx=10, pady=10)

        title_label = tk.Label(card, text=title,
                              font=('Segoe UI', 12, 'bold'),
                              bg=self.colors['bg_card'],
                              fg=self.colors['accent'])
        title_label.pack(anchor='w', padx=15, pady=(10, 5))

        content = tk.Frame(card, bg=self.colors['bg_card'])
        content.pack(fill='both', expand=True, padx=15, pady=(5, 15))

        return content


    def bind_hover(self, widget):

        original_bg = widget['bg']
        def on_enter(e):
            widget['bg'] = self.colors['accent']
        def on_leave(e):
            widget['bg'] = original_bg
        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)


    def on_entry_focus_in(self, event):

        if self.prompt_entry.get() == "e.g. ambient chill loop, jazz piano...":
            self.prompt_entry.delete(0, 'end')


    def on_entry_focus_out(self, event):

        if not self.prompt_entry.get():
            self.prompt_entry.insert(0, "e.g. ambient chill loop, jazz piano...")


    def set_preset(self, preset):

        self.prompt_entry.delete(0, 'end')
        self.prompt_entry.insert(0, preset)


    def capture_log(self, message):

        self.original_log(message)
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {message}\n"
        self.log_queue.put(formatted_msg)

        if "🎧 ORA IN RIPRODUZIONE:" in message or "Sample generato" in message or "Ottenuto loop perfetto" in message:

            self.log_queue.put("__UPDATE_NOW_PLAYING__")

    def _run_loop(self, prompt):

        self.app.start_loop(prompt)

    def start_loop(self):
        prompt = self.prompt_entry.get().strip()
        if prompt == "e.g. ambient chill loop, jazz piano..." or not prompt:
            messagebox.warning("Warning", "Please enter a music prompt!")
            return

        self.app.last_prompt = prompt
        self.save_settings()
        self.is_running = True


        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')


        self.log_start_button.config(state='disabled')
        self.log_stop_button.config(state='normal')

        self.prompt_entry.config(state='disabled')
        self.status_indicator.config(text="🟢")
        self.status_label.config(text=f"🟢 PLAYING")

        thread = threading.Thread(target=self._run_loop, args=(prompt,), daemon=True)
        thread.start()


    def stop_loop(self):
        self.app.stop_loop()
        self.is_running = False


        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')


        self.log_start_button.config(state='normal')
        self.log_stop_button.config(state='disabled')

        self.prompt_entry.config(state='normal')
        self.status_indicator.config(text="🔴")
        self.status_label.config(text=f"🔴 STOPPED - Model: {self.app.model}")

        self.current_loop_file = None
        self.last_title = None
        self.last_artist = None

        for var in self.np_info.values():
            var.set("---")


    def save_loop(self):

        if not self.app.is_playing:
            messagebox.warning("Warning", "No loop playing!")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )

        if filename:
            if self.app.save_current_loop(filename):
                messagebox.showinfo("Success", f"Loop saved: {filename}")
            else:
                messagebox.showerror("Error", "Unable to save loop")


    def validate_files(self):

        current_valid = self.app.validate_audio_file(self.app.CURRENT)
        next_valid = self.app.validate_audio_file(self.app.NEXT)

        msg = f"Current file: {'✅ VALID' if current_valid else '❌ INVALID'}\n"
        msg += f"Next file: {'✅ VALID' if next_valid else '❌ INVALID'}"

        messagebox.showinfo("File Validation", msg)


    def clear_log(self):

        self.log_text.delete(1.0, 'end')


    def update_duration(self):
        """Aggiorna la durata del sample e notifica il backend"""
        old_duration = self.app.duration
        self.app.duration = self.duration_var.get()

        # Notifica il backend del cambiamento se sta generando
        if hasattr(self.app, 'update_generation_params'):
            self.app.update_generation_params({'duration': self.app.duration})

        self.save_settings()
        self.update_duration_estimate()

        # Log del cambiamento
        if old_duration != self.app.duration:
            self.capture_log(f"🔄 Sample length: {self.app.duration}s")



    def update_driver(self, event=None):

        self.app.audio_driver = self.driver_var.get()
        os.environ["SDL_AUDIODRIVER"] = self.app.audio_driver
        self.save_settings()


    def toggle_debug(self):

        self.app.debug_mode = self.debug_var.get()
        self.save_settings()

    def update_loop(self):
        update_now_playing = False
        while not self.log_queue.empty():
            msg = self.log_queue.get()
            if msg == "__UPDATE_NOW_PLAYING__":
                update_now_playing = True
                if hasattr(self, "update_benchmark_table"):
                    self.update_benchmark_table()
            else:
                self.log_text.insert('end', msg)
                self.log_text.see('end')

        if self.app.is_playing:
            if self.app.is_generating:
                self.generation_label.config(text=f"{self.app.generation_status}")
                self.progress_label.config(text="⏳")
            else:
                self.generation_label.config(text="")

                if hasattr(self.app, 'current_loop_start_time') and self.app.current_loop_start_time:
                    timing = self.app.get_current_loop_timing()

                    if timing['min_time_satisfied']:
                        if timing['can_swap']:
                            timing_text = f"✅ {timing['elapsed']:.0f}s (ready)"
                        else:
                            timing_text = f"⏳ {timing['elapsed']:.0f}s (waiting NEXT)"
                    else:
                        remaining = timing['remaining_min_time']
                        timing_text = f"⏱️ {timing['elapsed']:.0f}s / {self.app.min_song_duration}s"

                    self.progress_label.config(text=timing_text)
                else:
                    self.progress_label.config(text="")

            current_file = self.app.CURRENT if hasattr(self.app, 'CURRENT') else None

            if update_now_playing or (current_file and current_file != self.current_loop_file):
                self.current_loop_file = current_file

                new_title = self.app.get_random_title()
                new_artist = self.app.get_random_artist()

                while new_title == self.last_title:
                    new_title = self.app.get_random_title()
                while new_artist == self.last_artist:
                    new_artist = self.app.get_random_artist()

                self.last_title = new_title
                self.last_artist = new_artist

                self.np_info['title'].set(new_title)
                self.np_info['artist'].set(new_artist)

                try:
                    duration = self.app.get_duration(self.app.CURRENT)
                    if duration > 0:
                        self.np_info['duration'].set(f"{duration:.1f} seconds")
                    else:
                        self.np_info['duration'].set("Calculating...")
                except:
                    self.np_info['duration'].set("---")

                if hasattr(self.app, 'PROMPT') and self.app.PROMPT:
                    self.np_info['genre'].set(self.app.PROMPT)

            if hasattr(self.app, 'current_loop_start_time') and self.app.current_loop_start_time:
                timing = self.app.get_current_loop_timing()
                try:
                    base_duration = self.app.get_duration(self.app.CURRENT)
                    if base_duration > 0:
                        if timing['min_time_satisfied']:
                            status_icon = "✅" if timing['can_swap'] else "⏳"
                            duration_text = f"{base_duration:.1f}s loop • {status_icon} {timing['elapsed']:.0f}s played"
                        else:
                            remaining = timing['remaining_min_time']
                            duration_text = f"{base_duration:.1f}s loop • ⏱️ {remaining:.0f}s left"

                        self.np_info['duration'].set(duration_text)
                except:
                    pass

        else:
            self.generation_label.config(text="")
            self.progress_label.config(text="")
            self.current_loop_file = None
            self.last_title = None
            self.last_artist = None

        self.root.after(100, self.update_loop)


    def save_settings(self):
        settings = {
            "model": self.app.model,
            "duration": self.app.duration,
            "min_song_duration": getattr(self.app, 'min_song_duration', 30),
            "min_sample_duration": getattr(self.app, 'min_sample_duration', 2.6),
            "audio_driver": self.app.audio_driver,
            "debug_mode": self.app.debug_mode,
            "benchmark_enabled": self.benchmark_var.get(),
            "last_prompt": getattr(self.app, "last_prompt", self.prompt_entry.get().strip())
        }

        with open("infiniloop_settings.json", "w") as f:
            json.dump(settings, f)



    def load_settings(self):
        try:
            with open("infiniloop_settings.json", "r") as f:
                settings = json.load(f)

            self.app.model = settings.get("model", self.app.model)
            if hasattr(self, 'model_var'):
                self.model_var.set(self.app.model)

            self.app.duration = settings.get("duration", self.app.duration)
            self.duration_var.set(self.app.duration)

            self.app.min_song_duration = settings.get("min_song_duration", 30)
            if hasattr(self, 'min_duration_var'):
                self.min_duration_var.set(self.app.min_song_duration)


            self.app.min_sample_duration = settings.get("min_sample_duration", 2.6)
            if hasattr(self, 'min_sample_var'):
                self.min_sample_var.set(self.app.min_sample_duration)

            self.app.audio_driver = settings.get("audio_driver", self.app.audio_driver)
            self.driver_var.set(self.app.audio_driver)

            last_prompt = settings.get("last_prompt", "")
            if last_prompt:
                self.prompt_entry.delete(0, 'end')
                self.prompt_entry.insert(0, last_prompt)
                self.app.last_prompt = last_prompt

            self.app.debug_mode = settings.get("debug_mode", self.app.debug_mode)
            self.debug_var.set(self.app.debug_mode)

            self.app.benchmark_enabled = settings.get("benchmark_enabled", True)
            if hasattr(self, 'benchmark_var'):
                self.benchmark_var.set(self.app.benchmark_enabled)

            os.environ["SDL_AUDIODRIVER"] = self.app.audio_driver

            if hasattr(self, 'model_menu'):
                self.update_model_ui()

        except Exception as e:
            self.capture_log(f"⚠️ Failed to load settings: {e}")



    def on_closing(self):

        if self.app.is_playing:
            result = messagebox.askyesno(
                "Confirm",
                "Loop still running.\nStop loop and exit?"
            )
            if result:
                self.app.stop_loop()
                self.save_settings()
                self.root.destroy()
        else:
            self.save_settings()
            self.root.destroy()


def main():

    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

    root = tk.Tk()

    try:
        root.iconphoto(False, tk.PhotoImage(data='''
            R0lGODlhEAAQAPeQAJqJJtq0ANOvANq1ANu1ANaxAN21ANy1ANyzANuyANqxANqyANmxANmw
            ANiwANivANeuANavANWuANStANOsANKsANGrANCqAM+pAM6oAM2nAMynAMumAMqlAMmkAMij
            AMejAMaiAMWhAMSgAMOfAMKeAMGdAMCcAL+bAL6aAL2ZALyYALuXALqWALmVALiUALeUALaT
            ALWSALSRALOQALKPALGOALCNAK+MAK6LAK2KAKyJAKuIAKqHAKmGAKiFAKaEAKWDAKSCAKOB
            AKKAAJ9+AJ59AJ18AJx7AJt6AJp5AJl4AJh3AJZ1AJV0AJRzAJNyAJJxAJFwAI9uAI5tAI1s
            AIxrAIpoAIhnAIdmAIZlAIVkAIRjAINiAIJhAIFgAH9eAH5dAH1cAHxbAHtaAHpZAHlYAHhX
            AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
        '''))
    except:
        pass

    app = InfiniLoopGUI(root)

    root.update_idletasks()
    width = 700
    height = 850
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')

    root.mainloop()


if __name__ == "__main__":
    main()
