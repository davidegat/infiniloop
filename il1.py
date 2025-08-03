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

# Import original class
from ilterm import InfiniLoopTerminal

class InfiniLoopGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎵 InfiniLoop - AI Music Generator")
        self.root.geometry("600x850")
        self.root.minsize(500, 700)
        self.root.maxsize(700, 950)

        # Initialize core app
        self.app = InfiniLoopTerminal()
        self.is_running = False
        self.log_queue = Queue()

        # Variables for tracking current loop
        self.current_loop_file = None
        self.last_title = None
        self.last_artist = None

        # Override log_message method
        self.original_log = self.app.log_message
        self.app.log_message = self.capture_log

        # Modern dark theme colors
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

        # Setup styles
        self.setup_styles()

        # Create UI
        self.create_ui()

        # Load saved settings
        self.load_settings()

        # Start update loop
        self.update_loop()

        # Handle closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_styles(self):
        """Configure ttk styles for modern dark theme"""
        style = ttk.Style()
        style.theme_use('clam')

        # Colori generali
        style.configure(".",
            background=self.colors['bg'],
            foreground=self.colors['text'],
            bordercolor=self.colors['border'],
            focuscolor='none',
            relief='flat')

        # Frame
        style.configure("Card.TFrame",
            background=self.colors['bg_card'],
            relief='raised',
            borderwidth=2)

        # Etichette
        style.configure("Heading.TLabel",
            background=self.colors['bg'],
            foreground=self.colors['accent'],
            font=('Segoe UI', 14, 'bold'))

        style.configure("Title.TLabel",
            background=self.colors['bg'],
            foreground=self.colors['accent'],
            font=('Segoe UI', 24, 'bold'))

        # Bottoni
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

        # Notebook
        style.configure("TNotebook",
            background=self.colors['bg'],
            borderwidth=0)

        # Base tab (deselezionata): compatta
        style.configure("TNotebook.Tab",
            background=self.colors['bg_secondary'],
            foreground=self.colors['text_secondary'],
            padding=[10, 5],  # <-- linguetta più piccola
            font=('Segoe UI', 10))

        # Tab selezionata: linguetta più grande
        style.map("TNotebook.Tab",
            background=[('selected', self.colors['bg_card'])],
            foreground=[('selected', self.colors['text'])],
            padding=[('selected', [20, 12])])  # <-- linguetta più grande se selezionata


    def create_ui(self):
        """Create user interface"""
        self.root.configure(bg=self.colors['bg'])

        # Main container
        main_container = tk.Frame(self.root, bg=self.colors['bg'])
        main_container.pack(fill='both', expand=True, padx=20, pady=20)

        # Header
        self.create_header(main_container)

        # Notebook for tabs
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill='both', expand=True, pady=(20, 0))

        # Tab 1: Controls
        self.create_controls_tab()

        # Tab 2: Settings
        self.create_settings_tab()

        # Tab 3: Log
        self.create_log_tab()

        # Status bar
        self.create_status_bar(main_container)

    def create_header(self, parent):
        """Create header with title and status"""
        header_frame = tk.Frame(parent, bg=self.colors['accent'], height=80)
        header_frame.pack(fill='x', pady=(0, 10))
        header_frame.pack_propagate(False)

        # Gradient effect (simulated with frames)
        gradient_frame = tk.Frame(header_frame, bg=self.colors['accent'])
        gradient_frame.pack(fill='both', expand=True, padx=2, pady=2)

        # Content
        content_frame = tk.Frame(gradient_frame, bg=self.colors['accent'])
        content_frame.pack(fill='both', expand=True, padx=20)

        # Logo and title
        title_frame = tk.Frame(content_frame, bg=self.colors['accent'])
        title_frame.pack(side='left', fill='y')

        logo_label = tk.Label(title_frame, text="🎵", font=('Segoe UI', 32),
                             bg=self.colors['accent'], fg='white')
        logo_label.pack(side='left', padx=(0, 15))

        title_label = tk.Label(title_frame, text="InfiniLoop",
                              font=('Segoe UI', 24, 'bold'),
                              bg=self.colors['accent'], fg='white')
        title_label.pack(side='left')

        subtitle_label = tk.Label(title_frame, text="AI Music Generator",
                                 font=('Segoe UI', 11),
                                 bg=self.colors['accent'], fg='#e0e0e0')
        subtitle_label.pack(side='left', padx=(10, 0))

        # Status indicator
        self.status_indicator = tk.Label(content_frame, text="⚫",
                                        font=('Segoe UI', 20),
                                        bg=self.colors['accent'], fg='white')
        self.status_indicator.pack(side='right', padx=10)

    def create_controls_tab(self):
        """Create main controls tab"""
        controls_frame = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.notebook.add(controls_frame, text="🎛️ Controls")

        # Scrollable frame
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

        # Prompt section
        prompt_frame = self.create_card(scrollable_frame, "🎼 Music Generation")

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

        # Placeholder text
        self.prompt_entry.insert(0, "e.g. ambient chill loop, jazz piano...")
        self.prompt_entry.bind('<FocusIn>', self.on_entry_focus_in)
        self.prompt_entry.bind('<FocusOut>', self.on_entry_focus_out)

        # Preset buttons
        preset_frame = tk.Frame(prompt_frame, bg=self.colors['bg_card'])
        preset_frame.pack(fill='x', pady=(0, 10))

        tk.Label(preset_frame, text="Presets:",
                font=('Segoe UI', 10),
                bg=self.colors['bg_card'],
                fg=self.colors['text_secondary']).pack(side='left', padx=(0, 10))

        # Enhanced presets for better results
        presets = {
            "Ambient": "ambient atmospheric ethereal soundscape relaxing seamless nointro loop",
            "Reggae": "slow calm reggae riddim classic beats seamless nointro loop",
            "Electronic": "electronic synth dance beat techno edm seamless nointro loop",
            "Classical": "classical orchestral symphony piano violin seamless nointro loop",
            "Rock": "rock guitar drums electric bass heavy seamless nointro loop",
            "Lofi Rap": "lofi calm rap beats seamless nointro loop"
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

        # Control buttons
        button_frame = tk.Frame(prompt_frame, bg=self.colors['bg_card'])
        button_frame.pack(fill='x', pady=10)

        self.start_button = tk.Button(button_frame,
                                      text="▶️  START LOOP",
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

        # Save button
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

        # Now Playing section
        np_frame = self.create_card(scrollable_frame, "🎧 Now Playing")

        self.np_info = {
            'title': tk.StringVar(value="---"),
            'artist': tk.StringVar(value="---"),
            'duration': tk.StringVar(value="---"),
            'genre': tk.StringVar(value="---")
        }

        for label, var in [("Title:", self.np_info['title']),
                           ("Artist:", self.np_info['artist']),
                           ("Duration:", self.np_info['duration']),
                           ("Genre:", self.np_info['genre'])]:
            row = tk.Frame(np_frame, bg=self.colors['bg_card'])
            row.pack(fill='x', pady=5)
            tk.Label(row, text=label,
                    font=('Segoe UI', 10),
                    bg=self.colors['bg_card'],
                    fg=self.colors['text_secondary'],
                    width=10,
                    anchor='w').pack(side='left')
            tk.Label(row, textvariable=var,
                    font=('Segoe UI', 11, 'bold'),
                    bg=self.colors['bg_card'],
                    fg=self.colors['text']).pack(side='left')

    def create_settings_tab(self):
        """Create settings tab"""
        settings_frame = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.notebook.add(settings_frame, text="⚙️ Settings")

        # Settings card
        settings_card = self.create_card(settings_frame, "⚙️ Configuration")

        # Duration
        duration_frame = tk.Frame(settings_card, bg=self.colors['bg_card'])
        duration_frame.pack(fill='x', pady=10)

        tk.Label(duration_frame, text="Generation duration:",
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

        tk.Label(duration_frame, text="seconds",
                font=('Segoe UI', 10),
                bg=self.colors['bg_card'],
                fg=self.colors['text_secondary']).pack(side='left')

        # Audio driver
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

        # Fixed settings info
        info_frame = tk.Frame(settings_card, bg=self.colors['bg_card'])
        info_frame.pack(fill='x', pady=20)

        for label, value in [("AI Model:", "medium (balanced)"),
                             ("Algorithm:", "Advanced with fallback"),
                             ("Crossfade:", "1ms (minimum)")]:
            row = tk.Frame(info_frame, bg=self.colors['bg_card'])
            row.pack(fill='x', pady=3)
            tk.Label(row, text=label,
                    font=('Segoe UI', 10),
                    bg=self.colors['bg_card'],
                    fg=self.colors['text_secondary'],
                    width=15,
                    anchor='w').pack(side='left')
            tk.Label(row, text=value,
                    font=('Segoe UI', 10, 'bold'),
                    bg=self.colors['bg_card'],
                    fg=self.colors['accent']).pack(side='left')

        # Debug card
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

    def create_log_tab(self):
        """Create log tab"""
        log_frame = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.notebook.add(log_frame, text="📋 Log")

        # Log text area
        log_container = tk.Frame(log_frame, bg=self.colors['border'])
        log_container.pack(fill='both', expand=True, padx=10, pady=10)

        self.log_text = tk.Text(log_container,
                                font=('Consolas', 10),
                                bg=self.colors['bg_secondary'],
                                fg=self.colors['text_secondary'],
                                wrap='word',
                                relief='flat',
                                bd=0)
        self.log_text.pack(side='left', fill='both', expand=True, padx=2, pady=2)

        log_scrollbar = ttk.Scrollbar(log_container, command=self.log_text.yview)
        log_scrollbar.pack(side='right', fill='y')
        self.log_text.config(yscrollcommand=log_scrollbar.set)

        # Clear button
        clear_btn = tk.Button(log_frame,
                             text="🗑️  Clear Log",
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
        clear_btn.pack(pady=5)

    def create_status_bar(self, parent):
        """Create status bar"""
        status_frame = tk.Frame(parent, bg=self.colors['bg_secondary'], height=40)
        status_frame.pack(fill='x', pady=(10, 0))
        status_frame.pack_propagate(False)

        # Status label
        self.status_label = tk.Label(status_frame,
                                     text="🔴 Ready",
                                     font=('Segoe UI', 10, 'bold'),
                                     bg=self.colors['bg_secondary'],
                                     fg=self.colors['text'])
        self.status_label.pack(side='left', padx=15, pady=10)

        # Generation status
        self.generation_label = tk.Label(status_frame,
                                         text="",
                                         font=('Segoe UI', 10),
                                         bg=self.colors['bg_secondary'],
                                         fg=self.colors['text_secondary'])
        self.generation_label.pack(side='left', padx=15)

        # Progress bar (simulated with label)
        self.progress_label = tk.Label(status_frame,
                                       text="",
                                       font=('Segoe UI', 10),
                                       bg=self.colors['bg_secondary'],
                                       fg=self.colors['accent'])
        self.progress_label.pack(side='right', padx=15)

    def create_card(self, parent, title):
        """Create a styled card/panel"""
        card = tk.Frame(parent, bg=self.colors['bg_card'], relief='raised', bd=2)
        card.pack(fill='x', padx=10, pady=10)

        # Title
        title_label = tk.Label(card, text=title,
                              font=('Segoe UI', 12, 'bold'),
                              bg=self.colors['bg_card'],
                              fg=self.colors['accent'])
        title_label.pack(anchor='w', padx=15, pady=(10, 5))

        # Content frame
        content = tk.Frame(card, bg=self.colors['bg_card'])
        content.pack(fill='both', expand=True, padx=15, pady=(5, 15))

        return content

    def bind_hover(self, widget):
        """Add hover effect to widgets"""
        original_bg = widget['bg']
        def on_enter(e):
            widget['bg'] = self.colors['accent']
        def on_leave(e):
            widget['bg'] = original_bg
        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

    def on_entry_focus_in(self, event):
        """Remove placeholder on focus"""
        if self.prompt_entry.get() == "e.g. ambient chill loop, jazz piano...":
            self.prompt_entry.delete(0, 'end')

    def on_entry_focus_out(self, event):
        """Restore placeholder if empty"""
        if not self.prompt_entry.get():
            self.prompt_entry.insert(0, "e.g. ambient chill loop, jazz piano...")

    def set_preset(self, preset):
        """Set an enhanced preset in the prompt"""
        self.prompt_entry.delete(0, 'end')
        self.prompt_entry.insert(0, preset)

    def capture_log(self, message):
        """Capture log messages from original app"""
        self.original_log(message)
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {message}\n"
        self.log_queue.put(formatted_msg)

        # Intercept swap messages to update Now Playing
        if "🎧 ORA IN RIPRODUZIONE:" in message or "Sample generato" in message or "Ottenuto loop perfetto" in message:
            # Signal to update track info
            self.log_queue.put("__UPDATE_NOW_PLAYING__")

    def start_loop(self):
        """Start the music loop"""
        prompt = self.prompt_entry.get().strip()
        if prompt == "e.g. ambient chill loop, jazz piano..." or not prompt:
            messagebox.warning("Warning", "Please enter a music prompt!")
            return

        self.is_running = True
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.prompt_entry.config(state='disabled')
        self.status_indicator.config(text="🟢")
        self.status_label.config(text="🟢 PLAYING")

        # Start in separate thread
        thread = threading.Thread(target=self._run_loop, args=(prompt,), daemon=True)
        thread.start()

    def _run_loop(self, prompt):
        """Run loop in separate thread"""
        try:
            self.app.start_loop(prompt)
        except Exception as e:
            self.log_queue.put(f"❌ Error: {str(e)}\n")

    def stop_loop(self):
        """Stop the music loop"""
        self.app.stop_loop()
        self.is_running = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.prompt_entry.config(state='normal')
        self.status_indicator.config(text="🔴")
        self.status_label.config(text="🔴 STOPPED")

        # Reset tracking variables
        self.current_loop_file = None
        self.last_title = None
        self.last_artist = None

        # Reset now playing
        for var in self.np_info.values():
            var.set("---")

    def save_loop(self):
        """Save current loop"""
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
        """Validate current audio files"""
        current_valid = self.app.validate_audio_file(self.app.CURRENT)
        next_valid = self.app.validate_audio_file(self.app.NEXT)

        msg = f"Current file: {'✅ VALID' if current_valid else '❌ INVALID'}\n"
        msg += f"Next file: {'✅ VALID' if next_valid else '❌ INVALID'}"

        messagebox.showinfo("File Validation", msg)

    def clear_log(self):
        """Clear the log"""
        self.log_text.delete(1.0, 'end')

    def update_duration(self):
        """Update generation duration"""
        self.app.duration = self.duration_var.get()
        self.save_settings()

    def update_driver(self, event=None):
        """Update audio driver"""
        self.app.audio_driver = self.driver_var.get()
        os.environ["SDL_AUDIODRIVER"] = self.app.audio_driver
        self.save_settings()

    def toggle_debug(self):
        """Toggle debug mode"""
        self.app.debug_mode = self.debug_var.get()
        self.save_settings()

    def update_loop(self):
        """UI update loop"""
        # Process log queue
        update_now_playing = False
        while not self.log_queue.empty():
            msg = self.log_queue.get()
            if msg == "__UPDATE_NOW_PLAYING__":
                update_now_playing = True
            else:
                self.log_text.insert('end', msg)
                self.log_text.see('end')

        # Update status
        if self.app.is_playing:
            if self.app.is_generating:
                self.generation_label.config(text=f"🎼 {self.app.generation_status}")
                self.progress_label.config(text="⏳ Wait...")
            else:
                self.generation_label.config(text="")
                self.progress_label.config(text="")

            # Check if current file changed or update signal received
            current_file = self.app.CURRENT if hasattr(self.app, 'CURRENT') else None

            if update_now_playing or (current_file and current_file != self.current_loop_file):
                # Update file tracking
                self.current_loop_file = current_file

                # Generate new track data
                new_title = self.app.get_random_title()
                new_artist = self.app.get_random_artist()

                # Ensure they're different from previous
                while new_title == self.last_title:
                    new_title = self.app.get_random_title()
                while new_artist == self.last_artist:
                    new_artist = self.app.get_random_artist()

                self.last_title = new_title
                self.last_artist = new_artist

                # Update UI
                self.np_info['title'].set(new_title)
                self.np_info['artist'].set(new_artist)

                # Update duration
                try:
                    duration = self.app.get_duration(self.app.CURRENT)
                    if duration > 0:
                        self.np_info['duration'].set(f"{duration:.1f} seconds")
                    else:
                        self.np_info['duration'].set("Calculating...")
                except:
                    self.np_info['duration'].set("---")

                # Update genre with current prompt
                if hasattr(self.app, 'PROMPT') and self.app.PROMPT:
                    self.np_info['genre'].set(self.app.PROMPT)

        else:
            self.generation_label.config(text="")
            self.progress_label.config(text="")
            # Reset tracking when stopped
            self.current_loop_file = None
            self.last_title = None
            self.last_artist = None

        # Repeat after 100ms
        self.root.after(100, self.update_loop)

    def save_settings(self):
        """Save settings"""
        settings = {
            'duration': self.app.duration,
            'driver': self.app.audio_driver,
            'debug': self.app.debug_mode,
            'last_prompt': self.prompt_entry.get()
        }
        try:
            with open('infiniloop_settings.json', 'w') as f:
                json.dump(settings, f, indent=2)
        except:
            pass

    def load_settings(self):
        """Load saved settings"""
        try:
            with open('infiniloop_settings.json', 'r') as f:
                settings = json.load(f)
                self.app.duration = settings.get('duration', 15)
                self.app.audio_driver = settings.get('driver', 'pulse')
                self.app.debug_mode = settings.get('debug', False)

                # Update UI
                self.duration_var.set(self.app.duration)
                self.driver_var.set(self.app.audio_driver)
                self.debug_var.set(self.app.debug_mode)

                # Restore last prompt
                last_prompt = settings.get('last_prompt', '')
                if last_prompt and last_prompt != "e.g. ambient chill loop, jazz piano...":
                    self.prompt_entry.delete(0, 'end')
                    self.prompt_entry.insert(0, last_prompt)
        except:
            pass

    def on_closing(self):
        """Handle application closing"""
        if self.app.is_playing:
            result = messagebox.askyesno(
                "Confirm Exit",
                "The loop is still running.\nDo you want to stop and exit?"
            )
            if result:
                self.app.stop_loop()
                self.save_settings()
                self.root.destroy()
        else:
            self.save_settings()
            self.root.destroy()

def main():
    """Application entry point"""
    # Set environment variables
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

    # Create main window
    root = tk.Tk()

    # Set icon (if available)
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

    # Create application
    app = InfiniLoopGUI(root)

    # Center the window
    root.update_idletasks()
    width = 700
    height = 850
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')

    # Start main loop
    root.mainloop()

if __name__ == "__main__":
    main()
