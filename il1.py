import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import threading
import os
from ilterm import InfiniLoopTerminal

class InfiniLoopUI:
    def __init__(self, root):
        self.root = root
        self.root.title("INFINI LOOP GUI")
        self.root.geometry("640x200")
        self.root.configure(bg="#121212")
        self.app = InfiniLoopTerminal()
        self.app.debug_mode = False

        self.prompt_var = tk.StringVar(value="lofi relaxing rap beats")
        self.duration_var = tk.IntVar(value=15)
        self.driver_var = tk.StringVar(value="pulse")

        self.setup_ui()

    def setup_ui(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("Dark.TFrame", background="#121212")
        style.configure("Dark.TLabel", background="#121212", foreground="#cccccc", font=("Segoe UI", 10))
        style.configure("Dark.TEntry", fieldbackground="#1f1f1f", background="#1f1f1f", foreground="#ffffff")
        style.configure("Dark.TCombobox", fieldbackground="#1f1f1f", background="#1f1f1f", foreground="#ffffff")
        style.configure("Dark.TButton", background="#1e1e1e", foreground="#00ffc3", font=("Segoe UI", 10, "bold"))

        main_frame = ttk.Frame(self.root, style="Dark.TFrame", padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Prompt
        prompt_label = ttk.Label(main_frame, text="🎵 Prompt musicale:", style="Dark.TLabel")
        prompt_label.grid(row=0, column=0, sticky="w", padx=(0, 10))
        prompt_entry = ttk.Entry(main_frame, textvariable=self.prompt_var, width=50, style="Dark.TEntry")
        prompt_entry.grid(row=0, column=1, sticky="ew")

        # Duration + Driver inline
        option_frame = ttk.Frame(main_frame, style="Dark.TFrame")
        option_frame.grid(row=1, column=0, columnspan=2, pady=(10, 0))

        duration_label = ttk.Label(option_frame, text="⏱️ Durata:", style="Dark.TLabel")
        duration_label.pack(side=tk.LEFT, padx=(0, 5))
        duration_spin = ttk.Spinbox(option_frame, from_=5, to=30, textvariable=self.duration_var, width=5)
        duration_spin.pack(side=tk.LEFT, padx=(0, 15))

        driver_label = ttk.Label(option_frame, text="🔊 Driver:", style="Dark.TLabel")
        driver_label.pack(side=tk.LEFT, padx=(0, 5))
        driver_menu = ttk.Combobox(option_frame, textvariable=self.driver_var, values=["pulse", "alsa", "dsp"], width=10, state="readonly")
        driver_menu.pack(side=tk.LEFT)

        # Buttons
        button_frame = ttk.Frame(main_frame, style="Dark.TFrame")
        button_frame.grid(row=2, column=0, columnspan=2, pady=15)

        start_button = ttk.Button(button_frame, text="▶️ Avvia", command=self.start_loop, style="Dark.TButton")
        start_button.pack(side=tk.LEFT, padx=10)

        stop_button = ttk.Button(button_frame, text="⏹️ Ferma", command=self.stop_loop, style="Dark.TButton")
        stop_button.pack(side=tk.LEFT, padx=10)

        status_button = ttk.Button(button_frame, text="📊 Stato", command=self.show_status, style="Dark.TButton")
        status_button.pack(side=tk.LEFT, padx=10)

        save_button = ttk.Button(button_frame, text="💾 Salva", command=self.save_loop, style="Dark.TButton")
        save_button.pack(side=tk.LEFT, padx=10)

        main_frame.columnconfigure(1, weight=1)

    def start_loop(self):
        prompt = self.prompt_var.get().strip()
        if not prompt:
            messagebox.showerror("Errore", "Inserisci un prompt valido.")
            return

        self.app.duration = self.duration_var.get()
        self.app.audio_driver = self.driver_var.get()

        threading.Thread(target=lambda: self.app.start_loop(prompt), daemon=True).start()

    def stop_loop(self):
        self.app.stop_loop()

    def show_status(self):
        status_text = f"Status: {'ATTIVO' if self.app.is_playing else 'FERMO'}\n"
        status_text += f"Prompt: {self.app.PROMPT}\n"
        status_text += f"Durata: {self.app.duration}s\n"
        status_text += f"Driver: {self.app.audio_driver}\n"
        messagebox.showinfo("Stato", status_text)

    def save_loop(self):
        filename = "loop_salvato.wav"
        if self.app.save_current_loop(filename):
            messagebox.showinfo("Salvato", f"Loop salvato in {filename}")
        else:
            messagebox.showerror("Errore", "Impossibile salvare il loop.")

if __name__ == '__main__':
    root = tk.Tk()
    app = InfiniLoopUI(root)
    root.mainloop()
