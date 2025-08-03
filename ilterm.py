import subprocess
import os
import time
import soundfile as sf
import numpy as np
import librosa
import threading
from pydub.utils import mediainfo
import sys
import random
from datetime import datetime
import queue
from scipy.signal import correlate
from scipy.spatial.distance import cosine
import argparse
import signal
import tempfile
import shutil
import contextlib

class InfiniLoopTerminal:
    def __init__(self):

        self.base_dir = os.path.abspath(".")
        self.FILE1 = os.path.join(self.base_dir, "music1.wav")
        self.FILE2 = os.path.join(self.base_dir, "music2.wav")
        self.CURRENT = self.FILE1
        self.NEXT = self.FILE2

        self.CROSSFADE_MS = 1
        self.CROSSFADE_SEC = self.CROSSFADE_MS / 1000.0
        self.PROMPT = ""
        self.model = "medium"
        self.duration = 15
        self.audio_driver = "pulse"

        self.is_playing = False
        self.stop_event = threading.Event()
        self.loop_thread = None
        self.generation_thread = None
        self.is_generating = False
        self.generation_status = "Inattivo"

        self.file_lock = threading.Lock()
        self.swap_lock = threading.Lock()

        self.temp_dir = tempfile.mkdtemp(prefix="ilterm_")

        self.debug_mode = False

        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        os.system("cls" if os.name == "nt" else "clear")

        print("\n🎵 INFINI LOOP TERMINAL - by gat\n")
        print("✅ Inizializzazione completata!\n")
        self.stop_requested = False

    def __del__(self):

        try:
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
        except:
            pass

    def signal_handler(self, signum, frame):

        print("\n🛑 Interruzione rilevata, arresto in corso...")
        self.stop_loop()

        self.kill_all_ffplay_processes()
        self.kill_all_musicgpt_processes()

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        sys.exit(0)

    def log_message(self, message):

        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n\n[{timestamp}] {message}")

    def debug_file_state(self, operation, filepath):

        if not self.debug_mode:
            return

        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        try:
            size = os.path.getsize(filepath) if os.path.exists(filepath) else 0
            valid = self.validate_audio_file(filepath) if size > 0 else False
            print(f"[{timestamp}] {operation}: {os.path.basename(filepath)} "
                  f"size={size} valid={valid}")
        except Exception as e:
            print(f"[{timestamp}] {operation}: {os.path.basename(filepath)} ERROR: {e}")

    def validate_audio_file(self, filepath):

        if not os.path.isfile(filepath) or os.path.getsize(filepath) < 1024:
            return False

        try:
            with sf.SoundFile(filepath) as sf_test:
                if sf_test.frames == 0:
                    return False
        except:
            return False

        try:
            y, sr = librosa.load(filepath, sr=None, mono=True, duration=1.0)
            if len(y) == 0 or sr == 0:
                return False
            if not np.isfinite(y).all():
                return False
        except:
            return False

        return True

    @contextlib.contextmanager
    def safe_temp_file(self, suffix='.wav'):

        temp_path = None
        try:
            fd, temp_path = tempfile.mkstemp(suffix=suffix, dir=self.temp_dir)
            os.close(fd)
            yield temp_path
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass

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

    def calculate_waveform_continuity(self, y, start, end, sr):
        t = max(64, min(sr // 40, (end - start) // 20))
        a = y[max(0, end - t):end]
        b = y[start:start + t]
        if not len(a) or not len(b): return 0.0

        m = min(len(a), len(b))
        a, b = a[-m:], b[:m]
        corr = np.corrcoef(a, b)[0,1] if m > 1 else 0.0
        c = abs(corr) if not np.isnan(corr) else 0.0

        rms = np.sqrt(np.mean((a - b) ** 2))
        rms_sim = 1 - min(1.0, rms / max(np.sqrt(np.mean(a**2)), np.sqrt(np.mean(b**2)), 1e-8))

        if m > 1:
            d1, d2 = np.diff(a)[-1], np.diff(b)[0]
            d_cont = 1 - min(1.0, abs(d1 - d2) / max(abs(d1), abs(d2), 1e-8))
        else:
            d_cont = 1.0

        return c * 0.4 + rms_sim * 0.4 + d_cont * 0.2


    def calculate_beat_alignment(self, start_sample, end_sample, beats, sr):

        if len(beats) == 0:
            return 0.5

        d_start = np.min(np.abs(beats - start_sample))
        d_end = np.min(np.abs(beats - end_sample))

        if len(beats) > 1:
            avg_beat = np.mean(np.diff(beats)) * 0.5
            align_start = 1 - min(1.0, d_start / avg_beat)
            align_end = 1 - min(1.0, d_end / avg_beat)
        else:
            align_start = align_end = 0.5

        return (align_start + align_end) / 2


    def calculate_phase_continuity(self, S, start_frame, end_frame, window=3):

        if start_frame < window or end_frame >= S.shape[1] - window:
            return 0.5

        start_ph = np.angle(S[:, start_frame - window:start_frame + window])
        end_ph = np.angle(S[:, end_frame - window:end_frame + window])

        diff = np.abs(np.mean(start_ph, axis=1) - np.mean(end_ph, axis=1))
        diff = np.minimum(diff, 2 * np.pi - diff)

        return max(0.0, 1 - np.mean(diff) / np.pi)


    def find_perfect_loop_simple(self, y, sr):
        self.log_message("🧠 Algoritmo di loop detection semplice...")

        S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512), ref=np.max)
        min_f = int(5 * sr / 512)
        max_f = int(min(len(y) / sr * 0.8, 25) * sr / 512)
        best_score, best_start, best_end = -np.inf, 0, 0

        for i in range(0, S.shape[1] - min_f):
            for j in range(i + min_f, min(i + max_f, S.shape[1])):
                a, b = S[:, i], S[:, j]
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                if na > 1e-8 and nb > 1e-8:
                    s = np.dot(a, b) / (na * nb)
                    if s > best_score: best_score, best_start, best_end = s, i, j

        if best_score < 0.1:
            raise Exception(f"Nessuna similarità trovata (score: {best_score:.3f})")

        s_samp, e_samp = best_start * 512, best_end * 512
        if s_samp >= e_samp or e_samp > len(y):
            raise Exception(f"Bounds non validi: {s_samp} → {e_samp}")

        dur = (e_samp - s_samp) / sr
        self.log_message(f"✅ Loop semplice trovato! Score: {best_score:.3f}, Durata: {dur:.1f}s")

        return {
            'start_sample': s_samp,
            'end_sample': e_samp,
            'score': best_score,
            'measures': int(dur / 2),
            'metrics': {'Spettrale': best_score, 'Waveform': 0.5, 'Beat Align': 0.5, 'Fase': 0.5}
        }


    def find_perfect_loop(self, y, sr):

        try:

            return self.find_perfect_loop_advanced(y, sr)
        except Exception as e:
            self.log_message(f"⚠️ Algoritmo avanzato fallito: {e}")
            self.log_message("🔄 Uso algoritmo semplice di fallback...")

            return self.find_perfect_loop_simple(y, sr)

    def find_perfect_loop_advanced(self, y, sr):
        self.log_message("🧠 Analisi avanzata multi-metrica in corso...")

        if not len(y): raise Exception("Audio input vuoto")
        if sr <= 0: raise Exception(f"Sample rate non valido: {sr}")

        try:
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='samples')
            tempo = tempo.item() if isinstance(tempo, np.ndarray) else tempo
        except Exception as e:
            raise Exception(f"Errore beat tracking: {e}")

        if not (30 < tempo <= 300):
            self.log_message(f"⚠️ Tempo sospetto: {tempo} BPM, uso algoritmo semplice")
            raise Exception("Tempo non valido")

        try:
            S_mag = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
            if not S_mag.size: raise Exception("STFT vuoto")
        except Exception as e:
            raise Exception(f"Errore STFT: {e}")

        beat_len = 60 / tempo
        best_score, best_start, best_end, best_meas, best_metrics = -np.inf, 0, 0, 0, {}
        found = 0

        for meas in [2, 4, 8]:
            samples = int(meas * 4 * beat_len * sr)
            if not (3 * sr <= samples <= len(y) * 0.9):
                self.log_message(f"⚠️ Durata {samples/sr:.1f}s fuori range, skip {meas} misure")
                continue

            for start in range(int(len(y)*0.05), len(y) - samples - int(len(y)*0.05), 2048):
                end = start + samples
                if end > len(y) or end - start < sr * 0.5: continue

                sf, ef = start // 512, end // 512
                if sf < 3 or ef >= S_mag.shape[1]: continue

                try:
                    s1 = np.mean(S_mag[:, sf - 3:sf + 3], axis=1)
                    s2 = np.mean(S_mag[:, ef - 3:ef + 3], axis=1)
                    if np.any(np.isnan(s1)) or np.any(np.isnan(s2)): continue
                    spec = 1 - cosine(s1, s2)
                    spec = 0.0 if np.isnan(spec) else spec
                except: spec = 0.0

                def safe(fn, *args, default=0.5):
                    try:
                        val = fn(*args)
                        return 0.0 if np.isnan(val) else val
                    except: return default

                wave = safe(self.calculate_waveform_continuity, y, start, end, sr)
                beat = safe(self.calculate_beat_alignment, start, end, beats, sr)
                phase = safe(self.calculate_phase_continuity, S_mag, sf, ef)

                score = spec * 0.5 + wave * 0.25 + beat * 0.15 + phase * 0.1
                if not np.isfinite(score): continue

                if score > best_score:
                    best_score, best_start, best_end, best_meas = score, start, end, meas
                    best_metrics = {'Spettrale': spec, 'Waveform': wave, 'Beat Align': beat, 'Fase': phase}
                    found += 1

        if not found or best_score < 0.1:
            raise Exception(f"Nessun loop valido (score: {best_score:.3f})")
        if best_start >= best_end or best_end > len(y):
            raise Exception(f"Loop bounds invalidi: {best_start} → {best_end}")

        dur = (best_end - best_start) / sr
        if dur < 1.0:
            raise Exception(f"Loop troppo corto: {dur:.1f}s")

        self.log_message(f"✅ Loop avanzato trovato! {best_meas} misure, Score: {best_score:.3f}, Durata: {dur:.1f}s")

        return {
            'start_sample': best_start,
            'end_sample': best_end,
            'score': best_score,
            'measures': best_meas,
            'metrics': best_metrics
        }

    def process_loop_detection(self, input_file, output_file):
        try:
            self.debug_file_state("PRE_LOOP_DETECTION", input_file)

            if not self.validate_audio_file(input_file):
                raise Exception(f"File di input non valido: {input_file}")

            y, sr = librosa.load(input_file, sr=None, mono=True)
            if not len(y): raise Exception("Audio caricato è vuoto")
            if sr <= 0: raise Exception(f"Sample rate non valido: {sr}")
            if np.isnan(y).any() or np.isinf(y).any():
                raise Exception("Audio contiene valori NaN o infiniti")
            if len(y) / sr < 2.0:
                raise Exception(f"Audio troppo corto per loop detection: {len(y)/sr:.1f}s")

            loop_info = self.find_perfect_loop(y, sr)
            s, e = loop_info['start_sample'], loop_info['end_sample']
            if s < 0 or e > len(y) or s >= e:
                raise Exception(f"Loop bounds non validi: {s} -> {e} (max: {len(y)})")

            self.log_message("🎯 Ottimizzazione zero-crossing...")
            s, e = self.find_optimal_zero_crossing(y, s), self.find_optimal_zero_crossing(y, e)
            if s < 0 or e > len(y) or s >= e:
                raise Exception(f"Loop bounds corrotti dopo zero-crossing: {s} -> {e}")

            print("\n📊 Metriche loop:")
            for k, v in loop_info['metrics'].items():
                print(f"   {k}: {v:.3f}")

            y_loop = y[s:e]
            dur = len(y_loop) / sr
            if dur < 1.0:
                raise Exception(f"Loop troppo corto: {dur:.1f}s")
            if np.isnan(y_loop).any() or np.isinf(y_loop).any():
                raise Exception("Loop estratto contiene valori NaN o infiniti")

            f = min(256, len(y_loop) // 100)
            if f:
                y_loop[:f] *= np.linspace(0, 1, f)
                y_loop[-f:] *= np.linspace(1, 0, f)

            if os.path.exists(output_file):
                os.remove(output_file)

            try:
                sf.write(output_file, y_loop, sr)
            except Exception as err:
                raise Exception(f"Errore scrittura file audio: {err}")

            if os.path.getsize(output_file) < 1024:
                raise Exception(f"File di output troppo piccolo")

            self.debug_file_state("POST_LOOP_DETECTION", output_file)

            if not self.validate_audio_file(output_file):
                try:
                    test_y, test_sr = librosa.load(output_file, sr=None, mono=True)
                    raise Exception(f"File scritto ma validazione fallita (dur: {len(test_y)/test_sr:.1f}s, samples: {len(test_y)})")
                except Exception as err:
                    raise Exception(f"File scritto ma non leggibile: {err}")

            self.log_message(f"🧬 Ottenuto loop perfetto! (Forse...)\n              {loop_info['measures']} misure, {dur:.1f}s, Score: {loop_info['score']:.3f}")

        except Exception as e:
            if os.path.exists(output_file):
                try:
                    os.remove(output_file)
                    self.log_message(f"🗑️ Rimosso file corrotto: {os.path.basename(output_file)}")
                except: pass
            self.log_message(f"❌ Errore loop detection: {e}")
            raise


    def generate_audio_safe(self, outfile):
        try:
            self.is_generating = True
            prompt = self.PROMPT
            model = self.model
            duration = self.duration
            self.generation_status = f"Generando con prompt: '{prompt}'"
            self.log_message("🎼 Genero un nuovo sample...")

            with self.safe_temp_file() as raw_temp, self.safe_temp_file() as processed_temp:
                self.debug_file_state("PRE_GENERATION", raw_temp)

                result = subprocess.run([
                    "./musicgpt-x86_64-unknown-linux-gnu",
                    prompt,
                    "--model", model,
                    "--secs", str(duration),
                    "--output", raw_temp,
                    "--no-playback",
                    "--no-interactive",
                    "--ui-no-open"
                ], check=True, capture_output=True, text=True)

                self.debug_file_state("POST_GENERATION", raw_temp)

                if not self.validate_audio_file(raw_temp):
                    raise Exception("File audio generato con errori dalla AI.")

                os.system("cls" if os.name == "nt" else "clear")
                self.log_message(f"🎼 Sample generato ({duration}s)!")
                self.generation_status = "Analisi loop..."

                self.process_loop_detection(raw_temp, processed_temp)

                if not self.validate_audio_file(processed_temp):
                    raise Exception("File corrotto dopo il loop detection")

                self.debug_file_state("PRE_FINAL_MOVE", processed_temp)
                with self.file_lock:
                    shutil.move(processed_temp, outfile)
                self.debug_file_state("POST_FINAL_MOVE", outfile)

                self.generation_status = "Completato"

        except subprocess.CalledProcessError as e:
            self.log_message(f"❌ Errore generazione: {e}\n{e.stderr.strip()}")
            self.generation_status = "Errore"
            raise

        except Exception as e:
            self.log_message(f"❌ Errore inaspettato: {str(e)}")
            self.generation_status = "Errore"
            raise

        finally:
            self.is_generating = False


    def get_duration(self, filepath):

        try:
            with self.file_lock:
                if not os.path.exists(filepath):
                    return 0.0
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
            if not self.validate_audio_file(filepath):
                self.log_message(f"⚠️ File non valido per riproduzione: {filepath}")
                return

            env = os.environ.copy()
            env["SDL_AUDIODRIVER"] = self.audio_driver

            self.debug_file_state("START_PLAYBACK", filepath)

            process = subprocess.Popen(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", filepath],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            return_code = process.wait()
            self.debug_file_state("END_PLAYBACK", filepath)

            if return_code != 0:
                self.log_message(f"⚠️ ffplay terminato con codice {return_code}")

        except subprocess.TimeoutExpired:
            self.log_message("⚠️ ffplay timeout - terminazione forzata")
            self._kill_process_safely(process)

        except Exception as e:
            self.log_message(f"⚠️ ffplay crash rilevato: {str(e)}")
            self._kill_process_safely(process)

        finally:
            if process and process.poll() is None:
                self._kill_process_safely(process)

    def loop_current_crossfade_blocking(self, filepath, crossfade_sec, stop_event):

        try:
            duration = self.get_duration(filepath)
            if duration <= 0:
                self.log_message(f"⚠️ File audio non valido: {filepath}")
                return

            delay = max(0, duration - crossfade_sec)
            title = self.get_random_title()
            artist = self.get_random_artist()

            print(f"\n🎧 ORA IN RIPRODUZIONE:")
            print(f"   Titolo:  {title}")
            print(f"   Artista: {artist}")
            print(f"   Loop:    {duration:.1f} secondi")
            print(f"   Genere:  {self.PROMPT}\n")

            retry_count = 0
            max_retries = 3

            while self.is_playing and not stop_event.is_set():
                if not self.validate_audio_file(filepath):
                    self.log_message(f"⚠️ File corrotto rilevato durante loop: {filepath}")
                    break

                try:
                    thread = threading.Thread(
                        target=self.play_with_ffplay, args=(filepath,), daemon=True
                    )
                    thread.start()

                    if stop_event.wait(delay):
                        break

                    retry_count = 0

                except Exception as e:
                    retry_count += 1
                    self._log_retry_error(retry_count, max_retries, e)

                    if retry_count >= max_retries or stop_event.wait(1.0):
                        break

        except Exception as e:
            self.log_message(f"❌ Errore nel loop: {str(e)}")

    def _log_retry_error(self, count, max_count, exc):
        self.log_message(f"⚠️ Errore riproduzione (tentativo {count}/{max_count}): {str(exc)}")


    def safe_file_swap(self):

        with self.swap_lock:
            try:

                self.stop_event.set()


                if self.loop_thread and self.loop_thread.is_alive():
                    max_wait = min(self.get_duration(self.CURRENT) + 3.0, 10.0)
                    self.loop_thread.join(timeout=max_wait)

                    if self.loop_thread.is_alive():
                        self.log_message("⚠️ Timeout attesa: forzo terminazione ffplay")
                        self.kill_all_ffplay_processes()
                        self.loop_thread.join(timeout=2.0)


                if not self.validate_audio_file(self.NEXT):
                    raise Exception(f"⚠️ File NEXT non valido: {self.NEXT}")


                with self.file_lock:
                    self.CURRENT, self.NEXT = self.NEXT, self.CURRENT


                self.stop_event = threading.Event()
                return True

            except Exception as e:
                self.log_message(f"❌ Errore durante lo swap: {str(e)}")
                self.stop_event = threading.Event()
                return False


    def run_loop(self):


        loop_thread = threading.Thread(
            target=self.loop_current_crossfade_blocking,
            args=(self.CURRENT, self.CROSSFADE_SEC, self.stop_event),
            daemon=True
        )
        loop_thread.start()
        self.loop_thread = loop_thread

        consecutive_errors = 0
        max_consecutive_errors = 5

        while self.is_playing:
            try:

                self.generate_audio_safe(self.NEXT)

                if not self.is_playing:
                    break


                if not self.safe_file_swap():
                    self.log_message("❌ Swap fallito, rigenerazione...")
                    continue

                if not self.is_playing:
                    break


                loop_thread = threading.Thread(
                    target=self.loop_current_crossfade_blocking,
                    args=(self.CURRENT, self.CROSSFADE_SEC, self.stop_event),
                    daemon=True
                )
                loop_thread.start()
                self.loop_thread = loop_thread


                consecutive_errors = 0

            except Exception as e:
                consecutive_errors += 1
                self.log_message(f"❌ Errore nel ciclo ({consecutive_errors}/{max_consecutive_errors}): {str(e)}")

                if consecutive_errors >= max_consecutive_errors:
                    self.log_message("❌ Troppi errori consecutivi, arresto loop")
                    self.is_playing = False
                    break


                self.kill_all_ffplay_processes()


                if not self.stop_event.wait(2.0):
                    continue
                else:
                    break

    def main_loop(self):

        max_retries = 3
        retry_count = 0

        while retry_count < max_retries and self.is_playing:
            try:

                self.kill_all_ffplay_processes()
                self.kill_all_musicgpt_processes()


                for file_path, file_name in [(self.CURRENT, "primo"), (self.NEXT, "secondo")]:
                    if not self.validate_audio_file(file_path):
                        self.log_message(f"📁 Generazione sample iniziale ({file_name})...")
                        self.generate_audio_safe(file_path)
                        if not self.is_playing:
                            return


                        if not self.validate_audio_file(file_path):
                            raise Exception(f"File {file_path} non generato correttamente")

                        self.log_message(f"✅ File {os.path.basename(file_path)} generato e validato")


                self.run_loop()


                break

            except Exception as e:
                retry_count += 1
                self.log_message(f"❌ Errore nel loop principale (tentativo {retry_count}/{max_retries}): {str(e)}")

                if retry_count >= max_retries:
                    self.log_message("❌ Troppi errori, arresto applicazione")
                    self.is_playing = False
                    return


                self.log_message(f"🔄 Reinizializzazione in corso...")
                self.kill_all_ffplay_processes()
                self.kill_all_musicgpt_processes()


                time.sleep(2)


                try:
                    for filepath in [self.CURRENT, self.NEXT]:
                        if os.path.exists(filepath):
                            os.remove(filepath)
                            self.log_message(f"🗑️ Rimosso {os.path.basename(filepath)}")
                except Exception as remove_error:
                    self.log_message(f"⚠️ Errore rimozione file: {remove_error}")

    def start_loop(self, prompt):
        self.stop_requested = False

        self.PROMPT = prompt.strip()
        if not self.PROMPT:
            print("❌ Errore: Inserisci un prompt!")
            return False

        self.is_playing = True


        self.loop_thread = threading.Thread(target=self.main_loop, daemon=True)
        self.loop_thread.start()

        return True

    def stop_loop(self):

        self.is_playing = False
        if hasattr(self, 'stop_event'):
            self.stop_event.set()
        self.log_message("⏹️ Loop fermato")


        self.kill_all_ffplay_processes()
        self.kill_all_musicgpt_processes()

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
        except Exception as e:

            pass

    def kill_all_musicgpt_processes(self):

        try:
            result = subprocess.run(["pgrep", "-f", "musicgpt-x86_64-unknown-linux-gnu"],
                                  capture_output=True, text=True)
            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    try:
                        subprocess.run(["kill", "-9", pid], check=False, timeout=2)
                    except:
                        pass
                self.log_message(f"🛑 MusicGPT terminato")
        except Exception as e:

            pass

    def save_current_loop(self, filename):

        with self.file_lock:

            current_file = self.CURRENT

            if not self.validate_audio_file(current_file):
                print("❌ Nessun loop valido da salvare!")
                return False

            try:

                shutil.copy2(current_file, filename)
                self.log_message(f"💾 Loop salvato: {filename} (da {os.path.basename(current_file)})")
                return True
            except Exception as e:
                self.log_message(f"❌ Errore salvataggio: {str(e)}")
                return False

    def print_status(self):

        status = "🟢 IN RIPRODUZIONE" if self.is_playing else "🔴 FERMO"
        generation = "🎼 SÌ" if self.is_generating else "💤 NO"

        print(f"\n📊 STATO INFINI LOOP:")
        print(f"   Status:       {status}")
        print(f"   Prompt:       '{self.PROMPT}'")
        print(f"   Durata gen.:  {self.duration}s")
        print(f"   Driver audio: {self.audio_driver}")
        print(f"   Generazione:  {generation}")
        if self.is_generating:
            print(f"   Stato gen.:   {self.generation_status}")


        print(f"\n📂 STATO FILE:")
        with self.file_lock:
            print(f"   File corrente: {os.path.basename(self.CURRENT)}")
            if self.validate_audio_file(self.CURRENT):
                size = os.path.getsize(self.CURRENT)
                print(f"                  ✅ Valido ({size} bytes)")
            else:
                print(f"                  ❌ Non valido o mancante")

            print(f"   File prossimo: {os.path.basename(self.NEXT)}")
            if self.validate_audio_file(self.NEXT):
                size = os.path.getsize(self.NEXT)
                print(f"                  ✅ Valido ({size} bytes)")
            else:
                print(f"                  ❌ Non valido o mancante")
        print()

def interactive_mode(app):

    try:
        print("   MODALITÀ INTERATTIVA")
        print("    start <prompt>  - Avvia loop con prompt")
        print("    stop            - Ferma loop")
        print("    help            - Mostra tutti i comandi")
        print("    quit            - Esci")
    except Exception as e:
        print(f"❌ Errore nell'inizializzazione modalità interattiva: {e}")
        import traceback
        traceback.print_exc()
        return

    while True:
        try:
            cmd = input("\n🎛️ > ").strip().split()
            if not cmd:
                continue

            if cmd[0] == "start":
                if len(cmd) < 2:
                    print("❌ Uso: start <prompt>")
                    continue
                prompt = " ".join(cmd[1:])
                app.start_loop(prompt)

            elif cmd[0] == "stop":
                app.stop_loop()

            elif cmd[0] == "status":
                app.print_status()

            elif cmd[0] == "save":
                if len(cmd) < 2:
                    print("❌ Uso: save <filename>")
                    continue
                filename = cmd[1]

                if not filename.endswith('.wav'):
                    filename += '.wav'
                if app.save_current_loop(filename):
                    print(f"✅ Loop salvato come: {filename}")
                else:
                    print("❌ Impossibile salvare il loop")

            elif cmd[0] == "debug":
                if len(cmd) > 1 and cmd[1] in ["on", "off"]:
                    app.debug_mode = (cmd[1] == "on")
                    print(f"🐛 Debug mode: {'ON' if app.debug_mode else 'OFF'}")
                else:
                    print(f"🐛 Debug mode attuale: {'ON' if app.debug_mode else 'OFF'}")
                    print("💡 Usa 'debug on' o 'debug off' per cambiare")

            elif cmd[0] == "validate":
                if len(cmd) < 2:
                    print("❌ Uso: validate <current|next|both>")
                    continue

                target = cmd[1].lower()
                if target in ["current", "both"]:
                    valid = app.validate_audio_file(app.CURRENT)
                    print(f"📁 File corrente: {'✅ VALIDO' if valid else '❌ NON VALIDO'}")

                if target in ["next", "both"]:
                    valid = app.validate_audio_file(app.NEXT)
                    print(f"📁 File prossimo: {'✅ VALIDO' if valid else '❌ NON VALIDO'}")

            elif cmd[0] == "set":
                if len(cmd) < 2:
                    print("❌ Opzioni disponibili:")
                    print("    duration  - Durata sample generati")
                    print("    driver    - Driver audio di sistema")
                    print("💡 Usa 'set <opzione>' per cambiare")
                    continue

                option = cmd[1]
                if option == "duration":
                    print("\n⏱️ DURATA GENERAZIONE:")
                    print("    Range: 5-30 secondi")
                    print("    Consiglio: 10-15s per loop brevi, 20-30s per brani più lunghi")
                    try:
                        duration = int(input("Durata in secondi [5-30]: "))
                        if 5 <= duration <= 30:
                            app.duration = duration
                            print(f"✅ Durata: {app.duration}s")
                        else:
                            print("❌ Durata deve essere tra 5 e 30 secondi")
                    except ValueError:
                        print("❌ Valore non numerico")

                elif option == "driver":
                    print("\n🔊 DRIVER AUDIO DISPONIBILI:")
                    print("    pulse - PulseAudio (Linux standard, consigliato)")
                    print("    alsa  - ALSA (Linux low-level)")
                    print("    dsp   - OSS (sistemi Unix/BSD)")
                    choice = input("Scegli driver [pulse/alsa/dsp]: ").strip().lower()
                    if choice in ['pulse', 'pulseaudio']:
                        app.audio_driver = 'pulse'
                        print("✅ Driver: PulseAudio")
                    elif choice in ['alsa']:
                        app.audio_driver = 'alsa'
                        print("✅ Driver: ALSA")
                    elif choice in ['dsp', 'oss']:
                        app.audio_driver = 'dsp'
                        print("✅ Driver: OSS")
                    else:
                        print("❌ Driver non valido")
                else:
                    print(f"❌ Opzione '{option}' non riconosciuta")
                    print("💡 Opzioni: duration, driver")

            elif cmd[0] == "help":
                print("\n🆘 COMANDI DISPONIBILI:")
                print("   start '<prompt>'    - Avvia loop infinito con prompt")
                print("   stop                - Ferma riproduzione corrente")
                print("   status              - Mostra stato dettagliato sistema")
                print("   save <file.wav>     - Salva loop corrente su file")
                print("   validate <target>   - Valida file audio (current/next/both - per debug")
                print("   debug <on|off>      - Attiva/disattiva i messaggi di debug")
                print("   set <opzione>       - Cambia impostazioni (vedi sotto)")
                print("   help                - Mostra questo aiuto")
                print("   quit/exit/q         - Esci dal programma")
                print("\n⚙️ OPZIONI CONFIGURABILI:")
                print("   set duration        - Cambia durata generazione (5-30s)")
                print("   set driver          - Cambia driver audio (pulse/alsa/dsp)")
                print("\n💡 ESEMPI:")
                print("   start 'ambient chill loop'")
                print("   start 'jazz piano solo'")
                print("   save my_loop.wav")
                print("   validate both")
                print("   debug on")

            elif cmd[0] in ["quit", "exit", "q"]:
                app.stop_loop()
                print("👋 Arrivederci!")
                break

            else:
                print(f"❌ Comando non riconosciuto: {cmd[0]}")
                print("💡 Usa 'help' per vedere i comandi disponibili")

        except KeyboardInterrupt:
            app.stop_loop()
            print("\n👋 Arrivederci!")
            break
        except EOFError:
            app.stop_loop()
            print("\n👋 Arrivederci!")
            break
        except Exception as e:
            print(f"❌ Errore imprevisto: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

def main():
    parser = argparse.ArgumentParser(
        description="INFINI LOOP TERMINAL - Infinite AI Music Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi d'uso:
  %(prog)s --prompt "ambient loop"
  %(prog)s --duration 20 --prompt "jazz loop"
  %(prog)s --interactive
  %(prog)s --generate-only "rock loop" output.wav

Impostazioni fisse:
  Algoritmo: Avanzato con fallback (spettrale + waveform + beat + fase)
  Modello AI: Medium (bilanciato)
  Crossfade: 1ms (minimo)

CORREZIONI APPLICATE:
  ✅ Race condition nel file swapping eliminata
  ✅ Validazione audio completa implementata
  ✅ File temporanei sicuri con context manager
  ✅ Swap sincronizzato con fine loop naturale
  ✅ Debug mode per troubleshooting
        """
    )


    parser.add_argument("--prompt", "-p", type=str,
                       help="Prompt per la generazione")

    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Modalità interattiva")

    parser.add_argument("--generate-only", "-g", nargs=2, metavar=("PROMPT", "OUTPUT"),
                       help="Genera solo un loop e salva (prompt, file_output)")


    parser.add_argument("--duration", "-d", type=int, default=15,
                       help="Durata generazione in secondi (5-30)")

    parser.add_argument("--driver", choices=["pulse", "alsa", "dsp"],
                       default="pulse", help="Driver audio")


    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Output dettagliato")

    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Output minimale")

    parser.add_argument("--no-debug", action="store_true",
                       help="Disabilita debug mode")

    args = parser.parse_args()


    if args.duration < 5 or args.duration > 30:
        print("❌ Errore: Durata deve essere tra 5 e 30 secondi")
        sys.exit(1)


    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    os.environ["SDL_AUDIODRIVER"] = args.driver
    os.environ["ALSA_CARD"] = "default"


    app = InfiniLoopTerminal()


    app.duration = args.duration
    app.audio_driver = args.driver
    app.debug_mode = False if args.no_debug else False

    print(f"🧠 Algoritmo:        Avanzato con fallback")
    print(f"🤖 Modello AI:       Medium")
    print(f"⏱️ Durata sample:    {app.duration}s")
    print(f"🔊 Driver audio:     {app.audio_driver}")
    print(f"🐛 Debug mode:       {'ON' if app.debug_mode else 'OFF'}")

    try:

        if args.generate_only:
            prompt, output_file = args.generate_only
            app.PROMPT = prompt
            print(f"\n🎼 Generazione singola: '{prompt}'")

            app.generate_audio_safe(output_file)
            print(f"✅ Loop salvato: {output_file}")
            return


        elif args.interactive:
            interactive_mode(app)
            return


        elif args.prompt:
            if app.start_loop(args.prompt):
                print("🎵 Loop avviato! Premi Ctrl+C per fermare")
                try:

                    while app.is_playing:
                        time.sleep(1)
                except KeyboardInterrupt:
                    app.stop_loop()
                    print("\n👋 Arrivederci!")
            return


        else:
            print("\n💡 Nessun prompt specificato:")
            interactive_mode(app)

    except KeyboardInterrupt:
        app.stop_loop()
        print("\n👋 Arrivederci!")
    except Exception as e:
        print(f"\n❌ Errore: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
