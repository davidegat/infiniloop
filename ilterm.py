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
        self.generation_status = "Idle"

        self.file_lock = threading.Lock()
        self.swap_lock = threading.Lock()

        self.temp_dir = tempfile.mkdtemp(prefix="ilterm_")

        self.debug_mode = False

        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        os.system("cls" if os.name == "nt" else "clear")

        print("\n🎵 INFINI LOOP TERMINAL - by gat\n")
        print("✅ Initialization completed!\n")
        self.stop_requested = False

    def __del__(self):

        try:
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
        except:
            pass

    def signal_handler(self, signum, frame):

        print("\n🛑 Interrupt detected, stopping...")
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
        self.log_message("🧠 Simple loop detection algorithm...")

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
            raise Exception(f"No similarity found (score: {best_score:.3f})")

        s_samp, e_samp = best_start * 512, best_end * 512
        if s_samp >= e_samp or e_samp > len(y):
            raise Exception(f"Invalid bounds: {s_samp} → {e_samp}")

        dur = (e_samp - s_samp) / sr
        self.log_message(f"✅ Simple loop found! Score: {best_score:.3f}, Duration: {dur:.1f}s")

        return {
            'start_sample': s_samp,
            'end_sample': e_samp,
            'score': best_score,
            'measures': int(dur / 2),
            'metrics': {'Spectral': best_score, 'Waveform': 0.5, 'Beat Align': 0.5, 'Phase': 0.5}
        }


    def find_perfect_loop(self, y, sr):

        try:

            return self.find_perfect_loop_advanced(y, sr)
        except Exception as e:
            self.log_message(f"⚠️ Advanced algorithm failed: {e}")
            self.log_message("🔄 Using simple fallback algorithm...")

            return self.find_perfect_loop_simple(y, sr)

    def find_perfect_loop_advanced(self, y, sr):
        self.log_message("🧠 Advanced multi-metric analysis in progress...")

        if not len(y): raise Exception("Empty audio input")
        if sr <= 0: raise Exception(f"Invalid sample rate: {sr}")

        try:
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='samples')
            tempo = tempo.item() if isinstance(tempo, np.ndarray) else tempo
        except Exception as e:
            raise Exception(f"Beat tracking error: {e}")

        if not (30 < tempo <= 300):
            self.log_message(f"⚠️ Suspicious tempo: {tempo} BPM, using simple algorithm")
            raise Exception("Invalid tempo")

        try:
            S_mag = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
            if not S_mag.size: raise Exception("Empty STFT")
        except Exception as e:
            raise Exception(f"STFT error: {e}")

        beat_len = 60 / tempo
        best_score, best_start, best_end, best_meas, best_metrics = -np.inf, 0, 0, 0, {}
        found = 0

        for meas in [2, 4, 8]:
            samples = int(meas * 4 * beat_len * sr)
            if not (3 * sr <= samples <= len(y) * 0.9):
                self.log_message(f"⚠️ Duration {samples/sr:.1f}s out of range, skipping {meas} measures")
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
                    best_metrics = {'Spectral': spec, 'Waveform': wave, 'Beat Align': beat, 'Phase': phase}
                    found += 1

        if not found or best_score < 0.1:
            raise Exception(f"No valid loop found (score: {best_score:.3f})")
        if best_start >= best_end or best_end > len(y):
            raise Exception(f"Invalid loop bounds: {best_start} → {best_end}")

        dur = (best_end - best_start) / sr
        if dur < 1.0:
            raise Exception(f"Loop too short: {dur:.1f}s")

        self.log_message(f"✅ Advanced loop found! {best_meas} measures, Score: {best_score:.3f}, Duration: {dur:.1f}s")

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
                raise Exception(f"Invalid input file: {input_file}")

            y, sr = librosa.load(input_file, sr=None, mono=True)
            if not len(y): raise Exception("Loaded audio is empty")
            if sr <= 0: raise Exception(f"Invalid sample rate: {sr}")
            if np.isnan(y).any() or np.isinf(y).any():
                raise Exception("Audio contains NaN or infinite values")
            if len(y) / sr < 2.0:
                raise Exception(f"Audio too short for loop detection: {len(y)/sr:.1f}s")

            loop_info = self.find_perfect_loop(y, sr)
            s, e = loop_info['start_sample'], loop_info['end_sample']
            if s < 0 or e > len(y) or s >= e:
                raise Exception(f"Invalid loop bounds: {s} -> {e} (max: {len(y)})")

            self.log_message("🎯 Zero-crossing optimization...")
            s, e = self.find_optimal_zero_crossing(y, s), self.find_optimal_zero_crossing(y, e)
            if s < 0 or e > len(y) or s >= e:
                raise Exception(f"Loop bounds corrupted after zero-crossing: {s} -> {e}")

            print("\n📊 Loop metrics:")
            for k, v in loop_info['metrics'].items():
                print(f"   {k}: {v:.3f}")

            y_loop = y[s:e]
            dur = len(y_loop) / sr
            if dur < 1.0:
                raise Exception(f"Loop too short: {dur:.1f}s")
            if np.isnan(y_loop).any() or np.isinf(y_loop).any():
                raise Exception("Extracted loop contains NaN or infinite values")

            f = min(256, len(y_loop) // 100)
            if f:
                y_loop[:f] *= np.linspace(0, 1, f)
                y_loop[-f:] *= np.linspace(1, 0, f)

            if os.path.exists(output_file):
                os.remove(output_file)

            try:
                sf.write(output_file, y_loop, sr)
            except Exception as err:
                raise Exception(f"Error writing audio file: {err}")

            if os.path.getsize(output_file) < 1024:
                raise Exception(f"Output file too small")

            self.debug_file_state("POST_LOOP_DETECTION", output_file)

            if not self.validate_audio_file(output_file):
                try:
                    test_y, test_sr = librosa.load(output_file, sr=None, mono=True)
                    raise Exception(f"File written but validation failed (dur: {len(test_y)/test_sr:.1f}s, samples: {len(test_y)})")
                except Exception as err:
                    raise Exception(f"File written but not readable: {err}")

            self.log_message(f"🧬 Perfect loop obtained! (Maybe...)\n              {loop_info['measures']} measures, {dur:.1f}s, Score: {loop_info['score']:.3f}")

        except Exception as e:
            if os.path.exists(output_file):
                try:
                    os.remove(output_file)
                    self.log_message(f"🗑️ Removed corrupted file: {os.path.basename(output_file)}")
                except: pass
            self.log_message(f"❌ Loop detection error: {e}")
            raise


    def generate_audio_safe(self, outfile):
        try:
            self.is_generating = True
            prompt = self.PROMPT
            model = self.model
            duration = self.duration
            self.generation_status = f"Generating with prompt: '{prompt}'"
            self.log_message("🎼 Generating new sample...")

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
                    raise Exception("Audio file generated with errors from AI.")

                os.system("cls" if os.name == "nt" else "clear")
                self.log_message(f"🎼 Sample generated ({duration}s)!")
                self.generation_status = "Loop analysis..."

                self.process_loop_detection(raw_temp, processed_temp)

                if not self.validate_audio_file(processed_temp):
                    raise Exception("File corrupted after loop detection")

                self.debug_file_state("PRE_FINAL_MOVE", processed_temp)
                with self.file_lock:
                    shutil.move(processed_temp, outfile)
                self.debug_file_state("POST_FINAL_MOVE", outfile)

                self.generation_status = "Completed"

        except subprocess.CalledProcessError as e:
            self.log_message(f"❌ Generation error: {e}\n{e.stderr.strip()}")
            self.generation_status = "Error"
            raise

        except Exception as e:
            self.log_message(f"❌ Unexpected error: {str(e)}")
            self.generation_status = "Error"
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
        return "UNTITLED"

    def get_random_artist(self):

        try:
            with open("artisti.txt", "r") as f:
                artists = [line.strip() for line in f if line.strip()]
            return random.choice(artists).upper() if artists else "UNKNOWN ARTIST"
        except Exception:
            return "UNKNOWN ARTIST"

    def play_with_ffplay(self, filepath):

        try:
            if not self.validate_audio_file(filepath):
                self.log_message(f"⚠️ Invalid file for playback: {filepath}")
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
                self.log_message(f"⚠️ ffplay terminated with code {return_code}")

        except subprocess.TimeoutExpired:
            self.log_message("⚠️ ffplay timeout - forced termination")
            self._kill_process_safely(process)

        except Exception as e:
            self.log_message(f"⚠️ ffplay crash detected: {str(e)}")
            self._kill_process_safely(process)

        finally:
            if process and process.poll() is None:
                self._kill_process_safely(process)

    def loop_current_crossfade_blocking(self, filepath, crossfade_sec, stop_event):

        try:
            duration = self.get_duration(filepath)
            if duration <= 0:
                self.log_message(f"⚠️ Invalid audio file: {filepath}")
                return

            delay = max(0, duration - crossfade_sec)
            title = self.get_random_title()
            artist = self.get_random_artist()

            print(f"\n🎧 NOW PLAYING:")
            print(f"   Title:   {title}")
            print(f"   Artist:  {artist}")
            print(f"   Loop:    {duration:.1f} seconds")
            print(f"   Genre:   {self.PROMPT}\n")

            retry_count = 0
            max_retries = 3

            while self.is_playing and not stop_event.is_set():
                if not self.validate_audio_file(filepath):
                    self.log_message(f"⚠️ Corrupted file detected during loop: {filepath}")
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
            self.log_message(f"❌ Error in loop: {str(e)}")

    def _log_retry_error(self, count, max_count, exc):
        self.log_message(f"⚠️ Playback error (attempt {count}/{max_count}): {str(exc)}")


    def safe_file_swap(self):

        with self.swap_lock:
            try:

                self.stop_event.set()


                if self.loop_thread and self.loop_thread.is_alive():
                    max_wait = min(self.get_duration(self.CURRENT) + 3.0, 10.0)
                    self.loop_thread.join(timeout=max_wait)

                    if self.loop_thread.is_alive():
                        self.log_message("⚠️ Timeout waiting: forcing ffplay termination")
                        self.kill_all_ffplay_processes()
                        self.loop_thread.join(timeout=2.0)


                if not self.validate_audio_file(self.NEXT):
                    raise Exception(f"⚠️ Invalid NEXT file: {self.NEXT}")


                with self.file_lock:
                    self.CURRENT, self.NEXT = self.NEXT, self.CURRENT


                self.stop_event = threading.Event()
                return True

            except Exception as e:
                self.log_message(f"❌ Error during swap: {str(e)}")
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
                    self.log_message("❌ Swap failed, regenerating...")
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
                self.log_message(f"❌ Error in cycle ({consecutive_errors}/{max_consecutive_errors}): {str(e)}")

                if consecutive_errors >= max_consecutive_errors:
                    self.log_message("❌ Too many consecutive errors, stopping loop")
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


                for file_path, file_name in [(self.CURRENT, "first"), (self.NEXT, "second")]:
                    if not self.validate_audio_file(file_path):
                        self.log_message(f"📁 Generating initial sample ({file_name})...")
                        self.generate_audio_safe(file_path)
                        if not self.is_playing:
                            return


                        if not self.validate_audio_file(file_path):
                            raise Exception(f"File {file_path} not generated correctly")

                        self.log_message(f"✅ File {os.path.basename(file_path)} generated and validated")


                self.run_loop()


                break

            except Exception as e:
                retry_count += 1
                self.log_message(f"❌ Error in main loop (attempt {retry_count}/{max_retries}): {str(e)}")

                if retry_count >= max_retries:
                    self.log_message("❌ Too many errors, stopping application")
                    self.is_playing = False
                    return


                self.log_message(f"🔄 Reinitializing...")
                self.kill_all_ffplay_processes()
                self.kill_all_musicgpt_processes()


                time.sleep(2)


                try:
                    for filepath in [self.CURRENT, self.NEXT]:
                        if os.path.exists(filepath):
                            os.remove(filepath)
                            self.log_message(f"🗑️ Removed {os.path.basename(filepath)}")
                except Exception as remove_error:
                    self.log_message(f"⚠️ File removal error: {remove_error}")

    def start_loop(self, prompt):
        self.stop_requested = False

        self.PROMPT = prompt.strip()
        if not self.PROMPT:
            print("❌ Error: Please enter a prompt!")
            return False

        self.is_playing = True


        self.loop_thread = threading.Thread(target=self.main_loop, daemon=True)
        self.loop_thread.start()

        return True

    def stop_loop(self):

        self.is_playing = False
        if hasattr(self, 'stop_event'):
            self.stop_event.set()
        self.log_message("⏹️ Loop stopped")


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
                self.log_message(f"🛑 MusicGPT terminated")
        except Exception as e:

            pass

    def save_current_loop(self, filename):

        with self.file_lock:

            current_file = self.CURRENT

            if not self.validate_audio_file(current_file):
                print("❌ No valid loop to save!")
                return False

            try:

                shutil.copy2(current_file, filename)
                self.log_message(f"💾 Loop saved: {filename} (from {os.path.basename(current_file)})")
                return True
            except Exception as e:
                self.log_message(f"❌ Save error: {str(e)}")
                return False

    def print_status(self):

        status = "🟢 PLAYING" if self.is_playing else "🔴 STOPPED"
        generation = "🎼 YES" if self.is_generating else "💤 NO"

        print(f"\n📊 INFINI LOOP STATUS:")
        print(f"   Status:       {status}")
        print(f"   Prompt:       '{self.PROMPT}'")
        print(f"   Gen. duration: {self.duration}s")
        print(f"   Audio driver: {self.audio_driver}")
        print(f"   Generating:   {generation}")
        if self.is_generating:
            print(f"   Gen. status:  {self.generation_status}")


        print(f"\n📂 FILE STATUS:")
        with self.file_lock:
            print(f"   Current file: {os.path.basename(self.CURRENT)}")
            if self.validate_audio_file(self.CURRENT):
                size = os.path.getsize(self.CURRENT)
                print(f"                 ✅ Valid ({size} bytes)")
            else:
                print(f"                 ❌ Invalid or missing")

            print(f"   Next file:    {os.path.basename(self.NEXT)}")
            if self.validate_audio_file(self.NEXT):
                size = os.path.getsize(self.NEXT)
                print(f"                 ✅ Valid ({size} bytes)")
            else:
                print(f"                 ❌ Invalid or missing")
        print()

def interactive_mode(app):

    try:
        print("   INTERACTIVE MODE")
        print("    start <prompt>  - Start loop with prompt")
        print("    stop            - Stop loop")
        print("    help            - Show all commands")
        print("    quit            - Exit")
    except Exception as e:
        print(f"❌ Error initializing interactive mode: {e}")
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
                    print("❌ Usage: start <prompt>")
                    continue
                prompt = " ".join(cmd[1:])
                app.start_loop(prompt)

            elif cmd[0] == "stop":
                app.stop_loop()

            elif cmd[0] == "status":
                app.print_status()

            elif cmd[0] == "save":
                if len(cmd) < 2:
                    print("❌ Usage: save <filename>")
                    continue
                filename = cmd[1]

                if not filename.endswith('.wav'):
                    filename += '.wav'
                if app.save_current_loop(filename):
                    print(f"✅ Loop saved as: {filename}")
                else:
                    print("❌ Unable to save loop")

            elif cmd[0] == "debug":
                if len(cmd) > 1 and cmd[1] in ["on", "off"]:
                    app.debug_mode = (cmd[1] == "on")
                    print(f"🐛 Debug mode: {'ON' if app.debug_mode else 'OFF'}")
                else:
                    print(f"🐛 Current debug mode: {'ON' if app.debug_mode else 'OFF'}")
                    print("💡 Use 'debug on' or 'debug off' to change")

            elif cmd[0] == "validate":
                if len(cmd) < 2:
                    print("❌ Usage: validate <current|next|both>")
                    continue

                target = cmd[1].lower()
                if target in ["current", "both"]:
                    valid = app.validate_audio_file(app.CURRENT)
                    print(f"📁 Current file: {'✅ VALID' if valid else '❌ INVALID'}")

                if target in ["next", "both"]:
                    valid = app.validate_audio_file(app.NEXT)
                    print(f"📁 Next file: {'✅ VALID' if valid else '❌ INVALID'}")

            elif cmd[0] == "set":
                if len(cmd) < 2:
                    print("❌ Available options:")
                    print("    duration  - Generated sample duration")
                    print("    driver    - System audio driver")
                    print("💡 Use 'set <option>' to change")
                    continue

                option = cmd[1]
                if option == "duration":
                    print("\n⏱️ GENERATION DURATION:")
                    print("    Range: 5-30 seconds")
                    print("    Tip: 10-15s for short loops, 20-30s for longer tracks")
                    try:
                        duration = int(input("Duration in seconds [5-30]: "))
                        if 5 <= duration <= 30:
                            app.duration = duration
                            print(f"✅ Duration: {app.duration}s")
                        else:
                            print("❌ Duration must be between 5 and 30 seconds")
                    except ValueError:
                        print("❌ Non-numeric value")

                elif option == "driver":
                    print("\n🔊 AVAILABLE AUDIO DRIVERS:")
                    print("    pulse - PulseAudio (Linux standard, recommended)")
                    print("    alsa  - ALSA (Linux low-level)")
                    print("    dsp   - OSS (Unix/BSD systems)")
                    choice = input("Choose driver [pulse/alsa/dsp]: ").strip().lower()
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
                        print("❌ Invalid driver")
                else:
                    print(f"❌ Option '{option}' not recognized")
                    print("💡 Options: duration, driver")

            elif cmd[0] == "help":
                print("\n🆘 AVAILABLE COMMANDS:")
                print("   start '<prompt>'    - Start infinite loop with prompt")
                print("   stop                - Stop current playback")
                print("   status              - Show detailed system status")
                print("   save <file.wav>     - Save current loop to file")
                print("   validate <target>   - Validate audio files (current/next/both - for debug)")
                print("   debug <on|off>      - Enable/disable debug messages")
                print("   set <option>        - Change settings (see below)")
                print("   help                - Show this help")
                print("   quit/exit/q         - Exit program")
                print("\n⚙️ CONFIGURABLE OPTIONS:")
                print("   set duration        - Change generation duration (5-30s)")
                print("   set driver          - Change audio driver (pulse/alsa/dsp)")
                print("\n💡 EXAMPLES:")
                print("   start 'ambient chill loop'")
                print("   start 'jazz piano solo'")
                print("   save my_loop.wav")
                print("   validate both")
                print("   debug on")

            elif cmd[0] in ["quit", "exit", "q"]:
                app.stop_loop()
                print("👋 Goodbye!")
                break

            else:
                print(f"❌ Unknown command: {cmd[0]}")
                print("💡 Use 'help' to see available commands")

        except KeyboardInterrupt:
            app.stop_loop()
            print("\n👋 Goodbye!")
            break
        except EOFError:
            app.stop_loop()
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

def main():
    parser = argparse.ArgumentParser(
        description="INFINI LOOP TERMINAL - Infinite AI Music Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  %(prog)s --prompt "ambient loop"
  %(prog)s --duration 20 --prompt "jazz loop"
  %(prog)s --interactive
  %(prog)s --generate-only "rock loop" output.wav

Fixed settings:
  Algorithm: Advanced with fallback (spectral + waveform + beat + phase)
  AI Model: Medium (balanced)
  Crossfade: 1ms (minimum)

APPLIED FIXES:
  ✅ Race condition in file swapping eliminated
  ✅ Complete audio validation implemented
  ✅ Safe temporary files with context manager
  ✅ Swap synchronized with natural loop end
  ✅ Debug mode for troubleshooting
        """
    )


    parser.add_argument("--prompt", "-p", type=str,
                       help="Prompt for generation")

    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Interactive mode")

    parser.add_argument("--generate-only", "-g", nargs=2, metavar=("PROMPT", "OUTPUT"),
                       help="Generate only one loop and save (prompt, output_file)")


    parser.add_argument("--duration", "-d", type=int, default=15,
                       help="Generation duration in seconds (5-30)")

    parser.add_argument("--driver", choices=["pulse", "alsa", "dsp"],
                       default="pulse", help="Audio driver")


    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Detailed output")

    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Minimal output")

    parser.add_argument("--no-debug", action="store_true",
                       help="Disable debug mode")

    args = parser.parse_args()


    if args.duration < 5 or args.duration > 30:
        print("❌ Error: Duration must be between 5 and 30 seconds")
        sys.exit(1)


    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    os.environ["SDL_AUDIODRIVER"] = args.driver
    os.environ["ALSA_CARD"] = "default"


    app = InfiniLoopTerminal()


    app.duration = args.duration
    app.audio_driver = args.driver
    app.debug_mode = False if args.no_debug else False

    print(f"🧠 Algorithm:        Advanced with fallback")
    print(f"🤖 AI Model:         Medium")
    print(f"⏱️ Sample duration:  {app.duration}s")
    print(f"🔊 Audio driver:     {app.audio_driver}")
    print(f"🐛 Debug mode:       {'ON' if app.debug_mode else 'OFF'}")

    try:

        if args.generate_only:
            prompt, output_file = args.generate_only
            app.PROMPT = prompt
            print(f"\n🎼 Single generation: '{prompt}'")

            app.generate_audio_safe(output_file)
            print(f"✅ Loop saved: {output_file}")
            return


        elif args.interactive:
            interactive_mode(app)
            return


        elif args.prompt:
            if app.start_loop(args.prompt):
                print("🎵 Loop started! Press Ctrl+C to stop")
                try:

                    while app.is_playing:
                        time.sleep(1)
                except KeyboardInterrupt:
                    app.stop_loop()
                    print("\n👋 Goodbye!")
            return


        else:
            print("\n💡 No prompt specified:")
            interactive_mode(app)

    except KeyboardInterrupt:
        app.stop_loop()
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
