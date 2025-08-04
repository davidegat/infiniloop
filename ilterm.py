import argparse
import contextlib
import os
import queue
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from datetime import datetime

import librosa
import numpy as np
import psutil
import soundfile as sf
from pydub.utils import mediainfo
from scipy.signal import correlate
from scipy.spatial.distance import cosine
import pyloudnorm as pyln

class InfiniLoopTerminal:
    def __init__(self):

        self.base_dir = os.path.abspath(".")
        self.FILE1 = os.path.join(self.base_dir, "music1.wav")
        self.FILE2 = os.path.join(self.base_dir, "music2.wav")
        self.CURRENT = self.FILE1
        self.NEXT = self.FILE2

        self.CROSSFADE_MS = 3000
        self.CROSSFADE_SEC = self.CROSSFADE_MS / 1000.0
        self.PROMPT = ""
        self.model = "medium"
        self.duration = 8
        if os.path.exists("/proc/asound") and os.access("/dev/snd", os.R_OK | os.X_OK):
            self.audio_driver = "alsa"
        else:
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
        self._temp_files_to_cleanup = []

    def cleanup_temp_files(self):

        for temp_file in self._temp_files_to_cleanup:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                self.log_message(f"⚠️ Could not clean temp file, kinda? {temp_file}: {e}")
        self._temp_files_to_cleanup.clear()



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

    def find_optimal_zero_crossing(self, y, sample, window_size=512):


        adaptive_window = min(window_size, len(y) // 100)
        start = max(0, sample - adaptive_window // 2)
        end = min(len(y), sample + adaptive_window // 2)

        if end - start < 4:
            return sample


        y_segment = y[start:end]
        signs = np.sign(y_segment)
        sign_changes = np.where(np.diff(signs) != 0)[0] + start

        if len(sign_changes) == 0:
            return sample


        amplitudes = np.abs(y[sign_changes]) + np.abs(y[sign_changes + 1])
        best_idx = np.argmin(amplitudes)

        return sign_changes[best_idx]

    def calculate_waveform_continuity(self, y, start, end, sr):


        t = max(128, min(sr // 30, (end - start) // 15))


        a_start = max(0, end - t)
        b_end = min(len(y), start + t)

        a = y[a_start:end]
        b = y[start:b_end]

        if len(a) < 32 or len(b) < 32:
            return 0.0


        min_len = min(len(a), len(b))
        a, b = a[-min_len:], b[:min_len]


        a_norm = a - np.mean(a)
        b_norm = b - np.mean(b)

        if np.std(a_norm) > 1e-8 and np.std(b_norm) > 1e-8:
            corr = np.corrcoef(a_norm, b_norm)[0, 1]
            corr_score = abs(corr) if not np.isnan(corr) else 0.0
        else:
            corr_score = 0.0


        rms_a, rms_b = np.sqrt(np.mean(a**2)), np.sqrt(np.mean(b**2))
        max_rms = max(rms_a, rms_b, 1e-8)
        rms_diff = abs(rms_a - rms_b) / max_rms
        rms_score = max(0.0, 1.0 - rms_diff)


        try:
            fft_a = np.abs(np.fft.rfft(a))
            fft_b = np.abs(np.fft.rfft(b))
            if len(fft_a) > 1 and len(fft_b) > 1:
                spectral_corr = np.corrcoef(fft_a, fft_b)[0, 1]
                spectral_score = abs(spectral_corr) if not np.isnan(spectral_corr) else 0.0
            else:
                spectral_score = 0.0
        except:
            spectral_score = 0.0


        if min_len > 2:
            da, db = np.diff(a), np.diff(b)
            d_last, d_first = da[-1], db[0]
            max_d = max(abs(d_last), abs(d_first), 1e-8)
            deriv_score = max(0.0, 1.0 - abs(d_last - d_first) / max_d)
        else:
            deriv_score = 1.0


        return (corr_score * 0.35 + rms_score * 0.25 +
                spectral_score * 0.25 + deriv_score * 0.15)


    def calculate_beat_alignment(self, start_sample, end_sample, beats, sr):

        if len(beats) == 0:
            return 0.5


        if len(beats) > 1:
            avg_beat_interval = np.mean(np.diff(beats))
            tolerance = avg_beat_interval * 0.1
        else:
            tolerance = sr * 0.1


        d_start = np.min(np.abs(beats - start_sample))
        d_end = np.min(np.abs(beats - end_sample))


        def alignment_score(distance, tolerance):
            if distance <= tolerance:
                return 1.0
            else:

                return np.exp(-((distance - tolerance) / tolerance))

        align_start = alignment_score(d_start, tolerance)
        align_end = alignment_score(d_end, tolerance)


        both_aligned_bonus = 0.1 if (align_start > 0.8 and align_end > 0.8) else 0.0

        return min(1.0, (align_start + align_end) / 2 + both_aligned_bonus)


    def calculate_phase_continuity(self, S, start_frame, end_frame, window=3):

        if start_frame < window or end_frame >= S.shape[1] - window:
            return 0.5

        start_ph = np.angle(S[:, start_frame - window:start_frame + window])
        end_ph = np.angle(S[:, end_frame - window:end_frame + window])

        diff = np.abs(np.mean(start_ph, axis=1) - np.mean(end_ph, axis=1))
        diff = np.minimum(diff, 2 * np.pi - diff)

        return max(0.0, 1 - np.mean(diff) / np.pi)

    def find_perfect_loop_simple(self, y, sr):

        self.log_message("🥁 Switching to Beat-focused algorithm...")


        try:

            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='samples')

            if len(beats) < 2:
                raise Exception("Not enough beats detected")


            beat_intervals = np.diff(beats)
            avg_beat_interval = float(np.median(beat_intervals))
            beat_consistency = float(1.0 - (np.std(beat_intervals) / avg_beat_interval))


            tempo_value = float(tempo) if not isinstance(tempo, np.ndarray) else float(tempo[0])

            self.log_message(f"🥁 BPM: {tempo_value:.1f}, Cons: {beat_consistency:.3f}")
            self.log_message(f"🥁 Beats: {len(beats)}, avg int: {avg_beat_interval/sr:.3f}s\n")

        except Exception as e:
            self.log_message(f"⚠️ Failed: {e}")

            tempo_value = 120.0
            avg_beat_interval = float(0.5 * sr)
            beats = np.arange(0, len(y), int(avg_beat_interval))
            beat_consistency = 0.5

        beat_duration = avg_beat_interval / sr
        total_duration = len(y) / sr


        if len(beats) >= 8:

            preferred_structures = [
                (4, "1 measure"), (8, "2 measures"), (16, "4 measures"),
                (12, "3 measures"), (2, "half measure"), (6, "1.5 measures")
            ]
        elif len(beats) >= 4:

            preferred_structures = [
                (2, "half measure"), (3, "3 beats"), (4, "1 measure"),
                (6, "1.5 measures"), (8, "2 measures")
            ]
        else:

            preferred_structures = [
                (1, "1 beat"), (2, "half measure"), (3, "3 beats"),
                (4, "1 measure"), (5, "5 beats")
            ]

        candidates = []

        for num_beats, description in preferred_structures:
            target_duration_samples = int(num_beats * avg_beat_interval)
            target_duration_sec = target_duration_samples / sr


            if not (1.5 <= target_duration_sec <= min(total_duration * 0.95, 30.0)):
                continue

            max_start_sample = len(y) - target_duration_samples
            valid_candidates_for_structure = 0


            search_positions = []


            for beat_start in beats:
                if beat_start <= max_start_sample:
                    search_positions.append(int(beat_start))


            if len(search_positions) < 10:
                step = int(0.25 * sr)
                for pos in range(0, max_start_sample, step):
                    if pos not in search_positions:
                        search_positions.append(pos)

            for start_sample in search_positions:
                end_sample = start_sample + target_duration_samples

                if end_sample > len(y):
                    continue


                try:

                    beat_score = self.calculate_beat_alignment(start_sample, end_sample, beats, sr)


                    waveform_score = self.calculate_waveform_continuity(y, start_sample, end_sample, sr)


                    start_beat_distance = float(min(np.abs(beats - start_sample))) if len(beats) > 0 else float('inf')
                    end_beat_distance = float(min(np.abs(beats - end_sample))) if len(beats) > 0 else float('inf')

                    perfect_alignment_bonus = 0.0
                    tolerance = avg_beat_interval * 0.1

                    if start_beat_distance < tolerance:
                        perfect_alignment_bonus += 0.2
                    if end_beat_distance < tolerance:
                        perfect_alignment_bonus += 0.2


                    structure_bonus = 0.0
                    if num_beats in [4, 8, 16]:
                        structure_bonus = 0.15
                    elif num_beats in [2, 12, 3]:
                        structure_bonus = 0.1
                    elif num_beats in [1, 5, 6]:
                        structure_bonus = 0.05


                    consistency_bonus = beat_consistency * 0.1


                    rhythm_focused_score = (
                        beat_score * 0.50 +
                        waveform_score * 0.35 +
                        perfect_alignment_bonus * 0.5 +
                        structure_bonus +
                        consistency_bonus +
                        0.1
                    )

                    candidates.append({
                        'start': start_sample,
                        'end': end_sample,
                        'score': rhythm_focused_score,
                        'beats': num_beats,
                        'description': description,
                        'metrics': {
                            'Beat Align': beat_score,
                            'Waveform': waveform_score,
                            'Perfect Alignment': perfect_alignment_bonus,
                            'Structure': structure_bonus,
                            'Consistency': consistency_bonus
                        }
                    })

                    valid_candidates_for_structure += 1

                except Exception as e:
                    continue

            if valid_candidates_for_structure > 0:
                self.log_message(f"✅ {valid_candidates_for_structure} candidates for {description}")


        if not candidates:
            raise Exception("No candidates found")


        candidates.sort(key=lambda x: x['score'], reverse=True)
        best = candidates[0]


        if best['score'] < 0.15:
            raise Exception(f"Best score too low: {best['score']:.3f}")


        self.log_message("🎯 Beat-preserving zero-crossing optimization...\n")

        original_start, original_end = best['start'], best['end']

        try:

            small_window = min(512, int(avg_beat_interval * 0.1))

            optimized_start = self.find_optimal_zero_crossing(y, original_start, window_size=small_window)
            optimized_end = self.find_optimal_zero_crossing(y, original_end, window_size=small_window)


            if 0 <= optimized_start < optimized_end <= len(y):

                new_beat_score = self.calculate_beat_alignment(optimized_start, optimized_end, beats, sr)


                if new_beat_score >= best['metrics']['Beat Align'] * 0.8:
                    best['start'] = optimized_start
                    best['end'] = optimized_end
                    best['metrics']['Beat Align'] = new_beat_score
                else:
                    self.log_message("⚠️ Zero-crossing rejected (would compromise beat alignment)")
            else:
                self.log_message("⚠️ Zero-crossing optimization produced invalid bounds")

        except Exception as e:
            self.log_message(f"⚠️ Zero-crossing optimization failed: {e}")


        duration = (best['end'] - best['start']) / sr

        self.log_message(f"✅ Potential loop found!\n              Checking duration...\n")


        return {
            'start_sample': best['start'],
            'end_sample': best['end'],
            'score': best['score'],
            'measures': max(1, best['beats'] // 4),
            'metrics': {
                'Spectral': 0.5,
                'Waveform': best['metrics']['Waveform'],
                'Beat Align': best['metrics']['Beat Align'],
                'Phase': 0.5
            }
        }

    def find_perfect_loop(self, y, sr):

        try:

            return self.find_perfect_loop_advanced(y, sr)
        except Exception as e:
            self.log_message(f"❌ Failed!\n")

            return self.find_perfect_loop_simple(y, sr)

    def find_perfect_loop_advanced(self, y, sr):

        self.log_message("🧠 Advanced loop detection:")


        if not len(y) or sr <= 0:
            raise Exception(f"Invalid input: empty audio from AI?")


        try:
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='samples')
            tempo = float(tempo)
        except Exception as e:
            raise Exception(f"Beat tracking error, bad AI generation maybe: {e}")

        if not 30 < tempo <= 300:
            raise Exception(f"Invalid tempo: {tempo} BPM, are you serious?")


        hop_length = 256
        S_complex = librosa.stft(y, n_fft=2048, hop_length=hop_length)
        S_mag = np.abs(S_complex)

        if not S_mag.size:
            raise Exception("Empty STFT")

        beat_len = 60 / tempo
        best = {'score': -np.inf, 'start': 0, 'end': 0, 'measures': 0, 'metrics': {}}


        mel_features = librosa.feature.melspectrogram(
            S=S_mag**2, sr=sr, hop_length=hop_length, n_mels=64
        )
        mel_features = librosa.power_to_db(mel_features)


        def calculate_all_metrics(start, end, sf, ef):
            metrics = {}


            try:
                mel_start = np.mean(mel_features[:, max(0, sf-2):sf+3], axis=1)
                mel_end = np.mean(mel_features[:, ef-2:min(mel_features.shape[1], ef+3)], axis=1)

                if np.any(np.isnan(mel_start)) or np.any(np.isnan(mel_end)):
                    metrics['spectral'] = 0.0
                else:
                    metrics['spectral'] = max(0.0, 1 - cosine(mel_start, mel_end))
            except:
                metrics['spectral'] = 0.0


            try:
                metrics['waveform'] = self.calculate_waveform_continuity(y, start, end, sr)
            except:
                metrics['waveform'] = 0.0


            try:
                metrics['beat'] = self.calculate_beat_alignment(start, end, beats, sr)
            except:
                metrics['beat'] = 0.5


            try:
                if sf >= 3 and ef < S_complex.shape[1] - 3:
                    phase_start = np.angle(S_complex[:, sf-1:sf+2])
                    phase_end = np.angle(S_complex[:, ef-1:ef+2])


                    important_freqs = slice(0, S_complex.shape[0] // 2)

                    diff = np.abs(np.mean(phase_start[important_freqs], axis=1) -
                                np.mean(phase_end[important_freqs], axis=1))
                    diff = np.minimum(diff, 2 * np.pi - diff)
                    metrics['phase'] = max(0.0, 1 - np.mean(diff) / np.pi)
                else:
                    metrics['phase'] = 0.5
            except:
                metrics['phase'] = 0.5

            return metrics


        best_score_threshold = 0.8

        for meas in [4, 8, 12, 16]:
            samples = int(meas * 4 * beat_len * sr)

            if not (3 * sr <= samples <= len(y) * 0.85):
                continue


            search_step = max(512, samples // 100)
            start_range = range(
                int(len(y) * 0.05),
                len(y) - samples - int(len(y) * 0.05),
                search_step
            )

            for start in start_range:
                end = start + samples
                if end > len(y):
                    continue

                sf, ef = start // hop_length, end // hop_length
                if sf < 3 or ef >= S_mag.shape[1] - 3:
                    continue


                metrics = calculate_all_metrics(start, end, sf, ef)


                score = (metrics['spectral'] * 0.4 +
                        metrics['waveform'] * 0.3 +
                        metrics['beat'] * 0.2 +
                        metrics['phase'] * 0.1)

                if score > best['score']:
                    best.update({
                        'score': score, 'start': start, 'end': end,
                        'measures': meas, 'metrics': metrics
                    })


                    if score > best_score_threshold:
                        self.log_message(f"🎯 Score: {score:.3f}, Excellent!")
                        break


            if best['score'] > best_score_threshold:
                break


        if best['score'] < 0.15:
            raise Exception(f"No interesting loop.")

        dur = (best['end'] - best['start']) / sr
        if dur < 1.5:
            raise Exception(f"Loop too short: {dur:.1f}s\n   Discarding sample...\n")

        self.log_message(f"✅ Perfect loop found?\n")

        return {
            'start_sample': best['start'],
            'end_sample': best['end'],
            'score': best['score'],
            'measures': best['measures'],
            'metrics': best['metrics']
        }

    def process_loop_detection(self, input_file, output_file):

        MIN_LOOP_DURATION = 2.6
        MAX_LOOP_ATTEMPTS = 1

        try:
            self.debug_file_state("PRE_LOOP_DETECTION", input_file)

            if not self.validate_audio_file(input_file):
                raise Exception(f"Invalid input file: {input_file}")

            y, sr = librosa.load(input_file, sr=None, mono=True)


            target_rms = 0.1
            current_rms = np.sqrt(np.mean(y ** 2))
            if current_rms > 0:
                y = y * (target_rms / current_rms)

            validations = [
                (not len(y), "Loaded audio is empty"),
                (sr <= 0, f"Invalid sample rate: {sr}"),
                (np.isnan(y).any() or np.isinf(y).any(), "Audio contains bad values"),
                (len(y) / sr < 2.0, f"Audio too short for loop detection: {len(y)/sr:.1f}s")
            ]

            for condition, message in validations:
                if condition:
                    raise Exception(message)


            for attempt in range(1, MAX_LOOP_ATTEMPTS + 1):
                try:

                    loop_info = self.find_perfect_loop(y, sr)
                    s, e = loop_info['start_sample'], loop_info['end_sample']

                    if not (0 <= s < e <= len(y)):
                        raise Exception(f"Invalid loop bounds: {s} -> {e} (max: {len(y)})")


                    initial_duration = (e - s) / sr

                    if initial_duration < MIN_LOOP_DURATION:
                        if attempt < MAX_LOOP_ATTEMPTS:
                            self.log_message(f"   ⚠️ Too short ({initial_duration:.1f}s < {MIN_LOOP_DURATION}s), retrying...\n")
                            continue
                        else:
                            raise Exception(f"{initial_duration:.1f} seconds is too short\n")

                    self.log_message("🎯 Zero-crossing optimization...")
                    s_opt, e_opt = (self.find_optimal_zero_crossing(y, pos) for pos in (s, e))

                    if not (0 <= s_opt < e_opt <= len(y)):

                        self.log_message("⚠️ Zero-crossing failed, using original positions")
                        s_opt, e_opt = s, e


                    final_duration = (e_opt - s_opt) / sr

                    if final_duration < MIN_LOOP_DURATION:
                        if attempt < MAX_LOOP_ATTEMPTS:
                            self.log_message(f"   ⚠️ Too short after optimization, retrying...")
                            continue
                        else:
                            raise Exception(f"All attempts failed!")

                    print("\n📊 Loop metrics:")
                    for k, v in loop_info['metrics'].items():
                        print(f"   {k}: {v:.3f}")

                    y_loop = y[s_opt:e_opt]

                    if np.isnan(y_loop).any() or np.isinf(y_loop).any():
                        raise Exception("Loop contains invalid values")


                    fade_samples = int(sr * 0.01)
                    if fade_samples:
                        fade_in, fade_out = np.linspace(0, 1, fade_samples), np.linspace(1, 0, fade_samples)
                        y_loop[:fade_samples] *= fade_in
                        y_loop[-fade_samples:] *= fade_out

                    if os.path.exists(output_file):
                        os.remove(output_file)

                    try:
                        sf.write(output_file, y_loop, sr)
                    except Exception as err:
                        raise Exception(f"Error writing audio file: {err}")

                    if os.path.getsize(output_file) < 1024:
                        raise Exception("Output too small")

                    self.debug_file_state("POST_LOOP_DETECTION", output_file)

                    if not self.validate_audio_file(output_file):
                        raise Exception("Output validation failed")

                    self.log_message(f"🧬 Loop ready! {final_duration:.1f}s\n")


                    return

                except Exception as attempt_error:
                    if attempt < MAX_LOOP_ATTEMPTS:
                        self.log_message(f"   ❌ Attempt {attempt} failed: {attempt_error}")
                        continue
                    else:

                        raise attempt_error


            raise Exception(f"Failed to find loop after {MAX_LOOP_ATTEMPTS} attempts")

        except Exception as e:
            if os.path.exists(output_file):
                try:
                    os.remove(output_file)
                except:
                    pass

            raise


    def generate_audio_safe(self, outfile):
        try:
            self.is_generating = True
            prompt = self.PROMPT
            model = self.model
            duration = self.duration
            self.generation_status = f"Generating '{prompt}'"
            self.log_message("🎼 Generating new sample...\n")

            with self.safe_temp_file() as raw_temp, self.safe_temp_file() as processed_temp:
                self.debug_file_state("PRE_GENERATION", raw_temp)

                start_time = time.time()

                result = subprocess.run([
                    "ionice", "-c", "2", "-n", "7",
                    "nice", "-n", "10",
                    "./musicgpt-x86_64-unknown-linux-gnu",
                    prompt,
                    "--model", model,
                    "--secs", str(duration),
                    "--output", raw_temp,
                    "--no-playback",
                    "--no-interactive",
                    "--ui-no-open"
                ], check=True, capture_output=True, text=True)

                elapsed = time.time() - start_time
                self.log_message(f"⏱️ AI made it in {elapsed:.2f}s!")

                self.debug_file_state("POST_GENERATION", raw_temp)

                if not self.validate_audio_file(raw_temp):
                    raise Exception("AI generated invalid audio file")

                # 🔊 Normalizzazione LUFS (volume percepito)
                import pyloudnorm as pyln
                y, sr = librosa.load(raw_temp, sr=None, mono=True)
                meter = pyln.Meter(sr)
                loudness = meter.integrated_loudness(y)
                target_lufs = -14.0  # standard Spotify/YouTube

                y = pyln.normalize.loudness(y, loudness, target_lufs)

                peak = np.max(np.abs(y))
                if peak > 0.999:
                    y /= peak  # evitato clipping
                    self.log_message("🎚️ Peak limited to avoid clipping")

                sf.write(raw_temp, y, sr)
                self.log_message(f"🎚️ Normalized from {loudness:.2f} to {target_lufs}\n")

                os.system("cls" if os.name == "nt" else "clear")
                self.generation_status = "Loop analysis running..."

                self.process_loop_detection(raw_temp, processed_temp)

                if not self.validate_audio_file(processed_temp):
                    raise Exception("Processed file validation failed")

                self.debug_file_state("PRE_FINAL_MOVE", processed_temp)
                with self.file_lock:
                    shutil.move(processed_temp, outfile)
                self.debug_file_state("POST_FINAL_MOVE", outfile)

                self.generation_status = "Completed!"

        except subprocess.CalledProcessError as e:
            self.log_message(f"❌ Generation error: {e}\n{e.stderr.strip()}")
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
        process = None
        try:
            if not self.validate_audio_file(filepath):
                self.log_message(f"❌ Invalid file for playback, can't tell why: {filepath}")
                return
            env = os.environ.copy()
            env["SDL_AUDIODRIVER"] = self.audio_driver
            self.debug_file_state("START_PLAYBACK", filepath)
            process = subprocess.Popen(
                [
                    "ionice", "-c", "2", "-n", "0",
                    "taskset", "-c", "2",
                    "ffplay",
                    "-nodisp",
                    "-autoexit",
                    "-loglevel", "quiet",
                    "-infbuf",
                    "-probesize", "32",
                    "-analyzeduration", "0",
                    "-loop", "0",  # Loop infinito nativo di ffplay
                    "-f", "wav",
                    os.path.abspath(filepath)
                ],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return_code = process.wait()
            self.debug_file_state("END_PLAYBACK", filepath)
            if return_code != 0:
                self.log_message(f"❌ ffplay terminated with code {return_code}")
        except subprocess.TimeoutExpired:
            self.log_message("❌ ffplay timeout - forced termination")
            if process:
                self._kill_process_safely(process)
        except Exception as e:
            self.log_message(f"❌ ffplay crash detected: {str(e)}")
            if process:
                self._kill_process_safely(process)
        finally:
            if process and process.poll() is None:
                self._kill_process_safely(process)


    def loop_current_crossfade_blocking(self, filepath, crossfade_sec_unused, stop_event):
        """
        Loop infinito di CURRENT, senza gestire il crossfade qui
        """
        try:
            if not self.validate_audio_file(filepath):
                self.log_message(f"❌ Invalid file for loop: {filepath}")
                return

            duration = self.get_duration(filepath)
            if duration <= 0:
                self.log_message(f"❌ Invalid duration: {filepath}")
                return

            title = self.get_random_title()
            artist = self.get_random_artist()
            print(f"\n🎧 NOW PLAYING:")
            print(f"   Title:   {title}")
            print(f"   Artist:  {artist}")
            print(f"   Loop:    {duration:.1f} seconds")
            print(f"   Genre:   {self.PROMPT}\n")

            env = os.environ.copy()
            env["SDL_AUDIODRIVER"] = self.audio_driver

            self.debug_file_state("START_PLAYBACK", filepath)

            # Avvia il processo con loop infinito
            current_process = subprocess.Popen([
                "ionice", "-c", "2", "-n", "0",
                "taskset", "-c", "2",
                "ffplay",
                "-nodisp",
                "-autoexit",
                "-loglevel", "quiet",
                "-infbuf",
                "-probesize", "32",
                "-analyzeduration", "0",
                "-loop", "0",
                "-f", "wav",
                os.path.abspath(filepath)
            ], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Attendi fino a stop_event senza fare crossfade
            while not stop_event.is_set() and current_process.poll() is None:
                stop_event.wait(0.1)

            # Termina il processo quando richiesto
            if current_process.poll() is None:
                self._kill_process_safely(current_process)

            self.debug_file_state("END_PLAYBACK", filepath)

        except Exception as e:
            self.log_message(f"❌ Error in playback: {str(e)}")
            if 'current_process' in locals() and current_process.poll() is None:
                self._kill_process_safely(current_process)


    def _kill_process_safely(self, process):
        """Terminazione sicura del processo con timeout"""
        try:
            # Prima prova SIGTERM
            process.terminate()
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            # Poi forza con SIGKILL
            process.kill()
            process.wait(timeout=1.0)
        except Exception:
            pass



    def safe_file_swap(self):
        with self.swap_lock:
            try:
                if not self.validate_audio_file(self.NEXT):
                    raise Exception(f"❌ Invalid NEXT file: {self.NEXT}")

                # Applica fade-in una tantum direttamente al file NEXT

                env = os.environ.copy()
                env["SDL_AUDIODRIVER"] = self.audio_driver

                self.log_message("🎵 Mixing loops...\n")

                # Avvia NEXT in background PRIMA di fermare CURRENT
                next_process = subprocess.Popen([
                    "ionice", "-c", "2", "-n", "0",
                    "taskset", "-c", "3",
                    "ffplay",
                    "-nodisp",
                    "-autoexit",
                    "-loglevel", "quiet",
                    "-infbuf",
                    "-probesize", "32",
                    "-analyzeduration", "0",
                    "-loop", "0",
                    "-f", "wav",
                    os.path.abspath(self.NEXT)
                ], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                next_pid = next_process.pid

                time.sleep(self.CROSSFADE_SEC)

                self.stop_event.set()

                if self.loop_thread and self.loop_thread.is_alive():
                    self.loop_thread.join(timeout=2.0)

                self.kill_all_ffplay_processes(exclude_pid=next_pid)

                with self.file_lock:
                    self.CURRENT, self.NEXT = self.NEXT, self.CURRENT

                self.stop_event = threading.Event()

                return True

            except Exception as e:
                self.log_message(f"❌ Error during swap: {str(e)}")
                self.stop_event = threading.Event()
                return False



    def run_loop(self):
        self.stop_event = threading.Event()
        self.loop_thread = threading.Thread(
            target=self.loop_current_crossfade_blocking,
            args=(self.CURRENT, self.CROSSFADE_SEC, self.stop_event),
            daemon=True
        )
        self.loop_thread.start()

        consecutive_errors = 0
        max_consecutive_errors = 2

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

                # NON avviamo un nuovo thread qui perché safe_file_swap
                # ha già avviato il processo ffplay per il nuovo CURRENT
                # Aspettiamo solo che sia necessario il prossimo swap

                consecutive_errors = 0

            except Exception as e:
                consecutive_errors += 1
                self.log_message(f"❌ Error! {str(e)}")


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
                    self.log_message("❌ Too many errors, give me a break. Stopping application")
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
                    self.log_message(f"❌ File removal error: {remove_error}. Huh.")

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
        self.cleanup_temp_files()

    def kill_all_ffplay_processes(self, exclude_pid=None):
        """
        Termina tutti i processi ffplay TRANNE quello con exclude_pid
        """
        try:
            result = subprocess.run(["pgrep", "-f", "ffplay"], capture_output=True, text=True)
            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    try:
                        pid_int = int(pid)
                        # Salta il PID da escludere
                        if exclude_pid and pid_int == exclude_pid:
                            continue

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
                self.log_message(f"❌ Save error: {str(e)}. Dunno...")
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
            traceback.print_exc()
            continue

def main():
    parser = argparse.ArgumentParser(
        description="InfiniLoop TERMINAL - Local AI Infinite Music Generator",
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
        sys.exit(1)

if __name__ == "__main__":
    main()
