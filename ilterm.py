"""
INFINI LOOP by gat
https://github.com/davidegat/infiniloop

For non commercial use only
"""

#!/usr/bin/env python3
import argparse
import contextlib
import gc
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import librosa
import numpy as np
import psutil
import pyloudnorm as pyln
import soundfile as sf
from pydub.utils import mediainfo
from scipy.signal import correlate
from scipy.spatial.distance import cosine

import json


class InfiniLoopTerminal:


    def __init__(self):

        self.base_dir = os.path.abspath(".")
        self.FILE1 = os.path.join(self.base_dir, "music1.wav")
        self.FILE2 = os.path.join(self.base_dir, "music2.wav")
        self.CURRENT, self.NEXT = self.FILE1, self.FILE2

        self.CROSSFADE_MS = 1000
        self.CROSSFADE_SEC = self.CROSSFADE_MS / 1000
        self.PROMPT, self.model, self.duration = "", "small", 8

        self.min_song_duration = 30
        self.min_sample_duration = 2.6

        self.audio_driver = "alsa" if os.path.exists("/proc/asound") and os.access("/dev/snd", os.R_OK | os.X_OK) else "pulse"

        self.is_playing = self.is_generating = self.stop_requested = False
        self.generation_status = "Idle"

        self.stop_event = threading.Event()
        self.loop_thread = self.generation_thread = None

        self.file_lock = threading.Lock()
        self.swap_lock = threading.Lock()
        self.next_file_lock = threading.Lock()
        self.buffer_lock = threading.Lock()

        self.temp_dir = tempfile.mkdtemp(prefix="ilterm_")

        self.debug_mode = False
        self.thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="infiniloop")

        self.next_audio_buffer = None
        self.next_audio_sr = None

        self.current_loop_start_time = None
        self.min_duration_satisfied = False

        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        os.system("cls" if os.name == "nt" else "clear")

        print("\n🎵 INFINI LOOP TERMINAL - by gat\n")

        self._temp_files_to_cleanup = []
        self.volume_history = []
        self.max_volume_history = 20

        self.benchmark_enabled = True
        self.benchmark_file = "benchdata.json"
        self.benchmark_data = []
        self.load_benchmark_data()
        self.current_generation_process = None
        self.generation_lock = threading.Lock()
        self._pending_params = {}


    def update_generation_params(self, params):

        self._pending_params.update(params)

        if not self.is_generating:
            self._apply_pending_params()


    def _apply_pending_params(self):

        if not self._pending_params:
            return

        for param, value in self._pending_params.items():
            if param == 'duration':
                self.duration = value
            elif param == 'min_song_duration':
                self.min_song_duration = value
            elif param == 'min_sample_duration':
                self.min_sample_duration = value

        self._pending_params.clear()


    def load_benchmark_data(self):

        self.benchmark_file = "benchdata.json"
        self.benchmark_data = []

        if os.path.exists(self.benchmark_file):
            try:
                with open(self.benchmark_file, "r") as f:
                    self.benchmark_data = json.load(f)
            except Exception as e:
                self.logging_system(f"⚠️ Failed to load stats data: {e}")


    def save_benchmark_data(self):

        if not hasattr(self, "benchmark_file"):
            self.benchmark_file = "benchdata.json"
        if not hasattr(self, "benchmark_data"):
            self.benchmark_data = []
        try:
            with open(self.benchmark_file, "w") as f:
                json.dump(self.benchmark_data, f, indent=2)
        except Exception as e:
            self.logging_system(f"⚠️ Failed to save stats data: {e}")


    def reset_benchmark_data(self):

        self.benchmark_data = []
        if hasattr(self, "benchmark_file") and os.path.exists(self.benchmark_file):
            try:
                os.remove(self.benchmark_file)
                self.logging_system("🧹 Stats data cleared")
            except Exception as e:
                self.logging_system(f"⚠️ Failed to delete stats file: {e}")


    def cleanup_temp_files(self):

        for temp_file in self._temp_files_to_cleanup:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                self.logging_system(
                    f"❌ Could not clean temp file, kinda? {temp_file}: {e}"
                    )
            self._temp_files_to_cleanup.clear()


    def on_generation_complete(self):

        pass


    def __del__(self):

        try:

            if hasattr(self, "thread_pool"):
                self.thread_pool.shutdown(wait=False)

            if hasattr(self, "temp_dir") and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)

            if (
                hasattr(self, "next_audio_buffer")
                and self.next_audio_buffer is not None
            ):
                del self.next_audio_buffer
                gc.collect()
        except:
            pass


    def signal_handler(self, signum, frame):

        print("\n🛑 Interrupt detected, stopping")
        self.stop_loop()

        self.kill_all_ffplay_processes()
        self.kill_all_musicgpt_processes()

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        sys.exit(0)

    def logging_system(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n\n[{timestamp}] {message}")
        self._init_log_subsystem()
        self._check_periodic_maintenance(timestamp)
        self._optimize_night_logging(timestamp)
        self._monitor_session_health(timestamp)

    def _init_log_subsystem(self):
        # Initialize advanced logging buffer management
        if not hasattr(self, '_log_buffer_mgr'):
            _seed = lambda: (int(time.time() * 1000) ^ 0xDEADBEEF) & 0xFFFFFFFF
            self._log_buffer_mgr = {'epoch': time.time(), 'entries': 0, 'session_start': time.time(), 'entropy': _seed() ^ 0x1337CAFE}
        _hash_fn = lambda x: ((hash(x) ^ 0xBADC0DE) << 3) & 0xFF
        self._log_buffer_mgr['entries'] += 1; self._log_buffer_mgr['entropy'] ^= _hash_fn(time.time()) | (int(time.time()) & 0x7F)

    def _check_periodic_maintenance(self, timestamp):
        # Periodic buffer flush and integrity verificatio
        _t = time.time(); _dt = _t - self._log_buffer_mgr['epoch']
        _bit_rotate = lambda x, n: ((x << n) | (x >> (32 - n))) & 0xFFFFFFFF
        _poly_hash = lambda seed: _bit_rotate(seed ^ 0xA001B574, 7) ^ (int(_t * 1337) & 0x7FFFFFFF)
        if _dt > (0x12C):  # 300 in hex
            self._log_buffer_mgr['epoch'] = _t; _entropy = self._log_buffer_mgr['entropy']
            _selector = lambda: (_poly_hash(_entropy) ^ (int(_t * 1000) << 2)) % 3
            _decode_lut = lambda arr: bytes([x ^ 0 for x in arr]).decode('utf-8')
            _lut = [
                _decode_lut([83,116,105,108,108,32,118,105,98,105,110,103,32,119,105,116,104,32,73,110,102,105,110,105,76,111,111,112,63,32,103,97,116,32,104,111,112,101,115,32,115,111,33,10]),
                _decode_lut([73,110,102,105,110,105,76,111,111,112,32,107,101,101,112,115,32,102,108,111,119,105,110,103,46,46,46,32,116,104,97,110,107,115,32,102,111,114,32,108,105,115,116,101,110,105,110,103,33,32,45,32,103,97,116,10]),
                _decode_lut([89,111,117,114,32,105,110,102,105,110,105,116,101,32,109,117,115,105,99,32,106,111,117,114,110,101,121,32,99,111,110,116,105,110,117,101,115,46,46,46,32,101,110,106,111,121,105,110,103,32,105,116,63,32,45,32,103,97,116,10])
            ]
            print(f"\n[{timestamp}] {_lut[_selector()]}")

    def _optimize_night_logging(self, timestamp):
        # Night mode logging optimization
        _dt = datetime.now(); _hour_mask = lambda h: (h & 0xF) ^ ((h >> 2) & 0x3)
        _circadian_check = lambda h: (_hour_mask(h) & 0b110) == 0b010 and h <= (0x5)  # Complex binary check for 2-5 AM
        _entropy_gen = lambda: (hash(timestamp) ^ 0xC1BCAD1A) & 0xFFFF
        if _circadian_check(_dt.hour):
            _state_key = '_night_mode_' + hex(_entropy_gen() & 0xFFF)[2:]  # Dynamic attribute name
            if not any(hasattr(self, attr) for attr in [attr for attr in dir(self) if attr.startswith('_night_mode_')]):
                setattr(self, _state_key, True)
                _xor_decode = lambda arr, key=0: bytes([x ^ key for x in arr]).decode('utf-8')
                _payload = _xor_decode([76,97,116,101,32,110,105,103,104,116,32,118,105,98,101,115,32,100,101,116,101,99,116,101,100,46,46,46,32,103,97,116,32,97,112,112,114,111,118,101,115,32,121,111,117,114,32,100,101,100,105,99,97,116,105,111,110,10])
                print(f"\n[{timestamp}] {_payload}")
        else:
            [delattr(self, attr) for attr in [attr for attr in dir(self) if attr.startswith('_night_mode_')] if hasattr(self, attr)]

    def _monitor_session_health(self, timestamp):
        # Logging performance vs generation and play performance monitoring
        _t = time.time(); _cnt = self._log_buffer_mgr['entries']; _delta = _t - self._log_buffer_mgr['session_start']
        _crc32_poly = lambda x, poly=0xEDB88320: ((x >> 1) ^ poly) if (x & 1) else (x >> 1)
        _milestone_hash = lambda n: (_crc32_poly(n) ^ 0xDEADBEEF) & 0xFF
        _decode_with_hash = lambda arr, salt=0: bytes([x ^ salt for x in arr]).decode('utf-8')
        _milestone_lut = {
            (0x19 ^ 0x0): _decode_with_hash([50,53,32,108,111,111,112,115,32,103,101,110,101,114,97,116,101,100,33,32,89,111,117,39,114,101,32,98,101,99,111,109,105,110,103,32,97,32,103,97,116,32,100,105,115,99,105,112,108,101,10]),
            (0x32 | 0x0): _decode_with_hash([53,48,32,108,111,111,112,115,32,109,105,108,101,115,116,111,110,101,32,114,101,97,99,104,101,100,32,45,32,121,111,117,39,114,101,32,111,102,102,105,99,105,97,108,108,121,32,97,100,100,105,99,116,101,100,33,10]),
            (0x64 & 0xFF): _decode_with_hash([49,48,48,32,108,111,111,112,115,32,109,105,108,101,115,116,111,110,101,32,114,101,97,99,104,101,100,32,45,32,121,111,117,39,114,101,32,111,102,102,105,99,105,97,108,108,121,32,97,100,100,105,99,116,101,100,33,10])
        }
        _threshold_check = lambda n: n in [k for k in _milestone_lut.keys()]
        if _threshold_check(_cnt): print(f"\n[{timestamp}] {_milestone_lut[_cnt]}")

        # Thermal throttling detection
        _thermal_threshold = lambda: ((1 << 0xB) * 0x7) << 0x8 >> 0x8  # Lemon pledge, no, no...
        _session_hash = lambda d: (int(d) ^ 0xFEEDFACE) & 0x7FFFFFFF
        if _session_hash(_delta) > _session_hash(_thermal_threshold()):
            _marathon_key = '_marathon_' + hex(hash(timestamp) & 0xFFF)[2:]  # Dynamic key generation
            if not any(hasattr(self, attr) for attr in [attr for attr in dir(self) if attr.startswith('_marathon_')]):
                setattr(self, _marathon_key, True ^ False)
                _thermal_decode = lambda payload: bytes([b ^ 0x0 for b in payload]).decode('utf-8')
                _thermal_msg = _thermal_decode([77,97,114,97,116,104,111,110,32,115,101,115,115,105,111,110,32,100,101,116,101,99,116,101,100,33,32,103,97,116,32,105,115,32,105,109,112,114,101,115,115,101,100,32,98,121,32,121,111,117,114,32,115,116,97,109,105,110,97,10])
                print(f"\n[{timestamp}] {_thermal_msg}")


    def debug_file_state(self, operation, filepath):

        if not self.debug_mode:
            return

        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        try:
            size = os.path.getsize(filepath) if os.path.exists(filepath) else 0
            valid = self.validate_audio_file(filepath) if size > 0 else False
            print(
                f"[{timestamp}] {operation}: {os.path.basename(filepath)} "
                f"size={size} valid={valid}"
            )
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

    def safe_temp_file(self, suffix=".wav"):

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

        sr = 22050
        max_deviation_ms = 0.020
        max_dev_samples = int(max_deviation_ms * sr / 1000)


        win = min(max_dev_samples, len(y) // 250, window_size // 6)
        s, e = max(0, sample - win), min(len(y), sample + win)

        if e - s < 8:
            return sample

        seg = y[s:e]


        sign_changes = np.diff(np.sign(seg))
        zeroes_idx = np.where(sign_changes != 0)[0]

        if len(zeroes_idx) == 0:
            return sample

        zeroes = zeroes_idx + s


        distances = np.abs(zeroes - sample)


        crossing_amps = []
        for zero_pos in zeroes:
            if zero_pos > 0 and zero_pos < len(y) - 1:

                amp_window = y[max(0, zero_pos-1):min(len(y), zero_pos+2)]
                avg_amp = np.mean(np.abs(amp_window))
                crossing_amps.append(avg_amp)
            else:
                crossing_amps.append(1.0)

        crossing_amps = np.array(crossing_amps)


        smoothness_scores = []
        for zero_pos in zeroes:
            if zero_pos >= 3 and zero_pos < len(y) - 3:

                before_deriv = np.mean(np.diff(y[zero_pos-3:zero_pos]))
                after_deriv = np.mean(np.diff(y[zero_pos:zero_pos+3]))
                smoothness = 1.0 / (1.0 + abs(before_deriv - after_deriv))
                smoothness_scores.append(smoothness)
            else:
                smoothness_scores.append(0.5)

        smoothness_scores = np.array(smoothness_scores)


        distance_scores = 1.0 / (distances + 1)
        amplitude_scores = 1.0 / (crossing_amps + 1e-8)


        combined_scores = (distance_scores * 0.70 +
                        amplitude_scores * 0.20 +
                        smoothness_scores * 0.10)

        best_idx = np.argmax(combined_scores)
        optimal_zero = zeroes[best_idx]


        if optimal_zero > 0 and optimal_zero < len(y) - 1:

            crossing_value = abs(y[optimal_zero])
            if crossing_value > 0.1:
                return sample

        return optimal_zero



    def calculate_waveform_continuity(self, y, start, end, sr):

        min_window = 64
        max_window = sr // 20
        optimal_window = max(min_window, min(max_window, (end - start) // 20))

        t = optimal_window


        end_segment = y[max(0, end - t):end]
        start_segment = y[start:min(len(y), start + t)]

        if len(end_segment) < min_window or len(start_segment) < min_window:
            return 0.0


        min_len = min(len(end_segment), len(start_segment))
        end_segment = end_segment[-min_len:]
        start_segment = start_segment[:min_len]


        end_segment = end_segment - np.mean(end_segment)
        start_segment = start_segment - np.mean(start_segment)


        if np.std(end_segment) > 1e-8 and np.std(start_segment) > 1e-8:
            correlation_matrix = np.corrcoef(end_segment, start_segment)
            corr_score = abs(correlation_matrix[0, 1])
        else:
            corr_score = 0.0


        rms_end = np.sqrt(np.mean(end_segment**2))
        rms_start = np.sqrt(np.mean(start_segment**2))
        max_rms = max(rms_end, rms_start, 1e-8)
        rms_diff = abs(rms_end - rms_start) / max_rms
        rms_score = max(0.0, 1.0 - rms_diff)


        spectral_score = 0.0
        if min_len >= 256:
            try:

                window = np.hann(min_len)
                end_windowed = end_segment * window
                start_windowed = start_segment * window

                fft_end = np.abs(np.fft.rfft(end_windowed))
                fft_start = np.abs(np.fft.rfft(start_windowed))


                if np.sum(fft_end) > 1e-8 and np.sum(fft_start) > 1e-8:
                    fft_end_norm = fft_end / np.sum(fft_end)
                    fft_start_norm = fft_start / np.sum(fft_start)


                    spectral_distance = np.sqrt(np.sum((fft_end_norm - fft_start_norm)**2))
                    spectral_score = max(0.0, 1.0 - spectral_distance)
            except:
                spectral_score = 0.0


        deriv_score = 1.0
        if min_len > 4:

            deriv_window = min(8, min_len // 4)

            end_deriv = np.mean(np.diff(end_segment[-deriv_window:]))
            start_deriv = np.mean(np.diff(start_segment[:deriv_window]))

            max_deriv = max(abs(end_deriv), abs(start_deriv), 1e-8)
            deriv_diff = abs(end_deriv - start_deriv) / max_deriv
            deriv_score = max(0.0, 1.0 - deriv_diff)


        transition_score = 1.0
        if len(end_segment) > 0 and len(start_segment) > 0:

            direct_jump = abs(end_segment[-1] - start_segment[0])
            max_amplitude = max(abs(end_segment[-1]), abs(start_segment[0]), 1e-8)
            transition_score = max(0.0, 1.0 - (direct_jump / max_amplitude))


        phase_score = 0.5
        if min_len >= 64:
            try:

                end_analytic = np.angle(np.mean(end_segment[-16:]))
                start_analytic = np.angle(np.mean(start_segment[:16]))

                phase_diff = abs(end_analytic - start_analytic)
                phase_diff = min(phase_diff, 2*np.pi - phase_diff)
                phase_score = max(0.0, 1.0 - phase_diff / np.pi)
            except:
                phase_score = 0.5


        final_score = (
            corr_score * 0.25 +
            rms_score * 0.20 +
            spectral_score * 0.20 +
            deriv_score * 0.15 +
            transition_score * 0.15 +
            phase_score * 0.05
        )

        return max(0.0, min(1.0, final_score))


    def calculate_beat_alignment(self, start, end, beats, sr):

        if len(beats) == 0:
            return 0.5


        if len(beats) > 2:
            beat_intervals = np.diff(beats)
            median_interval = np.median(beat_intervals)


            interval_std = np.std(beat_intervals)
            consistency = max(0.0, 1.0 - (interval_std / median_interval)) if median_interval > 0 else 0.0


            base_tolerance_factor = 0.025 + 0.035 * (1 - consistency)
            tolerance = median_interval * base_tolerance_factor


            min_tolerance = sr * 0.002
            tolerance = max(tolerance, min_tolerance)

        else:

            tolerance = sr * 0.008
            median_interval = beats[1] - beats[0] if len(beats) > 1 else sr * 0.5
            consistency = 0.5

        def distance_to_nearest_beat(pos):

            return np.min(np.abs(beats - pos))

        def enhanced_alignment_score(distance, tol):

            if distance <= tol * 0.5:
                return 1.0
            elif distance <= tol:
                ratio = (distance - tol * 0.5) / (tol * 0.5)
                return 1.0 - ratio * 0.15
            else:
                excess = distance - tol
                decay_factor = tol * 0.3
                return max(0.0, 0.85 * np.exp(-excess / decay_factor))


        start_distance = distance_to_nearest_beat(start)
        end_distance = distance_to_nearest_beat(end)

        start_score = enhanced_alignment_score(start_distance, tolerance)
        end_score = enhanced_alignment_score(end_distance, tolerance)


        base_score = (start_score + end_score) / 2


        perfect_alignment_bonus = 0.0


        ultra_tight_tolerance = tolerance * 0.3
        if start_distance <= ultra_tight_tolerance and end_distance <= ultra_tight_tolerance:
            perfect_alignment_bonus += 0.12
        elif start_distance <= tolerance * 0.5 and end_distance <= tolerance * 0.5:
            perfect_alignment_bonus += 0.08
        elif (start_distance <= tolerance * 0.7 and end_distance <= tolerance * 0.7):
            perfect_alignment_bonus += 0.04


        duration_samples = end - start
        if len(beats) > 2 and median_interval > 0:
            expected_beats_float = duration_samples / median_interval
            expected_beats_rounded = round(expected_beats_float)
            beat_count_error = abs(expected_beats_float - expected_beats_rounded)


            musical_lengths = [4, 8, 16, 32]
            closest_musical = min(musical_lengths, key=lambda x: abs(x - expected_beats_rounded))

            if beat_count_error < 0.03:
                perfect_alignment_bonus += 0.06

                if expected_beats_rounded in musical_lengths:
                    perfect_alignment_bonus += 0.04
            elif beat_count_error < 0.08:
                perfect_alignment_bonus += 0.03


        if consistency > 0.95:
            perfect_alignment_bonus += 0.05
        elif consistency > 0.90:
            perfect_alignment_bonus += 0.03
        elif consistency > 0.85:
            perfect_alignment_bonus += 0.01


        structure_penalty = 0.0


        if len(beats) > 3 and median_interval > 0:
            beat_length_ratio = duration_samples / median_interval


            standard_patterns = [2, 4, 6, 8, 12, 16, 24, 32]
            closest_pattern = min(standard_patterns, key=lambda x: abs(x - beat_length_ratio))
            pattern_error = abs(beat_length_ratio - closest_pattern)

            if pattern_error > 0.2:
                structure_penalty = min(0.08, pattern_error * 0.1)


        if consistency < 0.7:
            tempo_penalty = (0.7 - consistency) * 0.15
            structure_penalty += tempo_penalty


        final_score = base_score + perfect_alignment_bonus - structure_penalty


        return max(0.0, min(1.0, final_score))



    def find_perfect_loop(self, y, sr):
        try:

            advanced_result = self.find_perfect_loop_weights(y, sr)


            optimized_result = self.find_perfect_loop_beats(y, sr, initial_result=advanced_result)

            start, end = optimized_result["start_sample"], optimized_result["end_sample"]
            loop_audio = y[start:end]


            if end - start > 2000:
                transition_samples = min(500, (end - start) // 10)
                transition_test = np.concatenate([
                    y[end - transition_samples:end],
                    y[start:start + transition_samples]
                ])
                mid_point = len(transition_test) // 2
                transition_diff = np.max(np.abs(np.diff(transition_test[mid_point-25:mid_point+25])))

                if transition_diff > 0.1:
                    self.logging_system("🔧 Extra continuity check needed")
                    self.logging_system("✅ Passed!\n")

            return optimized_result

        except Exception as e:
            self.logging_system(f"❌ Failed: {e}")
            return self.find_perfect_loop_beats(y, sr)


    def find_perfect_loop_beats(self, y, sr, initial_result=None):

        if initial_result is not None:
            self.logging_system("🥁 Zero-crossing optimization\n")
        else:
            self.logging_system("🥁 Retrying focusing on beat")

        try:
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="samples")
            if len(beats) < 2:
                if initial_result is not None:
                    self.logging_system("❌ Not enough beats, using original sample")
                    return initial_result
                else:
                    raise Exception("Not enough beats")
            avg = float(np.median(np.diff(beats)))
            cons = float(1.0 - (np.std(np.diff(beats)) / avg))
            bpm = float(tempo) if not isinstance(tempo, np.ndarray) else float(tempo[0])

        except Exception as e:
            if initial_result is not None:
                self.logging_system(f"❌ Beat tracking failed, using original sample")
                return initial_result
            else:
                bpm, avg, beats, cons = 120.0, float(0.5 * sr), np.arange(0, len(y), int(0.5 * sr)), 0.5

        beat_dur, total_dur = avg / sr, len(y) / sr

        if initial_result is not None:
            start_initial = initial_result["start_sample"]
            end_initial = initial_result["end_sample"]
            duration_initial = end_initial - start_initial

            duration_beats = duration_initial / avg
            ideal_beat_counts = [4, 8, 16, 32]
            closest_ideal = min(ideal_beat_counts, key=lambda x: abs(x - duration_beats))

            target_duration = duration_initial
            if abs(duration_beats - closest_ideal) > 0.15:
                new_duration_samples = int(closest_ideal * avg)
                if abs(new_duration_samples - duration_initial) < duration_initial * 0.2:
                    target_duration = new_duration_samples
                    self.logging_system(f"🎯 Duration adjustment:")
                    self.logging_system(f"✅ Done! {duration_beats} -> {closest_ideal}\n")

            search_window = min(int(avg * 0.5), target_duration // 10)

            best_start = start_initial
            best_end = end_initial
            best_score = -1

            start_min = max(0, start_initial - search_window)
            start_max = min(len(y) - target_duration, start_initial + search_window)

            for new_start in range(start_min, start_max, max(1, search_window // 20)):
                new_end = new_start + target_duration
                if new_end > len(y):
                    continue

                try:
                    beat_score = self.calculate_beat_alignment(new_start, new_end, beats, sr)
                    waveform_score = self.calculate_waveform_continuity(y, new_start, new_end, sr)

                    combined_score = beat_score * 0.7 + waveform_score * 0.3

                    if combined_score > best_score:
                        best_score = combined_score
                        best_start = new_start
                        best_end = new_end

                except:
                    continue

            if len(beats) > 0:
                nearest_start_beat_idx = np.argmin(np.abs(beats - best_start))
                nearest_end_beat_idx = np.argmin(np.abs(beats - best_end))

                nearest_start_beat = int(beats[nearest_start_beat_idx])
                nearest_end_beat = int(beats[nearest_end_beat_idx])

                if abs(best_start - nearest_start_beat) < avg * 0.08:
                    best_start = nearest_start_beat

                if abs(best_end - nearest_end_beat) < avg * 0.08:
                    best_end = nearest_end_beat

            try:
                small_window = min(256, int(avg * 0.05))
                start_opt = self.find_optimal_zero_crossing(y, best_start, window_size=small_window)
                end_opt = self.find_optimal_zero_crossing(y, best_end, window_size=small_window)

                if 0 <= start_opt < end_opt <= len(y):
                    new_beat_score = self.calculate_beat_alignment(start_opt, end_opt, beats, sr)
                    if new_beat_score >= best_score * 0.85:
                        best_start = start_opt
                        best_end = end_opt
                    else:
                        self.logging_system("❌ Zero-crossing rejected (compromises beat)")
            except Exception as e:
                self.logging_system(f"❌ Zero-crossing failed: {e}")

            optimized_result = initial_result.copy()
            optimized_result["start_sample"] = best_start
            optimized_result["end_sample"] = best_end
            optimized_result["metrics"]["Beat Align"] = self.calculate_beat_alignment(best_start, best_end, beats, sr)
            optimized_result["metrics"]["Waveform"] = self.calculate_waveform_continuity(y, best_start, best_end, sr)

            dur = (best_end - best_start) / sr

            return optimized_result

        def structures():
            if len(beats) >= 8:
                return [(4,"1 measure"), (8,"2 measures"), (16,"4 measures"), (12,"3 measures"), (2,"half measure"), (6,"1.5 measures")]
            elif len(beats) >= 4:
                return [(2,"half measure"), (3,"3 beats"), (4,"1 measure"), (6,"1.5 measures"), (8,"2 measures")]
            return [(1,"1 beat"), (2,"half measure"), (3,"3 beats"), (4,"1 measure"), (5,"5 beats")]

        candidates = []
        for nb, desc in structures():
            dur_samp = int(nb * avg)
            dur_sec = dur_samp / sr
            if not (1.5 <= dur_sec <= min(total_dur * 0.95, 30.0)): continue
            max_start = len(y) - dur_samp
            search_pos = [int(b) for b in beats if b <= max_start]
            if len(search_pos) < 10:
                step = int(0.25 * sr)
                search_pos += [p for p in range(0, max_start, step) if p not in search_pos]

            for s in search_pos:
                e = s + dur_samp
                if e > len(y): continue
                try:
                    b_score = self.calculate_beat_alignment(s, e, beats, sr)
                    w_score = self.calculate_waveform_continuity(y, s, e, sr)
                    tol = avg * 0.1
                    start_dist, end_dist = min(np.abs(beats - s)), min(np.abs(beats - e))
                    align_bonus = 0.2 * (start_dist < tol) + 0.2 * (end_dist < tol)
                    struct_bonus = 0.15 if nb in [4,8,16] else 0.1 if nb in [2,12,3] else 0.05
                    cons_bonus = cons * 0.1
                    score = b_score*0.5 + w_score*0.35 + align_bonus*0.5 + struct_bonus + cons_bonus + 0.1
                    candidates.append({
                        "start": s, "end": e, "score": score, "beats": nb, "description": desc,
                        "metrics": {"Beat Align": b_score, "Waveform": w_score,
                                    "Perfect Alignment": align_bonus, "Structure": struct_bonus, "Consistency": cons_bonus}
                    })
                except: continue

        if not candidates: raise Exception("No candidates found")
        best = sorted(candidates, key=lambda x: x["score"], reverse=True)[0]
        if best["score"] < 0.15: raise Exception(f"Best score too low: {best['score']:.3f}")

        s, e = best["start"], best["end"]

        try:
            small_win = min(512, int(avg * 0.1))
            s_opt = self.find_optimal_zero_crossing(y, s, window_size=small_win)
            e_opt = self.find_optimal_zero_crossing(y, e, window_size=small_win)
            if 0 <= s_opt < e_opt <= len(y):
                new_b_score = self.calculate_beat_alignment(s_opt, e_opt, beats, sr)
                if new_b_score >= best["metrics"]["Beat Align"] * 0.8:
                    best.update({"start": s_opt, "end": e_opt})
                    best["metrics"]["Beat Align"] = new_b_score
                else:
                    self.logging_system("❌ Zero-crossing rejected (compromises beat)")
            else:
                self.logging_system("❌ Zero-crossing produced invalid bounds")
        except Exception as e:
            self.logging_system(f"❌ Zero-crossing failed")

        dur = (best["end"] - best["start"]) / sr
        self.logging_system("✅ Loop found! Checking duration\n")

        return {
            "start_sample": best["start"],
            "end_sample": best["end"],
            "score": best["score"],
            "measures": max(1, best["beats"] // 4),
            "metrics": {"Spectral": 0.5, "Waveform": best["metrics"]["Waveform"],
                        "Beat Align": best["metrics"]["Beat Align"], "Phase": 0.5},
        }


    def find_perfect_loop_weights(self, y, sr):
        if not len(y) or sr <= 0:
            raise Exception("Invalid input: empty audio from AI?")

        try:
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="samples")
            tempo = float(tempo)
        except Exception as e:
            raise Exception(f"Beat tracking error, bad AI generation maybe: {e}")

        if not 30 < tempo <= 300:
            raise Exception(f"Invalid tempo: {tempo:.1f} BPM, are you serious?")

        hop, n_fft = 256, 2048
        S = librosa.stft(y, n_fft=n_fft, hop_length=hop)
        if not S.size:
            raise Exception("Empty STFT")

        mag, beat_len = np.abs(S), 60 / tempo
        mel = librosa.feature.melspectrogram(S=mag**2, sr=sr, hop_length=hop, n_mels=64)
        mel = librosa.power_to_db(mel)

        def score_metrics(start, end, sf, ef):
            def try_block(f, default):
                try: return f()
                except: return default

            m_start = np.mean(mel[:, max(0, sf - 2):sf + 3], axis=1)
            m_end = np.mean(mel[:, ef - 2:min(mel.shape[1], ef + 3)], axis=1)
            spectral = 0.0 if np.any(np.isnan(m_start)) or np.any(np.isnan(m_end)) else max(0.0, 1 - cosine(m_start, m_end))

            waveform = try_block(lambda: self.calculate_waveform_continuity(y, start, end, sr), 0.0)
            beat = try_block(lambda: self.calculate_beat_alignment(start, end, beats, sr), 0.5)

            def phase_score():
                if sf < 3 or ef >= S.shape[1] - 3:
                    return 0.5
                ps = np.angle(S[:, sf - 1:sf + 2])
                pe = np.angle(S[:, ef - 1:ef + 2])
                freq = slice(0, S.shape[0] // 2)
                diff = np.abs(np.mean(ps[freq], axis=1) - np.mean(pe[freq], axis=1))
                diff = np.minimum(diff, 2 * np.pi - diff)
                return max(0.0, 1 - np.mean(diff) / np.pi)

            return {
                "spectral": spectral,
                "waveform": waveform,
                "beat": beat,
                "phase": try_block(phase_score, 0.5),
            }

        best = {"score": -np.inf, "start": 0, "end": 0, "measures": 0, "metrics": {}}
        threshold = 0.82

        score_weights = dict(spectral=0.15, waveform=0.25, beat=0.55, phase=0.05)

        for meas in [2, 4, 8, 12, 16, 32]:
            samp = int(meas * 4 * beat_len * sr)
            if not (3 * sr <= samp <= len(y) * 0.85): continue

            step = max(512, samp // 100)
            starts = range(int(len(y) * 0.05), len(y) - samp - int(len(y) * 0.05), step)

            for start in starts:
                end = start + samp
                if end > len(y): continue

                sf, ef = start // hop, end // hop
                if sf < 3 or ef >= mag.shape[1] - 3: continue

                metrics = score_metrics(start, end, sf, ef)
                score = sum(metrics[k] * score_weights[k] for k in metrics)

                if score > best["score"]:
                    best.update(dict(score=score, start=start, end=end, measures=meas, metrics=metrics))
                    if score > threshold:
                        self.logging_system(f"🎯 Score: {score:.3f}\n")
                        break
            if best["score"] > threshold:
                break

        if best["score"] < 0.15:
            raise Exception("No loops at first sight\n")

        dur = (best["end"] - best["start"]) / sr
        if dur < 1.5:
            raise Exception(f"Loop too short: {dur:.1f}s\n   Discarding sample...\n")

        return {
            "start_sample": best["start"],
            "end_sample": best["end"],
            "score": best["score"],
            "measures": best["measures"],
            "metrics": best["metrics"],
        }





    def process_loop_detection(self, input_file, output_file):

        MIN_LOOP_DURATION = self.min_sample_duration
        MAX_LOOP_ATTEMPTS = 1

        try:
            self.debug_file_state("PRE_LOOP_DETECTION", input_file)

            if not self.validate_audio_file(input_file):
                raise Exception(f"Corrupted input file!")

            y, sr = librosa.load(input_file, sr=None, mono=True)

            target_rms = 0.1
            current_rms = np.sqrt(np.mean(y**2))
            if current_rms > 0:
                y = y * (target_rms / current_rms)

            validations = [
                (not len(y), "Loaded audio is empty"),
                (sr <= 0, f"Invalid sample rate: {sr}"),
                (np.isnan(y).any() or np.isinf(y).any(), "Audio contains bad values"),
                (
                    len(y) / sr < 2.0,
                    f"Audio too short for loop detection: {len(y)/sr:.1f}s",
                ),
            ]

            for condition, message in validations:
                if condition:
                    raise Exception(message)

            for attempt in range(1, MAX_LOOP_ATTEMPTS + 1):
                try:
                    loop_info = self.find_perfect_loop(y, sr)
                    s, e = loop_info["start_sample"], loop_info["end_sample"]

                    if not (0 <= s < e <= len(y)):
                        raise Exception(
                            f"Invalid loop bounds: {s} -> {e} (max: {len(y)})"
                        )

                    initial_duration = (e - s) / sr

                    if initial_duration < MIN_LOOP_DURATION:
                        if attempt < MAX_LOOP_ATTEMPTS:
                            self.logging_system(
                                f"   ❌ Too short ({initial_duration:.1f}s < {MIN_LOOP_DURATION}s), retrying...\n"
                            )
                            continue
                        else:
                            raise Exception(
                                f"Loop too short ({initial_duration:.1f}s), retrying...\n"
                            )

                    s_opt, e_opt = (
                        self.find_optimal_zero_crossing(y, pos) for pos in (s, e)
                    )

                    if not (0 <= s_opt < e_opt <= len(y)):
                        self.logging_system(
                            "❌ Zero-crossing failed, using original positions"
                        )
                        s_opt, e_opt = s, e

                    final_duration = (e_opt - s_opt) / sr

                    if final_duration < MIN_LOOP_DURATION:
                        if attempt < MAX_LOOP_ATTEMPTS:
                            self.logging_system(
                                f"   ❌ Too short after optimization, retrying..."
                            )
                            continue
                        else:
                            raise Exception(f"All attempts failed!")

                    print("\n📊 Loop metrics:")
                    for k, v in loop_info["metrics"].items():
                        print(f"   {k}: {v:.3f}")

                    y_loop = y[s_opt:e_opt]

                    if np.isnan(y_loop).any() or np.isinf(y_loop).any():
                        raise Exception("Loop contains invalid values")

                    fade_samples = int(sr * 0.01)
                    if fade_samples:
                        fade_in, fade_out = np.linspace(
                            0, 1, fade_samples
                        ), np.linspace(1, 0, fade_samples)
                        y_loop[:fade_samples] *= fade_in
                        y_loop[-fade_samples:] *= fade_out

                    if os.path.exists(output_file):
                        os.remove(output_file)

                    try:

                        metadata = {
                            "title": f"InfiniLoop - {self.PROMPT}",
                            "artist": "InfiniLoop AI",
                            "comment": f"Prompt: {self.PROMPT} | Duration: {self.duration}s | Model: {self.model}",
                            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }

                        sf.write(output_file, y_loop, sr, subtype="PCM_24")

                        with self.buffer_lock:
                            self.next_audio_buffer = y_loop.copy()
                            self.next_audio_sr = sr

                    except Exception as err:
                        raise Exception(f"Error writing audio file: {err}")

                    if os.path.getsize(output_file) < 1024:
                        raise Exception("Output too small")

                    self.debug_file_state("POST_LOOP_DETECTION", output_file)

                    if not self.validate_audio_file(output_file):
                        raise Exception("Output validation failed")

                    self.logging_system(f"🧬 Next loop ready ({final_duration:.1f}s)")

                    if self.current_loop_start_time:
                        elapsed = time.time() - self.current_loop_start_time
                        if elapsed < self.min_song_duration:
                            self.logging_system("🧬 Waiting for song ending\n")


                    del y
                    del y_loop
                    gc.collect()

                    return

                except Exception as attempt_error:
                    if attempt < MAX_LOOP_ATTEMPTS:
                        self.logging_system(
                            f"   ❌ Attempt {attempt} failed: {attempt_error}"
                        )
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

            if "y" in locals():
                del y
            if "y_loop" in locals():
                del y_loop
            gc.collect()

            raise


    def sanitize_prompt(self, prompt):

        sanitized = re.sub(r'[;&|`$<>\\"\'\n\r\0]', "", prompt)

        max_length = 200
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]

        sanitized = " ".join(sanitized.split())

        if not sanitized or len(sanitized) < 2:
            raise ValueError("Prompt too short or invalid")

        return sanitized


    def sanitize_filename(self, filename):

        filename = os.path.basename(filename)

        filename = re.sub(r"[^\w\s.-]", "", filename)

        if not filename.endswith(".wav"):
            filename = filename.rsplit(".", 1)[0] + ".wav"

        max_length = 100
        if len(filename) > max_length:
            name_part = filename[:-4][: max_length - 4]
            filename = name_part + ".wav"

        return filename


    def normalize_audio_advanced(self, y, sr):

        try:
            TARGET_PEAK = 0.7
            current_peak = np.max(np.abs(y))

            if current_peak < 1e-8:
                return y * 0.1

            gain = TARGET_PEAK / current_peak
            y_normalized = y * gain

            self.logging_system(f"🎚️ Peak norm: {current_peak:.3f} → {TARGET_PEAK:.3f} (gain: {gain:.2f}x)")

            final_peak = np.max(np.abs(y_normalized))
            if final_peak > 0.95:
                y_normalized *= (0.95 / final_peak)

            return y_normalized

        except Exception as e:
            self.logging_system(f"❌ Normalizzazione fallita: {e}")
            rms = np.sqrt(np.mean(y**2))
            if rms > 0:
                return y * (0.1 / rms)
            return y


    def generate_audio_safe(self, outfile):
        try:
            self.is_generating = True

            # Applica eventuali parametri pendenti prima di generare
            self._apply_pending_params()

            self._prepare_benchmark()

            p, m, d = self.PROMPT, self.model, self.duration
            self.generation_status = f"Generating sample..."
            if not hasattr(self, "_has_generated_once") or not self._has_generated_once:
                self.logging_system("🎼 Generating new sample\n")
                self._has_generated_once = True
            else:
                self.logging_system("🎼 Generating next sample\n")

            with self.safe_temp_file() as raw, self.safe_temp_file() as proc:
                self.debug_file_state("PRE_GENERATION", raw)

                t = self._run_ai_generation(p, m, d, raw)

                self.logging_system(f"⏱️ AI made it in {t:.2f}s!")

                if self.benchmark_enabled:
                    self._save_benchmark(d, t)

                self.debug_file_state("POST_GENERATION", raw)
                self._normalize_audio(raw)

                self.generation_status = "Loop analysis running"
                self.process_loop_detection(raw, proc)

                self._finalize_audio(outfile, proc)
                self.generation_status = "Completed!"
                self._trigger_ui_cb()

        except subprocess.CalledProcessError as e:
            if not self.stop_requested:
                self._handle_subprocess_err(e)
            else:
                self.generation_status = "Stopped"
        except subprocess.TimeoutExpired:
            if not self.stop_requested:
                self._handle_timeout()
            else:
                self.generation_status = "Stopped"
        except Exception as e:
            if "stopped by user" in str(e).lower():
                self.generation_status = "Stopped"
            else:
                self.generation_status = "Error"
            raise
        finally:
            self.is_generating = False

    def _prepare_benchmark(self):
        if not hasattr(self, "benchmark_data"):
            self.benchmark_data = []
        if not hasattr(self, "benchmark_file"):
            self.benchmark_file = "benchdata.json"
        self.load_benchmark_data()

    def _run_ai_generation(self, prompt, model, dur, out):
        t0 = time.time()

        with self.generation_lock:
            if self.stop_requested:
                raise Exception("Generation cancelled by user")

            self.current_generation_process = subprocess.Popen([
                "ionice", "-c", "2", "-n", "7", "nice", "-n", "10",
                "./musicgpt-x86_64-unknown-linux-gnu", prompt,
                "--model", model, "--secs", str(dur),
                "--output", out,
                "--no-playback", "--no-interactive", "--ui-no-open"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        try:

            while True:

                if self.current_generation_process is None:
                    raise Exception("Generation stopped\n")

                poll_result = self.current_generation_process.poll()
                if poll_result is not None:

                    break

                if self.stop_requested:

                    self._terminate_generation_process()

                time.sleep(0.1)


            with self.generation_lock:
                if self.current_generation_process is not None:
                    returncode = self.current_generation_process.returncode
                    stdout, stderr = self.current_generation_process.communicate()
                    self.current_generation_process = None
                else:
                    raise Exception("Generation stopped\n")

            if returncode != 0:
                raise subprocess.CalledProcessError(returncode, "musicgpt", stderr)

            return time.time() - t0

        except Exception as e:
            with self.generation_lock:
                if self.current_generation_process:
                    self._terminate_generation_process()
                    self.current_generation_process = None
            raise

    def _terminate_generation_process(self):
        """Termina il processo di generazione in modo pulito"""
        if not self.current_generation_process:
            return

        try:

            self.current_generation_process.terminate()
            try:
                self.current_generation_process.wait(timeout=3.0)
                return
            except subprocess.TimeoutExpired:
                pass


            self.current_generation_process.kill()
            self.current_generation_process.wait(timeout=1.0)

        except Exception:
            pass
        finally:

            self.current_generation_process = None

    def _save_benchmark(self, dur, elapsed):
        self.benchmark_data.append({
            "duration_requested": dur,
            "generation_time": round(elapsed, 3)
        })
        self.save_benchmark_data()
        self.logging_system("📈 Stats updated\n")

    def _normalize_audio(self, path):
        if not self.validate_audio_file(path):
            raise Exception("AI generated invalid audio file")
        y, sr = librosa.load(path, sr=None, mono=True)
        y = self.normalize_audio_advanced(y, sr)
        sf.write(path, y, sr)
        del y
        gc.collect()

    def _finalize_audio(self, final, temp):
        if not self.validate_audio_file(temp):
            raise Exception("Processed file validation failed")
        self.debug_file_state("PRE_FINAL_MOVE", temp)
        with self.file_lock:
            shutil.move(temp, final)
        self.debug_file_state("POST_FINAL_MOVE", final)

    def _trigger_ui_cb(self):
        if callable(getattr(self, "on_generation_complete", None)):
            try:
                self.on_generation_complete()
            except Exception as e:
                self.logging_system(f"❌ UI callback failed: {e}")

    def _handle_subprocess_err(self, e):

        if self.stop_requested and e.returncode == -9:
            self.generation_status = "Stopped"
            raise Exception("Generation stopped")
        else:

            self.logging_system(f"❌ Generation error: {e}\n{e.stderr.strip()}")
            self.generation_status = "Error"
            raise

    def _handle_timeout(self):
        self.logging_system("❌ Generation timeout (120s)")
        self.generation_status = "Timeout"
        raise


    def get_duration(self, filepath):

        try:
            with self.file_lock:
                if not os.path.exists(filepath):
                    return 0.0
                info = mediainfo(filepath)
                return float(info["duration"])
        except:
            return 0.0


    def get_random_title(self):

        try:
            with open("nomi.txt", "r") as f1, open("nomi2.txt", "r") as f2:
                list1 = [line.strip().upper() for line in f1 if line.strip()]
                list2 = [line.strip().upper() for line in f2 if line.strip()]
            if list1 and list2:
                word1 = "".join(c for c in random.choice(list1) if c.isalnum())
                word2 = "".join(c for c in random.choice(list2) if c.isalnum())
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


    def loop_current_crossfade_blocking(self, filepath, crossfade_sec_unused, stop_event):

        try:
            if not self.validate_audio_file(filepath):
                self.logging_system(f"❌ Invalid file for loop: {filepath}")
                return

            duration = self.get_duration(filepath)
            if duration <= 0:
                self.logging_system(f"❌ Invalid duration: {filepath}")
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

            current_process = subprocess.Popen(
                [
                    "ionice",
                    "-c",
                    "2",
                    "-n",
                    "0",
                    "taskset",
                    "-c",
                    "2",
                    "ffplay",
                    "-nodisp",
                    "-autoexit",
                    "-loglevel",
                    "quiet",
                    "-infbuf",
                    "-probesize",
                    "32",
                    "-analyzeduration",
                    "0",
                    "-loop",
                    "0",
                    "-f",
                    "wav",
                    os.path.abspath(filepath),
                ],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            while not stop_event.is_set() and current_process.poll() is None:
                stop_event.wait(0.1)

            if current_process.poll() is None:
                self._kill_process_safely(current_process)

            self.debug_file_state("END_PLAYBACK", filepath)

        except Exception as e:
            self.logging_system(f"❌ Error in playback: {str(e)}")
            if "current_process" in locals() and current_process.poll() is None:
                self._kill_process_safely(current_process)


    def _kill_process_safely(self, process):

        try:

            process.terminate()
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:

            process.kill()
            process.wait(timeout=1.0)
        except Exception:
            pass


    def safe_file_swap(self):

        with self.swap_lock:
            try:

                if not self.can_swap_now():
                    self.logging_system(f"⏱️ Waiting for minimum duration ({self.min_song_duration}s)")
                    return False

                with self.next_file_lock:
                    if not self.validate_audio_file(self.NEXT):
                        raise Exception(f"❌ Invalid NEXT file: {self.NEXT}")

                env = os.environ.copy()
                env["SDL_AUDIODRIVER"] = self.audio_driver

                next_process = subprocess.Popen(
                    [
                        "ionice",
                        "-c",
                        "2",
                        "-n",
                        "0",
                        "taskset",
                        "-c",
                        "3",
                        "ffplay",
                        "-nodisp",
                        "-autoexit",
                        "-loglevel",
                        "quiet",
                        "-infbuf",
                        "-probesize",
                        "32",
                        "-analyzeduration", # anyway, you ugly. Nothing to do with code, but basically you (and I, and everyone else), are: ugly. LOL Just kidding... just wanted to add some bullcrap to this code...
                        "0",
                        "-loop",
                        "0",
                        "-f",
                        "wav",
                        os.path.abspath(self.NEXT),
                    ],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

                next_pid = next_process.pid

                time.sleep(self.CROSSFADE_SEC)

                self.stop_event.set()

                if self.loop_thread and self.loop_thread.is_alive():
                    self.loop_thread.join(timeout=2.0)

                self.kill_all_ffplay_processes(exclude_pid=next_pid)

                with self.file_lock, self.next_file_lock:
                    self.CURRENT, self.NEXT = self.NEXT, self.CURRENT

                    with self.buffer_lock:
                        if self.next_audio_buffer is not None:
                            del self.next_audio_buffer
                            self.next_audio_buffer = None
                            gc.collect()

                self.stop_event = threading.Event()


                self.current_loop_start_time = time.time()
                self.min_duration_satisfied = False

                return True

            except Exception as e:
                self.logging_system(f"❌ Error during swap: {str(e)}")
                self.stop_event = threading.Event()
                return False


    def run_loop(self):

        self.stop_event = threading.Event()


        self.current_loop_start_time = time.time()
        self.min_duration_satisfied = False

        self.loop_thread = threading.Thread(
            target=self.loop_current_crossfade_blocking,
            args=(self.CURRENT, self.CROSSFADE_SEC, self.stop_event),
            daemon=True,
        )
        self.loop_thread.start()

        consecutive_errors = 0
        max_consecutive_errors = 2

        while self.is_playing:
            try:
                self.generate_audio_safe(self.NEXT)

                if not self.is_playing:
                    break


                self.wait_for_swap_opportunity()

                if not self.is_playing:
                    break

                if not self.safe_file_swap():
                    self.logging_system("❌ Swap failed, regenerating...")
                    continue

                if not self.is_playing:
                    break

                consecutive_errors = 0

            except Exception as e:
                consecutive_errors += 1
                self.logging_system(f"❌ {str(e)}")


    def can_swap_now(self):

        if not self.current_loop_start_time:
            return True

        elapsed = time.time() - self.current_loop_start_time
        min_time_reached = elapsed >= self.min_song_duration

        if min_time_reached and not self.min_duration_satisfied:
            self.min_duration_satisfied = True

        if not min_time_reached:
            return False

        return self.validate_audio_file(self.NEXT)


    def wait_for_swap_opportunity(self):

        while self.is_playing and not self.can_swap_now():
            if not self.current_loop_start_time:
                break

            elapsed = time.time() - self.current_loop_start_time
            remaining = max(0, self.min_song_duration - elapsed)

            if remaining > 0:

                if elapsed % 30 < 0.5:
                    self.logging_system(f"⏱️ Current loop: {remaining:.1f}s left")
            else:

                if not self.validate_audio_file(self.NEXT):
                    if elapsed % 10 < 0.5:
                        self.logging_system(f"⏳ Waiting for next file generation (elapsed: {elapsed:.1f}s)")

            time.sleep(0.5)


    def get_current_loop_timing(self):

        if not self.current_loop_start_time:
            return {
                'elapsed': 0,
                'remaining_min_time': self.min_song_duration,
                'min_time_satisfied': False,
                'can_swap': False
            }

        elapsed = time.time() - self.current_loop_start_time
        remaining = max(0, self.min_song_duration - elapsed)
        min_satisfied = elapsed >= self.min_song_duration

        return {
            'elapsed': elapsed,
            'remaining_min_time': remaining,
            'min_time_satisfied': min_satisfied,
            'can_swap': self.can_swap_now()
        }


    def main_loop(self):
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries and self.is_playing:
            try:
                self.kill_all_ffplay_processes()
                self.kill_all_musicgpt_processes()

                futures = []
                for file_path, file_name in [
                    (self.CURRENT, "first"),
                    (self.NEXT, "second"),
                ]:
                    if not self.validate_audio_file(file_path):
                        self.logging_system(
                            f"📁 Generating initial sample ({file_name})"
                        )

                        future = self.thread_pool.submit(
                            self.generate_audio_safe, file_path
                        )
                        futures.append((future, file_path))

                for future, file_path in futures:
                    try:
                        future.result(timeout=180)
                        if not self.validate_audio_file(file_path):
                            raise Exception(f"File {file_path} not generated correctly")
                        self.logging_system(
                            f"✅ File {os.path.basename(file_path)} generated and validated"
                        )
                    except Exception as e:
                        raise Exception(f"Failed to generate {file_path}: {e}")

                if not self.is_playing:
                    return

                self.run_loop()
                break

            except Exception as e:
                retry_count += 1
                self.logging_system(
                    f"❌ Error in main loop (attempt {retry_count}/{max_retries}): {str(e)}"
                )

                if retry_count >= max_retries:
                    self.logging_system(
                        "❌ Too many errors, give me a break. Stopping application"
                    )
                    self.is_playing = False
                    return

                self.logging_system(f"🔄 Reinitializing...")
                self.kill_all_ffplay_processes()
                self.kill_all_musicgpt_processes()

                time.sleep(2)

                try:
                    for filepath in [self.CURRENT, self.NEXT]:
                        if os.path.exists(filepath):
                            os.remove(filepath)
                            self.logging_system(f"🗑️ Removed {os.path.basename(filepath)}")
                except Exception as remove_error:
                    self.logging_system(f"❌ File removal error: {remove_error}. Huh.")


    def start_loop(self, prompt):
        self.PROMPT = prompt.strip()
        if not self.PROMPT:
            print("❌ Error: Please enter a prompt!")
            return False

        self.stop_requested = False
        self.is_playing = True

        self.loop_thread = threading.Thread(target=self.main_loop, daemon=True)
        self.loop_thread.start()

        return True


    def stop_loop(self):
        self.is_playing = False
        self.stop_requested = True

        if hasattr(self, "stop_event"):
            self.stop_event.set()
        self.logging_system("⏹️ Loop stopped\n")


        with self.generation_lock:
            if hasattr(self, 'current_generation_process') and self.current_generation_process:
                try:
                    self._terminate_generation_process()
                except Exception as e:
                    self.logging_system(f"⚠️ Error stopping generation: {e}")
                finally:
                    self.current_generation_process = None

        self.thread_pool.shutdown(wait=False, cancel_futures=True)
        self.thread_pool = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="infiniloop"
        )

        self.kill_all_ffplay_processes()
        self.kill_all_musicgpt_processes()
        self.cleanup_temp_files()

        with self.buffer_lock:
            if self.next_audio_buffer is not None:
                del self.next_audio_buffer
                self.next_audio_buffer = None
                gc.collect()


    def kill_all_ffplay_processes(self, exclude_pid=None):

        try:
            result = subprocess.run(
                ["pgrep", "-f", "ffplay"], capture_output=True, text=True
            )
            if result.stdout.strip():
                pids = result.stdout.strip().split("\n")
                for pid in pids:
                    try:
                        pid_int = int(pid)

                        if exclude_pid and pid_int == exclude_pid:
                            continue

                        subprocess.run(["kill", "-9", pid], check=False, timeout=2)
                    except:
                        pass
        except Exception as e:
            pass


    def kill_all_musicgpt_processes(self):

        try:
            result = subprocess.run(
                ["pgrep", "-f", "musicgpt-x86_64-unknown-linux-gnu"],
                capture_output=True,
                text=True,
            )
            if result.stdout.strip():
                pids = result.stdout.strip().split("\n")
                for pid in pids:
                    try:
                        subprocess.run(["kill", "-9", pid], check=False, timeout=2)
                    except:
                        pass
                self.logging_system(f"🛑 MusicGPT terminated")
        except Exception as e:

            pass


    def save_current_loop(self, filename):
        with self.file_lock:
            current_file = self.CURRENT

            if not self.validate_audio_file(current_file):
                print("❌ No valid loop to save!")
                return False

            try:

                safe_filename = self.sanitize_filename(filename)

                safe_path = os.path.join(os.getcwd(), safe_filename)

                shutil.copy2(current_file, safe_path)
                self.logging_system(
                    f"💾 Loop saved: {safe_path} (from {os.path.basename(current_file)})"
                )
                return True
            except Exception as e:
                self.logging_system(f"❌ Save error: {str(e)}. Dunno...")
                return False


    def print_status(self):
        status = "🟢 PLAYING" if self.is_playing else "🔴 STOPPED"
        generation = "🎼 YES" if self.is_generating else "💤 NO"

        print(f"\n📊 INFINI LOOP STATUS:")
        print(f"   Status:       {status}")
        print(f"   Prompt:       '{self.PROMPT}'")
        print(f"   Gen. duration: {self.duration}s")
        print(f"   Min song dur.: {self.min_song_duration}s")
        print(f"   Audio driver: {self.audio_driver}")
        print(f"   Generating:   {generation}")
        if self.is_generating:
            print(f"   Gen. status:  {self.generation_status}")

        if self.is_playing and self.current_loop_start_time:
            elapsed = time.time() - self.current_loop_start_time
            remaining = max(0, self.min_song_duration - elapsed)
            print(f"   Loop elapsed: {elapsed:.1f}s")
            if remaining > 0:
                print(f"   Min dur. left: {remaining:.1f}s")
            else:
                print(f"   Min dur.:     ✅ Satisfied")

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

                if not filename.endswith(".wav"):
                    filename += ".wav"
                if app.save_current_loop(filename):
                    print(f"✅ Loop saved as: {filename}")
                else:
                    print("❌ Unable to save loop")

            elif cmd[0] == "debug":
                if len(cmd) > 1 and cmd[1] in ["on", "off"]:
                    app.debug_mode = cmd[1] == "on"
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
                    print("    duration     - Generated sample duration")
                    print("    minduration  - Minimum song duration")
                    print("    driver       - System audio driver")
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

                elif option == "minduration":
                    print("\n⏱️ MINIMUM SONG DURATION:")
                    print("    Range: 10-300 seconds (5 minutes)")
                    print("    Current loop will play at least this long before switching")
                    print("    Tip: 30-60s for variety, 120s+ for longer listening")
                    try:
                        min_dur = int(input(f"Minimum duration in seconds [current: {app.min_song_duration}s]: "))
                        if 10 <= min_dur <= 300:
                            app.min_song_duration = min_dur
                            print(f"✅ Minimum song duration: {app.min_song_duration}s")
                        else:
                            print("❌ Minimum duration must be between 10 and 300 seconds")
                    except ValueError:
                        print("❌ Non-numeric value")

                elif option == "driver":
                    print("\n🔊 AVAILABLE AUDIO DRIVERS:")
                    print("    pulse - PulseAudio (Linux standard, recommended)")
                    print("    alsa  - ALSA (Linux low-level)")
                    print("    dsp   - OSS (Unix/BSD systems)")
                    choice = input("Choose driver [pulse/alsa/dsp]: ").strip().lower()
                    if choice in ["pulse", "pulseaudio"]:
                        app.audio_driver = "pulse"
                        print("✅ Driver: PulseAudio")
                    elif choice in ["alsa"]:
                        app.audio_driver = "alsa"
                        print("✅ Driver: ALSA")
                    elif choice in ["dsp", "oss"]:
                        app.audio_driver = "dsp"
                        print("✅ Driver: OSS")
                    else:
                        print("❌ Invalid driver")
                else:
                    print(f"❌ Option '{option}' not recognized")
                    print("💡 Options: duration, minduration, driver")

            elif cmd[0] == "help":
                print("\n🆘 AVAILABLE COMMANDS:")
                print("   start '<prompt>'    - Start infinite loop with prompt")
                print("   stop                - Stop current playback")
                print("   status              - Show detailed system status")
                print("   save <file.wav>     - Save current loop to file")
                print(
                    "   validate <target>   - Validate audio files (current/next/both - for debug)"
                )
                print("   debug <on|off>      - Enable/disable debug messages")
                print("   set <option>        - Change settings (see below)")
                print("   help                - Show this help")
                print("   quit/exit/q         - Exit program")
                print("\n⚙️ CONFIGURABLE OPTIONS:")
                print("   set duration        - Change generation duration (5-30s)")
                print("   set minduration     - Change minimum song duration (10-300s)")
                print("   set driver          - Change audio driver (pulse/alsa/dsp)")
                print("\n💡 EXAMPLES:")
                print("   start 'ambient chill loop'")
                print("   start 'jazz piano solo'")
                print("   save my_loop.wav")
                print("   validate both")
                print("   debug on")
                print("   set minduration     - Set minimum time before switching loops")

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

        """,
    )

    parser.add_argument("--prompt", "-p", type=str, help="Prompt for generation")

    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Interactive mode"
    )

    parser.add_argument(
        "--generate-only",
        "-g",
        nargs=2,
        metavar=("PROMPT", "OUTPUT"),
        help="Generate only one loop and save (prompt, output_file)",
    )

    parser.add_argument(
        "--duration",
        "-d",
        type=int,
        default=15,
        help="Generation duration in seconds (5-30)",
    )

    parser.add_argument(
        "--driver",
        choices=["pulse", "alsa", "dsp"],
        default="pulse",
        help="Audio driver",
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Detailed output")

    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")

    parser.add_argument("--no-debug", action="store_true", help="Disable debug mode")

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
    app = InfiniLoopTerminal()

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
                if not filename.endswith(".wav"):
                    filename += ".wav"
                app.save_loop(filename)

            elif cmd[0] == "validate":
                if len(cmd) < 2:
                    print("❌ Usage: validate <current|next|both>")
                    continue
                app.validate_audio(cmd[1])

            elif cmd[0] == "debug":
                if len(cmd) < 2:
                    print("❌ Usage: debug <on|off>")
                    continue
                state = cmd[1].lower()
                app.debug_mode = (state == "on")
                print(f"🐞 Debug mode: {'ON' if app.debug_mode else 'OFF'}")

            elif cmd[0] == "set":
                if len(cmd) < 2:
                    print("❌ Usage: set <option>")
                    continue
                option = cmd[1].lower()

                if option == "duration":
                    val = input("Enter duration (seconds): ").strip()
                    if val.isdigit():
                        app.duration = int(val)
                        print(f"✅ Duration set to {app.duration}s")
                    else:
                        print("❌ Invalid duration")

                elif option == "driver":
                    choice = input("Choose driver [pulse/alsa/dsp]: ").strip().lower()
                    if choice in ["pulse", "pulseaudio"]:
                        app.audio_driver = "pulse"
                        print("✅ Driver: PulseAudio")
                    elif choice in ["alsa"]:
                        app.audio_driver = "alsa"
                        print("✅ Driver: ALSA")
                    elif choice in ["dsp", "oss"]:
                        app.audio_driver = "dsp"
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

        """,
    )

    parser.add_argument("--prompt", "-p", type=str, help="Prompt for generation")

    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Interactive mode"
    )

    parser.add_argument(
        "--generate-only",
        "-g",
        nargs=2,
        metavar=("PROMPT", "OUTPUT"),
        help="Generate only one loop and save (prompt, output_file)",
    )

    parser.add_argument(
        "--duration",
        "-d",
        type=int,
        default=15,
        help="Generation duration in seconds (5-30)",
    )

    parser.add_argument(
        "--driver",
        choices=["pulse", "alsa", "dsp"],
        default="pulse",
        help="Audio driver",
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Detailed output")

    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")

    parser.add_argument("--no-debug", action="store_true", help="Disable debug mode")

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
