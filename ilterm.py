#!/usr/bin/env python3
"""
INFINI LOOP TERMINAL - Lightweight Version - FIXED
Infinite AI Music Generation with Advanced Loop Detection
Terminal-only version by gat - Race condition and corruption fixes applied
"""

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
        # File paths - use absolute paths to avoid confusion
        self.base_dir = os.path.abspath(".")
        self.FILE1 = os.path.join(self.base_dir, "music1.wav")
        self.FILE2 = os.path.join(self.base_dir, "music2.wav")
        self.CURRENT = self.FILE1
        self.NEXT = self.FILE2

        # Fixed settings (no longer configurable)
        self.CROSSFADE_MS = 1
        self.CROSSFADE_SEC = self.CROSSFADE_MS / 1000.0
        self.PROMPT = ""
        self.model = "medium"  # Fixed to medium
        self.duration = 15
        self.audio_driver = "pulse"

        # Threading
        self.is_playing = False
        self.stop_event = threading.Event()
        self.loop_thread = None
        self.generation_thread = None
        self.is_generating = False
        self.generation_status = "Inattivo"

        # Lock for file operations - now more granular
        self.file_lock = threading.Lock()
        self.swap_lock = threading.Lock()  # Dedicated lock for file swapping

        # Temporary directory for safe operations
        self.temp_dir = tempfile.mkdtemp(prefix="ilterm_")

        # Debug mode
        self.debug_mode = False

        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        os.system("cls" if os.name == "nt" else "clear")

        print("\n🎵 INFINI LOOP TERMINAL - by gat\n")
        print("✅ Inizializzazione completata!\n")

    def __del__(self):
        """Cleanup on destruction"""
        try:
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
        except:
            pass

    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        print("\n🛑 Interruzione rilevata, arresto in corso...")
        self.stop_loop()
        # Ensure all processes are killed
        self.kill_all_ffplay_processes()
        self.kill_all_musicgpt_processes()
        # Cleanup temp directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        sys.exit(0)

    def log_message(self, message):
        """Print timestamped log messages"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n\n[{timestamp}] {message}")

    def debug_file_state(self, operation, filepath):
        """Debug dello stato del file"""
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
        """Validazione completa del file audio"""
        try:
            # Test 1: File esiste e ha dimensione minima
            if not os.path.exists(filepath):
                return False

            size = os.path.getsize(filepath)
            if size < 1024:
                return False

            # Test 2: File è leggibile da soundfile
            with sf.SoundFile(filepath) as sf_test:
                if sf_test.frames == 0:
                    return False

            # Test 3: File è leggibile da librosa (test più approfondito)
            try:
                y, sr = librosa.load(filepath, sr=None, mono=True, duration=1.0)  # Test solo primo secondo
                if len(y) == 0 or sr == 0:
                    return False

                # Test 4: Non ci sono valori NaN o infiniti
                if np.isnan(y).any() or np.isinf(y).any():
                    return False

            except Exception:
                return False

            return True

        except Exception:
            return False

    @contextlib.contextmanager
    def safe_temp_file(self, suffix='.wav'):
        """Context manager per file temporanei sicuri"""
        temp_path = None
        try:
            fd, temp_path = tempfile.mkstemp(suffix=suffix, dir=self.temp_dir)
            os.close(fd)  # Close file descriptor, keep path
            yield temp_path
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass

    def find_optimal_zero_crossing(self, y, sample, window_size=256):
        """Find optimal zero crossing point near given sample"""
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
        """Calculate waveform continuity between loop points"""
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

        # Correlation
        correlation = np.corrcoef(end_segment, start_segment)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0

        # RMS similarity
        rms_diff = np.sqrt(np.mean((end_segment - start_segment) ** 2))
        max_rms = max(np.sqrt(np.mean(end_segment ** 2)), np.sqrt(np.mean(start_segment ** 2)))

        if max_rms > 1e-8:
            rms_similarity = 1 - min(1.0, rms_diff / max_rms)
        else:
            rms_similarity = 1.0

        # Derivative continuity
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
        """Calculate beat alignment score"""
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
        """Calculate phase continuity between loop points"""
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

    def find_perfect_loop_simple(self, y, sr):
        """Algoritmo semplice e robusto basato su quello originale"""
        self.log_message("🧠 Algoritmo di loop detection semplice...")

        # Simple mel-spectrogram approach (like original)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512)
        S_dB = librosa.power_to_db(S, ref=np.max)

        min_duration_sec = 5  # Minimum 5 seconds instead of 7
        max_duration_sec = min(25, len(y) / sr * 0.8)  # Max 25s or 80% of file
        min_frames = int(min_duration_sec * sr / 512)
        max_frames = int(max_duration_sec * sr / 512)

        best_score = -np.inf
        best_start = 0
        best_end = 0

        # Simple similarity search
        for i in range(0, min(S_dB.shape[1] - min_frames, S_dB.shape[1] // 2)):
            for j in range(i + min_frames, min(S_dB.shape[1], i + max_frames)):
                if j >= S_dB.shape[1]:
                    break

                segment1 = S_dB[:, i]
                segment2 = S_dB[:, j]

                # Compute cosine similarity
                norm1 = np.linalg.norm(segment1)
                norm2 = np.linalg.norm(segment2)

                if norm1 > 1e-8 and norm2 > 1e-8:
                    score = np.dot(segment1, segment2) / (norm1 * norm2)
                    if score > best_score:
                        best_score = score
                        best_start = i
                        best_end = j

        if best_score < 0.1:  # Very low threshold
            raise Exception(f"Nessuna similarità trovata (best score: {best_score:.3f})")

        # Convert to samples
        start_sample = best_start * 512
        end_sample = best_end * 512

        # Validate bounds
        if start_sample >= end_sample or end_sample > len(y):
            raise Exception(f"Bounds non validi: {start_sample} -> {end_sample}")

        duration = (end_sample - start_sample) / sr
        self.log_message(f"✅ Loop semplice trovato! Score: {best_score:.3f}, Durata: {duration:.1f}s")

        return {
            'start_sample': start_sample,
            'end_sample': end_sample,
            'score': best_score,
            'measures': int(duration / 2),  # Estimate
            'metrics': {'Spettrale': best_score, 'Waveform': 0.5, 'Beat Align': 0.5, 'Fase': 0.5}
        }

    def find_perfect_loop(self, y, sr):
        """Advanced multi-metric loop detection algorithm - WITH FALLBACK"""
        try:
            # Try advanced algorithm first
            return self.find_perfect_loop_advanced(y, sr)
        except Exception as e:
            self.log_message(f"⚠️ Algoritmo avanzato fallito: {e}")
            self.log_message("🔄 Uso algoritmo semplice di fallback...")
            # Fallback to simple algorithm
            return self.find_perfect_loop_simple(y, sr)

    def find_perfect_loop_advanced(self, y, sr):
        """Advanced multi-metric loop detection algorithm - ORIGINAL COMPLEX VERSION"""
        self.log_message("🧠 Analisi avanzata multi-metrica in corso...")

        # Validate input
        if len(y) == 0:
            raise Exception("Audio input vuoto per loop detection")
        if sr <= 0:
            raise Exception(f"Sample rate non valido: {sr}")

        # Beat tracking with error handling
        try:
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='samples')
        except Exception as e:
            raise Exception(f"Errore beat tracking: {e}")

        if isinstance(tempo, np.ndarray):
            tempo = tempo.item()

        # More lenient tempo validation
        if tempo <= 30 or tempo > 300:  # Wider range
            self.log_message(f"⚠️ Tempo sospetto: {tempo} BPM, uso algoritmo semplice")
            raise Exception("Tempo non valido, serve fallback")

        # STFT for spectral analysis
        hop_length = 512
        try:
            S = librosa.stft(y, n_fft=2048, hop_length=hop_length)
            S_mag = np.abs(S)
        except Exception as e:
            raise Exception(f"Errore STFT: {e}")

        if S_mag.size == 0:
            raise Exception("STFT risulta vuoto")

        beat_length = 60.0 / tempo
        possible_measures = [2, 4, 8]  # Added 2 measures option

        best_score = -np.inf
        best_start = 0
        best_end = 0
        best_measure_count = 4
        best_metrics = {}

        valid_loops_found = 0

        for measures in possible_measures:
            target_beats = measures * 4
            target_duration = target_beats * beat_length
            target_samples = int(target_duration * sr)

            # More lenient bounds checking
            min_duration = 3.0  # Reduced from 2.0
            max_duration = len(y) * 0.9  # Increased from 0.7

            if target_samples < min_duration * sr or target_samples > max_duration:
                self.log_message(f"⚠️ Durata target {target_duration:.1f}s fuori range, skip {measures} misure")
                continue

            search_start = int(len(y) * 0.05)  # Reduced from 0.1
            search_end = len(y) - target_samples - int(len(y) * 0.05)

            if search_end <= search_start:
                self.log_message(f"⚠️ Range di ricerca non valido per {measures} misure")
                continue

            step_size = hop_length * 4  # Larger steps for efficiency

            progress_counter = 0
            total_steps = (search_end - search_start) // step_size

            for start_sample in range(search_start, search_end, step_size):
                end_sample = start_sample + target_samples

                # Strict bounds checking
                if end_sample >= len(y) or start_sample < 0:
                    break

                # Ensure we have enough data for analysis
                if end_sample - start_sample < sr * 0.5:  # At least 0.5 seconds
                    continue

                # Progress indicator (less frequent)
                progress_counter += 1
                if progress_counter % 100 == 0:
                    progress = (progress_counter / total_steps) * 100

                start_frame = start_sample // hop_length
                end_frame = end_sample // hop_length

                if end_frame >= S_mag.shape[1] or start_frame < 0:
                    continue

                # Spectral similarity with bounds checking
                window = 3  # Reduced window
                if start_frame < window or end_frame >= S_mag.shape[1] - window:
                    continue

                start_spectrum = np.mean(S_mag[:, max(0, start_frame-window):start_frame+window], axis=1)
                end_spectrum = np.mean(S_mag[:, max(0, end_frame-window):min(S_mag.shape[1], end_frame+window)], axis=1)

                # Validate spectra
                if len(start_spectrum) == 0 or len(end_spectrum) == 0:
                    continue
                if np.isnan(start_spectrum).any() or np.isnan(end_spectrum).any():
                    continue

                if np.linalg.norm(start_spectrum) > 1e-8 and np.linalg.norm(end_spectrum) > 1e-8:
                    try:
                        spectral_similarity = 1 - cosine(start_spectrum, end_spectrum)
                        if np.isnan(spectral_similarity):
                            spectral_similarity = 0.0
                    except:
                        spectral_similarity = 0.0
                else:
                    spectral_similarity = 0.0

                # Other metrics with error handling (simplified)
                try:
                    waveform_continuity = self.calculate_waveform_continuity(y, start_sample, end_sample, sr)
                    if np.isnan(waveform_continuity):
                        waveform_continuity = 0.0
                except:
                    waveform_continuity = 0.5  # Default value

                try:
                    beat_alignment = self.calculate_beat_alignment(start_sample, end_sample, beats, sr)
                    if np.isnan(beat_alignment):
                        beat_alignment = 0.0
                except:
                    beat_alignment = 0.5  # Default value

                try:
                    phase_continuity = self.calculate_phase_continuity(S, start_frame, end_frame)
                    if np.isnan(phase_continuity):
                        phase_continuity = 0.0
                except:
                    phase_continuity = 0.5  # Default value

                # Combined score (more weight on spectral)
                combined_score = (
                    spectral_similarity * 0.5 +  # Increased weight
                    waveform_continuity * 0.25 +  # Reduced weight
                    beat_alignment * 0.15 +       # Reduced weight
                    phase_continuity * 0.1        # Reduced weight
                )

                # Validate combined score
                if np.isnan(combined_score) or np.isinf(combined_score):
                    continue

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
                    valid_loops_found += 1

            print()  # New line after progress

        # Lower threshold for validity
        if valid_loops_found == 0 or best_score < 0.1:
            raise Exception(f"Nessun loop valido trovato con algoritmo avanzato (score: {best_score:.3f})")

        if best_start < 0 or best_end > len(y) or best_start >= best_end:
            raise Exception(f"Loop bounds finali non validi: {best_start} -> {best_end}")

        loop_duration = (best_end - best_start) / sr
        if loop_duration < 1.0:
            raise Exception(f"Loop finale troppo corto: {loop_duration:.1f}s")

        self.log_message(f"✅ Loop avanzato trovato! {best_measure_count} misure, Score: {best_score:.3f}, Durata: {loop_duration:.1f}s")

        return {
            'start_sample': best_start,
            'end_sample': best_end,
            'score': best_score,
            'measures': best_measure_count,
            'metrics': best_metrics
        }

    def process_loop_detection(self, input_file, output_file):
        """Main loop detection function with file safety - COMPLETELY FIXED"""
        try:
            self.debug_file_state("PRE_LOOP_DETECTION", input_file)

            # Validate input file first
            if not self.validate_audio_file(input_file):
                raise Exception(f"File di input non valido: {input_file}")

            # Load audio file
            y, sr = librosa.load(input_file, sr=None, mono=True)

            # Validate loaded audio data
            if len(y) == 0:
                raise Exception("Audio caricato è vuoto")
            if sr <= 0:
                raise Exception(f"Sample rate non valido: {sr}")
            if np.isnan(y).any() or np.isinf(y).any():
                raise Exception("Audio contiene valori NaN o infiniti")

            original_duration = len(y) / sr
            if original_duration < 2.0:
                raise Exception(f"Audio troppo corto per loop detection: {original_duration:.1f}s")

            # Find perfect loop
            loop_info = self.find_perfect_loop(y, sr)
            start_sample = loop_info['start_sample']
            end_sample = loop_info['end_sample']
            measures = loop_info['measures']
            score = loop_info['score']

            # Validate loop bounds
            if start_sample < 0 or end_sample > len(y) or start_sample >= end_sample:
                raise Exception(f"Loop bounds non validi: {start_sample} -> {end_sample} (max: {len(y)})")

            # Optimize zero-crossing
            self.log_message(f"🎯 Ottimizzazione zero-crossing...")
            start_sample = self.find_optimal_zero_crossing(y, start_sample)
            end_sample = self.find_optimal_zero_crossing(y, end_sample)

            # Re-validate after zero-crossing optimization
            if start_sample < 0 or end_sample > len(y) or start_sample >= end_sample:
                raise Exception(f"Loop bounds corrotti dopo zero-crossing: {start_sample} -> {end_sample}")

            # Print metrics
            print(f"\n📊 Metriche loop:")
            for metric, value in loop_info['metrics'].items():
                print(f"   {metric}: {value:.3f}")

            # Extract loop
            y_loop = y[start_sample:end_sample]

            # Validate extracted loop
            if len(y_loop) == 0:
                raise Exception("Loop estratto è vuoto")

            loop_duration = len(y_loop) / sr
            if loop_duration < 1.0:
                raise Exception(f"Loop troppo corto: {loop_duration:.1f}s")

            if np.isnan(y_loop).any() or np.isinf(y_loop).any():
                raise Exception("Loop estratto contiene valori NaN o infiniti")

            # Apply fade
            fade_samples = min(256, len(y_loop) // 100)
            if fade_samples > 0:
                fade_in = np.linspace(0, 1, fade_samples)
                fade_out = np.linspace(1, 0, fade_samples)
                y_loop[:fade_samples] *= fade_in
                y_loop[-fade_samples:] *= fade_out

            # Final validation before writing
            if len(y_loop) == 0:
                raise Exception("Loop finale è vuoto dopo fade")

            # Remove output file if it exists (ensure clean write)
            if os.path.exists(output_file):
                os.remove(output_file)

            # Write to output file with error handling
            try:
                sf.write(output_file, y_loop, sr)
            except Exception as write_error:
                raise Exception(f"Errore scrittura file audio: {write_error}")

            # Verify file was written correctly
            if not os.path.exists(output_file):
                raise Exception("File di output non creato")

            output_size = os.path.getsize(output_file)
            if output_size < 1024:  # Must be more than just header
                raise Exception(f"File di output troppo piccolo: {output_size} bytes")

            self.debug_file_state("POST_LOOP_DETECTION", output_file)

            # Final validation
            if not self.validate_audio_file(output_file):
                # Try to understand why validation failed
                try:
                    test_y, test_sr = librosa.load(output_file, sr=None, mono=True)
                    test_duration = len(test_y) / test_sr
                    raise Exception(f"File scritto ma validazione fallita (dur: {test_duration:.1f}s, samples: {len(test_y)})")
                except Exception as load_error:
                    raise Exception(f"File scritto ma non leggibile: {load_error}")

            duration = len(y_loop) / sr
            self.log_message(f"🧬 Ottenuto loop perfetto! (Forse...) \n              {measures} misure, {duration:.1f}s, Score: {score:.3f}")

        except Exception as e:
            # Clean up corrupted output file
            if os.path.exists(output_file):
                try:
                    os.remove(output_file)
                    self.log_message(f"🗑️ Rimosso file corrotto: {os.path.basename(output_file)}")
                except:
                    pass
            self.log_message(f"❌ Errore loop detection: {str(e)}")
            raise

    def generate_audio_safe(self, outfile):
        """Generate audio using MusicGPT with proper file handling - COMPLETELY REWRITTEN"""
        try:
            self.is_generating = True
            self.generation_status = f"Generando con prompt: '{self.PROMPT}'"
            self.log_message(f"\U0001F3BC Genero un nuovo sample...")

            # Use safe temporary file context
            with self.safe_temp_file() as raw_temp_path:
                with self.safe_temp_file() as processed_temp_path:

                    # Step 1: Generate to first temp file
                    self.debug_file_state("PRE_GENERATION", raw_temp_path)

                    result = subprocess.run([
                        "./musicgpt-x86_64-unknown-linux-gnu",
                        self.PROMPT,
                        "--model", self.model,
                        "--secs", str(self.duration),
                        "--output", raw_temp_path,
                        "--no-playback",
                        "--no-interactive",
                        "--ui-no-open"
                    ], check=True, capture_output=True, text=True)

                    self.debug_file_state("POST_GENERATION", raw_temp_path)

                    # Step 2: Validate generated file
                    if not self.validate_audio_file(raw_temp_path):
                        raise Exception("File audio generato con errori dalla AI.")

                    os.system("cls" if os.name == "nt" else "clear")
                    self.log_message(f"\U0001F3BC Sample generato ({self.duration}s)!")

                    self.generation_status = "Analisi loop..."

                    # Step 3: Process loop detection to second temp file
                    self.process_loop_detection(raw_temp_path, processed_temp_path)

                    # Step 4: Final validation
                    if not self.validate_audio_file(processed_temp_path):
                        raise Exception("File corrotto dopo il loop detection")

                    # Step 5: Atomic move to final destination
                    self.debug_file_state("PRE_FINAL_MOVE", processed_temp_path)

                    with self.file_lock:
                        shutil.move(processed_temp_path, outfile)

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
        """Get audio file duration safely"""
        try:
            with self.file_lock:
                if not os.path.exists(filepath):
                    return 0.0
                info = mediainfo(filepath)
                return float(info['duration'])
        except:
            return 0.0

    def get_random_title(self):
        """Get random track title"""
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
        """Get random artist name"""
        try:
            with open("artisti.txt", "r") as f:
                artists = [line.strip() for line in f if line.strip()]
            return random.choice(artists).upper() if artists else "ARTISTA SCONOSCIUTO"
        except Exception:
            return "ARTISTA SCONOSCIUTO"

    def play_with_ffplay(self, filepath):
        """Play audio file with ffplay - crash resistant"""
        ffplay_process = None
        try:
            env = os.environ.copy()
            env["SDL_AUDIODRIVER"] = self.audio_driver

            # Ensure file exists and is valid before playing
            if not self.validate_audio_file(filepath):
                self.log_message(f"⚠️ File non valido per riproduzione: {filepath}")
                return

            self.debug_file_state("START_PLAYBACK", filepath)

            # Start ffplay process
            ffplay_process = subprocess.Popen([
                "ffplay",
                "-nodisp",
                "-autoexit",
                "-loglevel", "quiet",
                filepath
            ], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Wait for process to complete
            return_code = ffplay_process.wait()

            self.debug_file_state("END_PLAYBACK", filepath)

            if return_code != 0:
                self.log_message(f"⚠️ ffplay terminato con codice {return_code}")

        except subprocess.TimeoutExpired:
            self.log_message("⚠️ ffplay timeout - terminazione forzata")
            if ffplay_process:
                try:
                    ffplay_process.kill()
                    ffplay_process.wait(timeout=2)
                except:
                    pass
        except Exception as e:
            self.log_message(f"⚠️ ffplay crash rilevato: {str(e)}")
            if ffplay_process:
                try:
                    ffplay_process.kill()
                    ffplay_process.wait(timeout=2)
                except:
                    pass
        finally:
            # Ensure process is always cleaned up
            if ffplay_process and ffplay_process.poll() is None:
                try:
                    ffplay_process.terminate()
                    ffplay_process.wait(timeout=1)
                except:
                    try:
                        ffplay_process.kill()
                        ffplay_process.wait(timeout=1)
                    except:
                        pass

    def loop_current_crossfade_blocking(self, filepath, crossfade_sec, stop_event):
        """Loop current file with crossfade - crash resistant"""
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

            while not stop_event.is_set() and self.is_playing:
                try:
                    # Validate file before each play (detect corruption early)
                    if not self.validate_audio_file(filepath):
                        self.log_message(f"⚠️ File corrotto rilevato durante loop: {filepath}")
                        break

                    # Create play thread with timeout
                    play_thread = threading.Thread(target=self.play_with_ffplay, args=(filepath,), daemon=True)
                    play_thread.start()

                    # Wait for delay or stop signal
                    if stop_event.wait(delay):
                        break

                    # Reset retry counter on successful play
                    retry_count = 0

                except Exception as e:
                    retry_count += 1
                    self.log_message(f"⚠️ Errore riproduzione (tentativo {retry_count}/{max_retries}): {str(e)}")

                    if retry_count >= max_retries:
                        self.log_message("❌ Troppi errori di riproduzione, salto al prossimo loop")
                        break

                    # Short delay before retry
                    if not stop_event.wait(1.0):
                        continue
                    else:
                        break

        except Exception as e:
            self.log_message(f"❌ Errore nel loop: {str(e)}")

    def safe_file_swap(self):
        """FIXED: Swap sicuro con sincronizzazione"""
        with self.swap_lock:  # Prevent concurrent swaps
            try:
                # Step 1: Stop current playback gracefully
                old_stop_event = self.stop_event
                old_stop_event.set()

                # Step 2: Wait for current loop to finish naturally
                if self.loop_thread and self.loop_thread.is_alive():
                    current_duration = self.get_duration(self.CURRENT)
                    max_wait = min(current_duration + 3.0, 10.0)  # Max 10 seconds wait
                    self.loop_thread.join(timeout=max_wait)

                    if self.loop_thread.is_alive():
                        self.log_message("⚠️ Timeout attesa - forzo terminazione ffplay")
                        self.kill_all_ffplay_processes()
                        self.loop_thread.join(timeout=2.0)

                # Step 3: Validate NEXT file before swapping
                if not self.validate_audio_file(self.NEXT):
                    raise Exception(f"File NEXT non valido: {self.NEXT}")

                # Step 4: Do the atomic swap
                with self.file_lock:
                    old_current = self.CURRENT
                    old_next = self.NEXT

                    self.CURRENT = old_next
                    self.NEXT = old_current

                # Step 5: Create new stop event for next loop
                self.stop_event = threading.Event()

                return True

            except Exception as e:
                self.log_message(f"❌ Errore durante swap: {str(e)}")
                # Try to restore state
                self.stop_event = threading.Event()
                return False

    def run_loop(self):
        """Main loop execution - COMPLETELY REWRITTEN with safe swapping"""
        # Start first loop
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
                # Generate next sample
                self.generate_audio_safe(self.NEXT)

                if not self.is_playing:
                    break

                # Safe file swap with synchronization
                if not self.safe_file_swap():
                    self.log_message("❌ Swap fallito, rigenerazione...")
                    continue

                if not self.is_playing:
                    break

                # Start new loop with swapped file
                loop_thread = threading.Thread(
                    target=self.loop_current_crossfade_blocking,
                    args=(self.CURRENT, self.CROSSFADE_SEC, self.stop_event),
                    daemon=True
                )
                loop_thread.start()
                self.loop_thread = loop_thread

                # Reset error counter on success
                consecutive_errors = 0

            except Exception as e:
                consecutive_errors += 1
                self.log_message(f"❌ Errore nel ciclo ({consecutive_errors}/{max_consecutive_errors}): {str(e)}")

                if consecutive_errors >= max_consecutive_errors:
                    self.log_message("❌ Troppi errori consecutivi, arresto loop")
                    self.is_playing = False
                    break

                # Clean up and retry
                self.kill_all_ffplay_processes()

                # Wait before retry
                if not self.stop_event.wait(2.0):
                    continue
                else:
                    break

    def main_loop(self):
        """Initialize and start main loop - crash resistant"""
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries and self.is_playing:
            try:
                # Clean up any existing processes
                self.kill_all_ffplay_processes()
                self.kill_all_musicgpt_processes()

                # Generate initial files if needed
                for file_path, file_name in [(self.CURRENT, "primo"), (self.NEXT, "secondo")]:
                    if not self.validate_audio_file(file_path):
                        self.log_message(f"📁 Generazione sample iniziale ({file_name})...")
                        self.generate_audio_safe(file_path)
                        if not self.is_playing:  # Check if stopped during generation
                            return

                        # Double-check the file is valid
                        if not self.validate_audio_file(file_path):
                            raise Exception(f"File {file_path} non generato correttamente")

                        self.log_message(f"✅ File {os.path.basename(file_path)} generato e validato")

                # Start infinite loop
                self.run_loop()

                # If we get here without exception, break retry loop
                break

            except Exception as e:
                retry_count += 1
                self.log_message(f"❌ Errore nel loop principale (tentativo {retry_count}/{max_retries}): {str(e)}")

                if retry_count >= max_retries:
                    self.log_message("❌ Troppi errori, arresto applicazione")
                    self.is_playing = False
                    return

                # Clean up and retry
                self.log_message(f"🔄 Reinizializzazione in corso...")
                self.kill_all_ffplay_processes()
                self.kill_all_musicgpt_processes()

                # Wait before retry
                time.sleep(2)

                # Remove corrupted files for retry
                try:
                    for filepath in [self.CURRENT, self.NEXT]:
                        if os.path.exists(filepath):
                            os.remove(filepath)
                            self.log_message(f"🗑️ Rimosso {os.path.basename(filepath)}")
                except Exception as remove_error:
                    self.log_message(f"⚠️ Errore rimozione file: {remove_error}")

    def start_loop(self, prompt):
        """Start the infinite loop"""
        self.PROMPT = prompt.strip()
        if not self.PROMPT:
            print("❌ Errore: Inserisci un prompt musicale!")
            return False

        self.is_playing = True

        # Start main loop in thread
        self.loop_thread = threading.Thread(target=self.main_loop, daemon=True)
        self.loop_thread.start()

        return True

    def stop_loop(self):
        """Stop the infinite loop"""
        self.is_playing = False
        if hasattr(self, 'stop_event'):
            self.stop_event.set()
        self.log_message("⏹️ Loop fermato")

        # Kill all audio processes
        self.kill_all_ffplay_processes()
        self.kill_all_musicgpt_processes()

    def kill_all_ffplay_processes(self):
        """Kill all running ffplay processes"""
        try:
            # Find and kill ffplay processes
            result = subprocess.run(["pgrep", "-f", "ffplay"], capture_output=True, text=True)
            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    try:
                        subprocess.run(["kill", "-9", pid], check=False, timeout=2)
                    except:
                        pass
        except Exception as e:
            # Silent fail - non critical
            pass

    def kill_all_musicgpt_processes(self):
        """Kill all running MusicGPT processes"""
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
            # Silent fail - non critical
            pass

    def save_current_loop(self, filename):
        """Save current loop to file"""
        with self.file_lock:
            # Find the currently playing file
            current_file = self.CURRENT

            if not self.validate_audio_file(current_file):
                print("❌ Nessun loop valido da salvare!")
                return False

            try:
                # Copy the current file
                shutil.copy2(current_file, filename)
                self.log_message(f"💾 Loop salvato: {filename} (da {os.path.basename(current_file)})")
                return True
            except Exception as e:
                self.log_message(f"❌ Errore salvataggio: {str(e)}")
                return False

    def print_status(self):
        """Print current status"""
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

        # Show file status with validation
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
    """Interactive terminal mode"""
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
                # Add .wav extension if not present
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
                print("   start '<prompt>'    - Avvia loop infinito con prompt musicale")
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
  %(prog)s --prompt "ambient loop"                    # Avvia con prompt
  %(prog)s --duration 20 --prompt "jazz loop"        # Durata 20s
  %(prog)s --interactive                              # Modalità interattiva
  %(prog)s --generate-only "rock loop" output.wav    # Solo generazione

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

    # Main options
    parser.add_argument("--prompt", "-p", type=str,
                       help="Prompt musicale per la generazione")

    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Modalità interattiva")

    parser.add_argument("--generate-only", "-g", nargs=2, metavar=("PROMPT", "OUTPUT"),
                       help="Genera solo un loop e salva (prompt, file_output)")

    # Remaining configurable settings
    parser.add_argument("--duration", "-d", type=int, default=15,
                       help="Durata generazione in secondi (5-30)")

    parser.add_argument("--driver", choices=["pulse", "alsa", "dsp"],
                       default="pulse", help="Driver audio")

    # Output options
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Output dettagliato")

    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Output minimale")

    parser.add_argument("--no-debug", action="store_true",
                       help="Disabilita debug mode")

    args = parser.parse_args()

    # Validate arguments
    if args.duration < 5 or args.duration > 30:
        print("❌ Errore: Durata deve essere tra 5 e 30 secondi")
        sys.exit(1)

    # Setup environment
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    os.environ["SDL_AUDIODRIVER"] = args.driver
    os.environ["ALSA_CARD"] = "default"

    # Create app instance
    app = InfiniLoopTerminal()

    # Apply settings
    app.duration = args.duration
    app.audio_driver = args.driver
    app.debug_mode = False if args.no_debug else False  # Always start with debug OFF

    print(f"🧠 Algoritmo:        Avanzato con fallback")
    print(f"🤖 Modello AI:       Medium")
    print(f"⏱️ Durata sample:    {app.duration}s")
    print(f"🔊 Driver audio:     {app.audio_driver}")
    print(f"🐛 Debug mode:       {'ON' if app.debug_mode else 'OFF'}")

    try:
        # Generate-only mode
        if args.generate_only:
            prompt, output_file = args.generate_only
            app.PROMPT = prompt
            print(f"\n🎼 Generazione singola: '{prompt}'")

            app.generate_audio_safe(output_file)
            print(f"✅ Loop salvato: {output_file}")
            return

        # Interactive mode
        elif args.interactive:
            interactive_mode(app)
            return

        # Direct prompt mode
        elif args.prompt:
            if app.start_loop(args.prompt):
                print("🎵 Loop avviato! Premi Ctrl+C per fermare")
                try:
                    # Keep main thread alive
                    while app.is_playing:
                        time.sleep(1)
                except KeyboardInterrupt:
                    app.stop_loop()
                    print("\n👋 Arrivederci!")
            return

        # No arguments - show help and enter interactive mode
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
