import os
import time
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from pydub import AudioSegment
import numpy as np


def fourier_transform(audio_data, frame_rate):
   n = len(audio_data)  # Number of samples

   yf = fft(audio_data)  # Maps time-domain signal to complex frequency domain representation, containing magnitude and phase info
   positive_amplitudes = np.abs(yf[:n // 2]) # Absolute value calculates magnitude of complex number (first half since it's symmetric)

   xf = fftfreq(n, 1 / frame_rate)  # Generates frequency bins corresponding to FFT indices
   positive_freqs = xf[:n // 2] # Retain only first half of xf since it's symmetric

   return positive_freqs, positive_amplitudes


def plot_fourier(frequency, amplitude, lower_bound, upper_bound, chart_title, chart_name):
   output_dir = "exports"
   os.makedirs(output_dir, exist_ok=True)
   output_path = os.path.join(output_dir, f"{chart_name}.png")

   plt.figure(figsize=(10, 6))
   # Add marker to show points
   plt.plot(frequency, amplitude, color='red', linewidth=0.8, marker='o', markersize=3)
   plt.title(f"{chart_title}")
   plt.xlabel("Frequency (Hz)")
   plt.ylabel("Amplitude")
   plt.grid(True, which="both", linestyle="--", linewidth=0.5)
   plt.xlim(lower_bound, upper_bound)

   plt.savefig(output_path, dpi=300, bbox_inches='tight')
   print(f"Graph saved to {output_path}")


def zero_padded_fourier_transform(audio_data, frame_rate, start_time, end_time, scale_n_by):
   if start_time == end_time:
      return np.array([]), np.array([])

   start_frame = int(start_time * frame_rate)
   end_frame = int(end_time * frame_rate)

   if start_frame >= len(audio_data) or end_frame > len(audio_data):
      raise ValueError(
         f"Frame indices out of bounds: start_frame={start_frame}, end_frame={end_frame}, total_frames={len(audio_data)}")

   if len(audio_data.shape) == 1:
      selected_audio_data = audio_data[start_frame:end_frame]
   else:
      selected_audio_data = audio_data[start_frame:end_frame, 0] if audio_data.ndim > 1 else audio_data[
                                                                                             start_frame:end_frame]

   n = len(selected_audio_data)
   zero_padded_data = np.pad(selected_audio_data, (0, round(scale_n_by * n)), 'constant')
   positive_freqs, positive_amplitudes = fourier_transform(zero_padded_data, frame_rate)

   return positive_freqs, positive_amplitudes


def live_zero_padded_fourier_transform(audio_data, frame_rate, scale_n_by):

   n = len(audio_data)
   zero_padded_data = np.pad(audio_data, (0, round(scale_n_by * n)), 'constant')

   positive_freqs, positive_amplitudes = fourier_transform(zero_padded_data, frame_rate)

   return positive_freqs, positive_amplitudes


def normalize_amplitude(amplitude_array):
   if len(amplitude_array) == 0:
      return amplitude_array  # Return the empty array as-is
   max_amplitude = max(amplitude_array)
   if max_amplitude == 0:
      return amplitude_array  # Return the array as-is if max is zero to avoid division by zero
   normalized_amplitudes = amplitude_array / max_amplitude
   return normalized_amplitudes


def weighted_sum(frequencies, amplitudes, peak_index):
   # Initialize left and right indices at the peak
   left_index = peak_index
   right_index = peak_index

   # Traverse both sides in parallel
   while left_index > 0 or right_index < len(amplitudes) - 1:
      # Check if left index can move
      if left_index > 0 and amplitudes[left_index - 1] <= amplitudes[left_index]:
         left_index -= 1
      # Check if right index can move
      if right_index < len(amplitudes) - 1 and amplitudes[right_index + 1] <= amplitudes[right_index]:
         right_index += 1
      # Stop when either side reaches a point higher than itself
      if (left_index > 0 and amplitudes[left_index - 1] > amplitudes[left_index]) or \
              (right_index < len(amplitudes) - 1 and amplitudes[right_index + 1] > amplitudes[right_index]):
         break

   # Extract the relevant frequencies and amplitudes
   relevant_frequencies = frequencies[left_index:right_index + 1]
   relevant_amplitudes = amplitudes[left_index:right_index + 1]

   # Compute the weighted average of the frequencies
   true_frequency = np.sum(relevant_frequencies * relevant_amplitudes) / np.sum(relevant_amplitudes)

   return true_frequency, left_index, right_index


def calculate_chunk_means(frequencies, amplitudes, center_freq, chunk_width=15):
   left_outer = (frequencies >= center_freq - chunk_width) & (frequencies < center_freq - chunk_width / 3)
   right_outer = (frequencies > center_freq + chunk_width / 3) & (frequencies <= center_freq + chunk_width)
   inner = (frequencies >= center_freq - chunk_width / 3) & (frequencies <= center_freq + chunk_width / 3)

   left_mean = np.mean(amplitudes[left_outer]) if np.any(left_outer) else 0
   right_mean = np.mean(amplitudes[right_outer]) if np.any(right_outer) else 0
   inner_mean = np.mean(amplitudes[inner]) if np.any(inner) else 0
   outer_mean = np.mean(amplitudes[left_outer | right_outer]) if np.any(left_outer | right_outer) else 0

   return inner_mean, outer_mean, left_mean, right_mean

def find_all_spikes(frequencies, amplitudes, is_parabola_threshold=1.95):
    amplitudes = amplitudes / np.max(amplitudes)
    peaks = []
    exclusion_ranges = []
    invalid_peaks_count = 0

    def blacklist_range(center_freq, tolerance):
        exclusion_low = center_freq - tolerance
        exclusion_high = center_freq + tolerance
        exclusion_ranges.append((exclusion_low, exclusion_high))

    while invalid_peaks_count < 10:
       # Step 1: Identify valid indices
       valid_indices = np.ones_like(amplitudes, dtype=bool)
       for low, high in exclusion_ranges:
          valid_indices &= ~((frequencies >= low) & (frequencies <= high))

       # Break if no valid indices are found
       if not np.any(valid_indices):
          break

       # Step 2: Find the current maximum index among valid amplitudes
       current_max_index = np.argmax(amplitudes[valid_indices])
       actual_index = np.where(valid_indices)[0][current_max_index]
       center_freq = frequencies[actual_index]

       # Step 3: Calculate chunk means starting with chunk_width=10
       inner_mean, outer_mean, left_mean, right_mean = calculate_chunk_means(frequencies, amplitudes, center_freq,
                                                                             chunk_width=10)

       # Check condition and retry with chunk_width=15 if necessary
       if amplitudes[actual_index] > 0.1 and inner_mean / max(outer_mean, 1e-10) < is_parabola_threshold:
          inner_mean, outer_mean, left_mean, right_mean = calculate_chunk_means(frequencies, amplitudes, center_freq,
                                                                                chunk_width=15)

       # Step 4: Check left-right imbalance
       if (left_mean > 2 * right_mean and left_mean > inner_mean) or (
               right_mean > 2 * left_mean and right_mean > inner_mean):
          blacklist_range(frequencies[actual_index], 0.05 * frequencies[actual_index])
          continue

       # Step 5: Validate peak using inner and outer means
       if outer_mean == 0 or inner_mean / outer_mean <= is_parabola_threshold:
          invalid_peaks_count += 1
          blacklist_range(frequencies[actual_index], 0.05 * frequencies[actual_index])
          continue

       # Step 6: Refine the peak frequency
       refined_frequency, left_index, right_index = weighted_sum(frequencies, amplitudes, actual_index)
       refined_amplitude = amplitudes[actual_index]
       peaks.append((refined_frequency, refined_amplitude))

       # Step 7: Blacklist the current frequency range
       blacklist_range(frequencies[actual_index], 0.05 * frequencies[actual_index])

    return sorted(peaks, key=lambda x: x[0])


def group_frequencies(peaks, min_overtones=2, min_fundamental_freq=80):
   def find_groups(peaks):
      # Filter out peaks with frequencies below the minimum fundamental frequency
      peaks = [(freq, amp) for freq, amp in peaks if freq >= min_fundamental_freq]

      # Return early if no valid peaks are left
      if not peaks:
         return []

      groups = {}

      def allowed_percent_error(ratio):
         if ratio < 2:
            return 0.01  # 1.01 tolerance for ratio close to 1
         elif ratio < 3:
            return 0.015  # 2.015 tolerance for ratio close to 2
         elif ratio < 4:
            return 0.02  # 3.02 tolerance for ratio close to 3
         elif ratio < 5:
            return 0.025  # 4.025 tolerance for ratio close to 4
         else:
            return None  # Exceeds maximum allowed ratio

      for freq, amp in peaks:
         shared = False
         for fundamental in list(groups.keys()):  # Check against existing fundamentals
            ratio = freq / fundamental
            error_margin = allowed_percent_error(round(ratio))
            if error_margin is None:  # Stop grouping if ratio exceeds maximum
               continue
            if abs(ratio - round(ratio)) < error_margin:  # Check if it is an overtone
               groups[fundamental].append((freq, amp))
               shared = True
               break
         if not shared:  # It's a new fundamental frequency
            groups[freq] = [(freq, amp)]

      # Find the lowest fundamental frequency
      lowest_fundamental = min(groups.keys())

      # Filter out groups based on the rules
      filtered_groups = [
         sorted(overtones, key=lambda x: x[0])  # Sort overtones by frequency
         for fundamental, overtones in groups.items()
         if (
                 fundamental <= 4.05 * lowest_fundamental  # Fundamental frequency constraint
                 and len(overtones) >= min_overtones  # Minimum overtones constraint
                 and sum(amp for _, amp in overtones) > 0.35  # Total amplitude constraint
         )
      ]

      return filtered_groups

   return find_groups(peaks)



def find_fundamental_frequency(groups):
   if not groups:
      return None  # No groups found

   highest_amplitude = 0
   fundamental_frequency = None

   for group in groups:
      if group:  # Ensure the group is not empty
         frequency, amplitude = group[0]  # The fundamental is always the first element
         if amplitude > highest_amplitude:
            highest_amplitude = amplitude
            fundamental_frequency = frequency

   return fundamental_frequency


def find_all_fundamental_frequencies(groups):
   if not groups:
      return []

   fundamental_frequencies = [(group[0][0], group[0][1]) for group in groups if group]
   if not fundamental_frequencies:
      return []

   return sorted(freq for freq, amp in fundamental_frequencies)


def find_nearest_interval(ratio):
   just_intonation_ratios = {
      "Unison": 1 / 1,
      "Minor Second": 16 / 15,
      "Major Second": 9 / 8,
      "Minor Third": 6 / 5,
      "Major Third": 5 / 4,
      "Perfect Fourth": 4 / 3,
      "Tritone": 7 / 5,
      "Perfect Fifth": 3 / 2,
      "Minor Sixth": 8 / 5,
      "Major Sixth": 5 / 3,
      "Minor Seventh": 9 / 5,
      "Major Seventh": 15 / 8,
      "Octave": 2 / 1,
      "Minor Ninth": 16 / 15 * 2,
      "Major Ninth": 9 / 8 * 2,
      "Minor Tenth": 6 / 5 * 2,
      "Major Tenth": 5 / 4 * 2
   }
   closest_name, closest_ratio = min(just_intonation_ratios.items(), key=lambda x: abs(ratio - x[1]))
   percent_error = round(float((ratio - closest_ratio) / closest_ratio) * 100, 6)
   cent_error = round(1200 * math.log2(ratio / closest_ratio), 6)
   return closest_name, percent_error, cent_error


import math


def find_nearest_note(frequency):
   # Frequency mapping for notes from C0 to C8 (A440 tuning)
   note_frequencies = {
      "C0": 16.35, "C#0": 17.32, "D0": 18.35, "Eb0": 19.45, "E0": 20.60, "F0": 21.83, "F#0": 23.12,
      "G0": 24.50, "G#0": 25.96, "A0": 27.50, "Bb0": 29.14, "B0": 30.87,
      "C1": 32.70, "C#1": 34.65, "D1": 36.71, "Eb1": 38.89, "E1": 41.20, "F1": 43.65, "F#1": 46.25,
      "G1": 49.00, "G#1": 51.91, "A1": 55.00, "Bb1": 58.27, "B1": 61.74,
      "C2": 65.41, "C#2": 69.30, "D2": 73.42, "Eb2": 77.78, "E2": 82.41, "F2": 87.31, "F#2": 92.50,
      "G2": 98.00, "G#2": 103.83, "A2": 110.00, "Bb2": 116.54, "B2": 123.47,
      "C3": 130.81, "C#3": 138.59, "D3": 146.83, "Eb3": 155.56, "E3": 164.81, "F3": 174.61, "F#3": 185.00,
      "G3": 196.00, "G#3": 207.65, "A3": 220.00, "Bb3": 233.08, "B3": 246.94,
      "C4": 261.63, "C#4": 277.18, "D4": 293.66, "Eb4": 311.13, "E4": 329.63, "F4": 349.23, "F#4": 369.99,
      "G4": 392.00, "G#4": 415.30, "A4": 440.00, "Bb4": 466.16, "B4": 493.88,
      "C5": 523.25, "C#5": 554.37, "D5": 587.33, "Eb5": 622.25, "E5": 659.25, "F5": 698.46, "F#5": 739.99,
      "G5": 783.99, "G#5": 830.61, "A5": 880.00, "Bb5": 932.33, "B5": 987.77,
      "C6": 1046.50, "C#6": 1108.73, "D6": 1174.66, "Eb6": 1244.51, "E6": 1318.51, "F6": 1396.91,
      "F#6": 1479.98, "G6": 1567.98, "G#6": 1661.22, "A6": 1760.00, "Bb6": 1864.66, "B6": 1975.53,
      "C7": 2093.00, "C#7": 2217.46, "D7": 2349.32, "Eb7": 2489.02, "E7": 2637.02, "F7": 2793.83,
      "F#7": 2959.96, "G7": 3135.96, "G#7": 3322.44, "A7": 3520.00, "Bb7": 3729.31, "B7": 3951.07,
      "C8": 4186.01
   }

   # Find the closest note by frequency
   closest_name, closest_freq = min(note_frequencies.items(), key=lambda x: abs(float(frequency) - float(x[1])))

   # Calculate percent error
   percent_error = round(float((float(frequency) - float(closest_freq)) / float(closest_freq)) * 100, 6)
   cent_error = round(float(1200 * math.log2(float(frequency) / float(closest_freq))), 6)

   return closest_name, percent_error, cent_error


def analyze_intervals(fundamental_frequencies):
   if len(fundamental_frequencies) == 1:
      frequency = fundamental_frequencies[0]
      note_name, percent_error, cent_error = find_nearest_note(frequency)
      return {
         "Name": note_name,
         "Percent Error": percent_error,
         "Cent Error": cent_error
      }

   elif len(fundamental_frequencies) == 2:
      larger, smaller = max(fundamental_frequencies), min(fundamental_frequencies)
      ratio = round(float(larger / smaller), 6)
      interval_name, percent_error, cent_error = find_nearest_interval(ratio)

      return {
         "Name": interval_name,
         "Percent Error": percent_error,
         "Cent Error": cent_error
      }

   else:
      return {}


def identify_notes(fundamental_frequencies):
   notes = []

   for frequency in fundamental_frequencies:
      note_name, _, cent_error = find_nearest_note(frequency)
      notes.append({
         "Name": note_name,
         "Cent Error": cent_error
      })

   return notes


def update_fundamental(curr, prev):
   diff = abs(curr - prev) / prev
   return curr if diff > 0.05 else 0.8 * prev + 0.2 * curr


def extract_audio_segment(input_path, output_path, start_time_seconds, end_time_seconds=None):
    try:
        # Load the audio
        audio = AudioSegment.from_wav(input_path)

        # Convert start and end times to milliseconds
        start_time_ms = start_time_seconds * 1000
        end_time_ms = end_time_seconds * 1000 if end_time_seconds else len(audio)

        # Extract the segment
        excerpt = audio[start_time_ms:end_time_ms]

        # Export the segment to a new file
        excerpt.export(output_path, format="wav")

        print(f"Excerpt saved as: {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
"""extract_audio_segment('/Users/branan/Downloads/How to play Twinkle Twinkle on the violin!.wav', 
                      '/Users/branan/Downloads/Twinkle_excerpt.wav', 
                      30, None)"""


def extract_audio_segment_for_localhost(input_path, output_path, start_time_seconds, end_time_seconds=None):
   try:
      # Load the audio file
      audio = AudioSegment.from_file(input_path)

      # Convert start and end times to milliseconds
      start_time_ms = start_time_seconds * 1000
      end_time_ms = end_time_seconds * 1000 if end_time_seconds else len(audio)

      # Extract the segment
      excerpt = audio[start_time_ms:end_time_ms]

      # Export the segment
      excerpt.export(output_path, format="wav")

      # Return the path to the exported file
      return output_path
   except Exception as e:
      print(f"Error in extract_audio_segment_for_localhost: {e}")
      return None


