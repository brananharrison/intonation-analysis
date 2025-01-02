import json
import pprint
import time
import wave

import pandas as pd
from PIL.ImageOps import scale
#from soupsieve.util import lower

import simpleaudio as sa
from functions import zero_padded_fourier_transform, plot_fourier, normalize_amplitude, weighted_sum, \
    find_all_spikes, group_frequencies, find_fundamental_frequency, find_all_fundamental_frequencies, analyze_intervals, \
    identify_notes
import os
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt


# Original "/Users/branan/Downloads/IMG_4261.wav"

# Sarabanda "/Users/branan/Downloads/IMG_4241.wav"
# 3 min "/Users/branan/Downloads/IMG_4293.wav"

audio_path = "/Users/branan/Downloads/How to play Twinkle Twinkle on the violin!.wav"

with wave.open(audio_path, 'rb') as wf:
   n_channels = wf.getnchannels()
   sample_width = wf.getsampwidth()
   frame_rate = wf.getframerate()
   n_frames = wf.getnframes()

   raw_data = wf.readframes(n_frames)
   audio_data = np.frombuffer(raw_data, dtype=np.int16)
   audio_data = audio_data.reshape(-1, 2)




start_time = 85.9
end_time = 86.0


def play_audio_segment(audio_data, frame_rate, start_time, end_time, n_channels):
   start_frame = int(start_time * frame_rate)
   end_frame = int(end_time * frame_rate)
   selected_data = audio_data[start_frame:end_frame]

   if n_channels == 2:
      selected_data = selected_data.flatten()

   audio_bytes = selected_data.astype(np.int16).tobytes()

   play_obj = sa.play_buffer(audio_bytes, num_channels=n_channels, bytes_per_sample=2, sample_rate=frame_rate)
   play_obj.wait_done()

play_audio_segment(audio_data, frame_rate, n_channels=n_channels,start_time=start_time,end_time=end_time)

frequencies, amplitude = zero_padded_fourier_transform(audio_data, frame_rate, start_time, end_time, scale_n_by=10)
plot_fourier(frequencies, normalize_amplitude(amplitude), 0, 2500, "Zero padding true", "Zero padding true")
result = find_all_spikes(frequencies, amplitude)
groups = group_frequencies(result)
fundamental_frequencies = find_all_fundamental_frequencies(groups)

print(identify_notes(fundamental_frequencies))



print("Result:")
for item in result:
    print(f"({item[0]:.2f}, {item[1]:.4f})")

print("\nGroups:")
for group in groups:
    print("[")
    for item in group:
        print(f"  ({item[0]:.2f}, {item[1]:.4f})")
    print("]")
print(fundamental_frequencies)


