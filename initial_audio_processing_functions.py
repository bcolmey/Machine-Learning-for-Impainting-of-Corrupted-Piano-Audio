from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io.wavfile import read
import os
import io
import array
import librosa

def load_audio(filepath):
    """Load audio file and return its data and original sampling rate."""
    audio = AudioSegment.from_file(filepath).set_channels(1)
    fs_original = audio.frame_rate
    return audio, fs_original

def characterize_distortion(filepath, threshold=100, min_gap_ms=100, max_gap_ms=360):
    """Characterize distortion gaps in an audio file."""
    audio = AudioSegment.from_file(filepath, format="wav").set_channels(1)

    # Get audio samples as an array of integers
    int_samples = array.array('h', audio.get_array_of_samples())
    samples = np.array(int_samples)  # Convert to a NumPy array
    silence = np.where(np.abs(samples) <= threshold)[0]
    
    # Split silence regions and calculate frame rate-based gap limits
    regions = np.split(silence, np.where(np.diff(silence) != 1)[0] + 1)
    frame_rate = audio.frame_rate
    min_samples, max_samples = int(frame_rate * min_gap_ms / 1000), int(frame_rate * max_gap_ms / 1000)
    
    # Filter gaps based on duration criteria and return as (start, end) tuples in samples
    return [(s[0], s[-1]) for s in regions if min_samples <= len(s) <= max_samples]

def characterize_distortions(samples, threshold=100, min_gap_ms=100, max_gap_ms=360, frame_rate=48000):
    """Characterize distortion gaps in audio samples."""
    silence = np.where(np.abs(samples) <= threshold)[0]
    regions = np.split(silence, np.where(np.diff(silence) != 1)[0] + 1)
    min_samples, max_samples = int(frame_rate * min_gap_ms / 1000), int(frame_rate * max_gap_ms / 1000)
    return [(s[0], s[-1] + 1) for s in regions if min_samples <= len(s) <= max_samples]  # Adjust end index

def plot_gaps_and_lengths(audio, fs, gaps, duration_sec, title):
    """Plot waveform and gap length distribution of audio."""
    max_samples = duration_sec * fs
    chunk = np.array(audio.get_array_of_samples()[:max_samples])
    time = np.linspace(0, duration_sec, num=len(chunk))

    # Create subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))

    # Plotting the waveform
    axs[0].plot(time, chunk, label="Audio", color='blue')
    axs[0].set_title(f"Waveform - {title}")
    axs[0].set_xlabel("Time (seconds)")
    axs[0].set_ylabel("Amplitude")
    for start, end in gaps:
        if start < max_samples:
            axs[0].axvspan(start / fs, min(end, max_samples) / fs, color='red', alpha=0.3)
    axs[0].grid(True)

    # Plotting the histogram of gap lengths
    gap_lengths = [(end - start) / fs for start, end in gaps if start < max_samples]
    axs[1].hist(gap_lengths, color="blue", bins='auto')
    axs[1].set_title(f"Distribution of Gap Lengths - {title}")
    axs[1].set_xlabel('Length (seconds)')
    axs[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def extract_lengths(distortions, frame_rate):
    """Extract gap lengths in seconds from distortion gaps."""
    return [(end - start) / frame_rate for start, end in distortions]

def adjust_gaps(audio, gaps, distortion_lengths, filepath, frame_rate):
    """Adjust gaps in audio to match a target gap size."""
    samples = np.array(audio.get_array_of_samples())
    
    for gap_index in range(len(gaps)):
        gaps = characterize_distortions(samples)
        if gap_index < len(gaps):
            start, end = gaps[gap_index]
            gap_length = end - start
            difference = 9599 - gap_length
            if difference > 0:
                additional_silence = np.zeros(difference, dtype=samples.dtype)
                samples = np.insert(samples, end, additional_silence)
            elif difference < 0:
                samples = np.delete(samples, range(end + difference, end))

    final_gaps = characterize_distortions(samples)
    return samples, final_gaps
    
def export_modified_audio(audio, original_filepath, new_filename, samples):
    """Export modified audio."""
    modified_audio = audio._spawn(samples.tobytes())
    directory_path = os.path.dirname(original_filepath)
    new_audio_path = os.path.join(directory_path, new_filename)
    modified_audio.export(new_audio_path, format="wav")
    print(f"Modified audio saved to {new_audio_path}")

def add_silences_to_audio(audio_path, output_path, silence_length_samples, interval_between_silences_ms):
    """
    Adds fixed-length silences to an audio file at regular intervals.
    """
    audio = AudioSegment.from_file(audio_path, format="wav").set_channels(1)
    silence_length_ms = (silence_length_samples / audio.frame_rate) * 1000
    final_audio = AudioSegment.empty()
    current_pos_ms = 0

    while current_pos_ms < len(audio):
        next_pos_ms = min(current_pos_ms + interval_between_silences_ms, len(audio))
        audio_segment = audio[current_pos_ms:next_pos_ms]
        final_audio += audio_segment

        if next_pos_ms < len(audio):
            silence = AudioSegment.silent(duration=silence_length_ms, frame_rate=audio.frame_rate)
            final_audio += silence

        current_pos_ms = next_pos_ms + silence_length_ms

    final_audio.export(output_path, format="wav")
def characterizing_original_audio(original_path):
    """Process and visualize original audio data before and after gap adjustments."""
    audio, fs_original = load_audio(original_path)
    samples = np.array(audio.get_array_of_samples())
    gaps_original = characterize_distortions(samples)
    
    duration_sec = 10  # Duration for plotting

    # Plot before processing
    plot_gaps_and_lengths(audio, fs_original, gaps_original, duration_sec, "Before Processing")

    modified_samples, final_gaps = adjust_gaps(audio, gaps_original, [], original_path, fs_original)

    # Create new AudioSegment object from modified samples
    modified_audio_data = io.BytesIO(modified_samples.tobytes())
    modified_audio = AudioSegment.from_raw(modified_audio_data, sample_width=audio.sample_width, frame_rate=audio.frame_rate, channels=audio.channels)

    # Plot after processing
    plot_gaps_and_lengths(modified_audio, fs_original, final_gaps, duration_sec, "After Processing")

    export_modified_audio(audio, original_path, 'modified_piano.wav', modified_samples)

def plot_comparison_of_audio(original_path, modified_path, fs, context_samples=9599, duration_sec=3):
    """Plot a comparison of original and modified audio."""
    original_audio = AudioSegment.from_file(original_path).set_channels(1)
    modified_audio = AudioSegment.from_file(modified_path).set_channels(1)

    # Limit processing to the first 'duration_sec' of the audio
    original_chunk = np.array(original_audio[:duration_sec * 1000].get_array_of_samples())
    modified_chunk = np.array(modified_audio[:duration_sec * 1000].get_array_of_samples())
    
    time = np.linspace(0, duration_sec, num=len(modified_chunk))

    plt.figure(figsize=(12, 8))
    plt.plot(time, original_chunk, label="Original Audio", color='red', alpha=0.5)
    plt.plot(time, modified_chunk, label="Modified Audio", color='darkblue', alpha=0.9)
    plt.title("Comparison of Original and Modified Audio (First 3 seconds)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()

def process_original_audio(original_path):
    """Process the original audio by adjusting gaps to a target size."""
    original_audio, fs_train = librosa.load(original_path, sr=None)
    gaps = characterize_distortion(original_path)

    context_length = int(0.2 * fs_train)
    target_gap_size_samples = 9598  # Adjusted to match your criteria
    exact_gaps = [gap for gap in gaps if (gap[1] - gap[0]) == target_gap_size_samples]

    max_length = 28799  # Target length for each segment
    X_real = []
    for gap in exact_gaps:
        segment = extract_context_with_silent_gap(original_audio, gap[0], gap[1], context_length)
        padded_segment = np.pad(segment, (0, max_length - len(segment)), 'constant')
        X_real.append(padded_segment)

    X_real= np.array(X_real)
    X_real_reshaped = X_real[..., np.newaxis, np.newaxis]  # Add two extra dimensions

    return X_real_reshaped

def split_audio_into_channels(X_real):
    """Split audio into channels and reshape for further processing."""
    print(f"X_real {X_real.shape}")
    # Transforming X_train to STFT representation
    X_real_tf = np.array([process_single_sample(sample) for sample in X_real]).squeeze(axis=1)
    X_real_tf = X_real_tf.astype(np.float32)

    print(f"X_real_tf shape: {X_real_tf.shape}")
    
        # Reshaping the X data
    before_gap_real = X_real_tf[:, :, :28, 0]
    before_gap_imag = X_real_tf[:, :, :28, 1]
    after_gap_real = X_real_tf[:, :, -28:, 0]
    after_gap_imag = X_real_tf[:, :, -28:, 1]

    X_tf_reshaped = np.stack((before_gap_real, before_gap_imag, after_gap_real, after_gap_imag), axis=3)  # Stack along the 4th dimension
    print(f"Reshaped X shape: {X_tf_reshaped.shape}")
    return X_tf_reshaped

def extract_context_with_silent_gap(audio, gap_start, gap_end, context_length):
    """Extracts context around a gap including a SILENT version of the gap."""
    before_gap = audio[gap_start-context_length:gap_start]
    silent_gap = np.zeros(gap_end - gap_start)  # This is the SILENT content of the gap
    after_gap = audio[gap_end:gap_end+context_length]
    return np.concatenate((before_gap, silent_gap, after_gap))

def extract_truth_context(audio, gap_start, gap_end, context_length):
    """Extracts context around a gap including the ORIGINAL content of the gap."""
    before_gap = audio[gap_start-context_length:gap_start]
    within_gap = audio[gap_start:gap_end]  # This is the ORIGINAL content of the gap
    after_gap = audio[gap_end:gap_end+context_length]
    return np.concatenate((before_gap, within_gap, after_gap))
