from keras.models import load_model
from keras.models import model_from_json
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from audio_transformations import plot_spectrum,stft_to_audio
from initial_audio_processing_functions import characterize_distortion


def blend_transition(context_section, predicted_section, start=True):
    for i in range(blend_bins):
        weight = (i + 1) / blend_bins  # Gradual transition weight
        if start:
            predicted_section[:, i] = (1 - weight) * context_section[:, -blend_bins + i] + weight * predicted_section[:, i]
        else:
            predicted_section[:, -i - 1] = (1 - weight) * predicted_section[:, -i - 1] + weight * context_section[:, i]



def predict_audio_and_plot(X_piano_val_reshaped,y_piano_val_reshaped,sample_index,blend_bins, sr = 48000, hop_length = 331, model_json_path, model_weights_path):
	# Load the model architecture from the JSON file
	with open("model.json", "r") as json_file:
	    loaded_model_json = json_file.read()
	# Reconstruct the model from the architecture
	loaded_model = model_from_json(loaded_model_json)
	# Load the model weights from the HDF5 file
	loaded_model.load_weights("best_model_weights.h5")
	# Compile the loaded model (Make sure to use the same optimizer and loss function as before)
	loaded_model.compile(optimizer=Adam(learning_rate=0.001), loss=custom_loss(loss_weights))
	# Now the model is ready for use

	sample_x = X_piano_val_reshaped[sample_index]
	sample_y = y_piano_val_reshaped[sample_index]
	# Make predictions using the loaded and compiled model
	predicted_y = loaded_model.predict(np.expand_dims(sample_x, axis=0))[0]
	# Extract contexts from sample_x
	left_context_real = sample_x[:, :28, 0]
	left_context_imag = sample_x[:, :28, 1]
	right_context_real = sample_x[:, 0:28, 2]
	right_context_imag = sample_x[:, 0:28, 3]
	# Correct dimensions for zero-padding
	gap_zeros_real = np.zeros((sample_y.shape[0], sample_y.shape[1]))
	gap_zeros_imag = np.zeros((sample_y.shape[0], sample_y.shape[1]))

	# Blend the start of predicted_y with the end of left_context
	blend_transition(left_context_real, predicted_y[:, :, 0], start=True)
	blend_transition(left_context_imag, predicted_y[:, :, 1], start=True)
	# Blend the end of predicted_y with the start of right_context
	blend_transition(right_context_real, predicted_y[:, :, 0], start=False)
	blend_transition(right_context_imag, predicted_y[:, :, 1], start=False)


	# Audio reconstructions
	context_audio = stft_to_audio(
	    np.concatenate([left_context_real, gap_zeros_real, right_context_real], axis=1),
	    np.concatenate([left_context_imag, gap_zeros_imag, right_context_imag], axis=1),
	    hop_length)

	original_audio = stft_to_audio(
	    np.concatenate([left_context_real, sample_y[:, :, 0], right_context_real], axis=1),
	    np.concatenate([left_context_imag, sample_y[:, :, 1], right_context_imag], axis=1),
	    hop_length)

	predicted_audio = stft_to_audio(
	    np.concatenate([left_context_real, predicted_y[:, :, 0], right_context_real], axis=1),
	    np.concatenate([left_context_imag, predicted_y[:, :, 1], right_context_imag], axis=1),
	    hop_length)

	# Plotting
	plt.figure(figsize=(15,18))

	# Spectrum: Context with Gap
	plt.subplot(6, 2, 1)
	plot_spectrum(
	    np.concatenate([left_context_real, gap_zeros_real, right_context_real], axis=1),
	    np.concatenate([left_context_imag, gap_zeros_imag, right_context_imag], axis=1),
	    hop_length=hop_length, sr=sr
	)
	plt.title("Context with Gap Spectrum")

	# Waveform: Context with Gap
	plt.subplot(6, 2, 2)
	librosa.display.waveshow(context_audio, sr=sr)
	plt.title("Context with Gap Waveform")

	# Spectrum: Original
	plt.subplot(6, 2, 3)
	plot_spectrum(
	    np.concatenate([left_context_real, sample_y[:, :, 0], right_context_real], axis=1),
	    np.concatenate([left_context_imag, sample_y[:, :, 1], right_context_imag], axis=1),
	    hop_length=hop_length, sr=sr
	)
	plt.title("Original Spectrum")

	# Waveform: Original
	plt.subplot(6, 2, 4)
	librosa.display.waveshow(original_audio, sr=sr)
	plt.title("Original Waveform")

	# Spectrum: Predicted
	plt.subplot(6, 2, 5)
	plot_spectrum(
	    np.concatenate([left_context_real, predicted_y[:, :, 0], right_context_real], axis=1),
	    np.concatenate([left_context_imag, predicted_y[:, :, 1], right_context_imag], axis=1),
	    hop_length=hop_length, sr=sr
	)
	plt.title("Predicted Spectrum")

	# Waveform: Predicted
	plt.subplot(6, 2, 6)
	librosa.display.waveshow(predicted_audio, sr=sr)
	plt.title("Predicted Waveform")

	plt.tight_layout()
	plt.show()


def fill_gaps_with_predictions(X_tf_reshaped,X_train_tf,audio_path, n_fft = 662,sr = 48000, hop_length = 331, model_json_path, model_weights_path)
	# Loop over the first 10 samples
	with open("model.json", "r") as json_file:
	    loaded_model_json = json_file.read()
	# Reconstruct the model from the architecture
	loaded_model = model_from_json(loaded_model_json)
	# Load the model weights from the HDF5 file
	loaded_model.load_weights("best_model_weights.h5")
	# Compile the loaded model (Make sure to use the same optimizer and loss function as before)
	loaded_model.compile(optimizer=Adam(learning_rate=0.001), loss=custom_loss(loss_weights))

	# Now the model is ready for use
	audio_segments= []
	for sample_index in range(X_train_tf.shape[0]):
    	sample_x = X_tf_reshaped[sample_index]
	   	# Make predictions using the loaded and compiled model
	    predicted_y = loaded_model.predict(np.expand_dims(sample_x, axis=0))[0]
	    # Number of bins for blending
	    left_context_real = sample_x[:, :28, 0]
	    left_context_imag = sample_x[:, :28, 1]
	    right_context_real = sample_x[:, -28:, 2]
	    right_context_imag = sample_x[:, -28:, 3]
	    X_train_tf[sample_index, :, 28:56, 0]=predicted_y[:, :, 0]
	    X_train_tf[sample_index, :, 28:56, 1]=predicted_y[:, :, 1]
	    audio_segments.append(stft_to_audio(predicted_y,predicted_y,331))

	y_original, sr = librosa.load(audio_path, sr=None)    
	gaps= characterize_distortion(audio_path)	
	# Replace the gaps in the original audio with the new audio segments
	for i, (start, end) in enumerate(gaps):
	    if i < len(audio_segments):
	        y_original[start:end] = audio_segments[i][:-1]  # Trim the last sample and 

	output_path = "/Users/benjamincolmey/Desktop/audio_predicted.wav"  # Set your desired output path
	# Save the modified audio to the output file
	sf.write(output_path, y_original, sr)	
	    