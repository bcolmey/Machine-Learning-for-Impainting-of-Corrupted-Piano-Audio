from audio_transformations import extract_context_with_silent_gap,extract_truth_context,process_audio_files,audio_to_tf_parts,process_batch,process_single_sample,process_to_tf_representation,plot_spectrum,reshape_data,stft_to_audio,fill_gap_with_repeated_contexts
import numpy as np
from keras.metrics import RootMeanSquaredError
from keras.losses import MeanAbsolutePercentageError, MeanSquaredLogarithmicError, LogCosh, Huber, MeanAbsoluteError, CosineSimilarity
import tensorflow as tf

def test_loss_functions(sample_index,X_piano_val_reshaped,y_piano_val_reshaped):
  
  # Assuming 'sample_x' and 'sample_y' are loaded with the correct shape from previous steps
  sample_x = X_piano_val_reshaped[sample_index]
  sample_y = y_piano_val_reshaped[sample_index]

  # Ground truth and initial guess data
  ground_truth_real = sample_y[:, :, 0]
  ground_truth_imag = sample_y[:, :, 1]
  initial_guess_real = sample_x[:, 0:28, 2]
  initial_guess_imag = sample_x[:, 0:28, 3]

  # Combine real and imaginary parts into complex numbers
  ground_truth_complex = ground_truth_real + 1j * ground_truth_imag
  initial_guess_complex = initial_guess_real + 1j * initial_guess_imag

  # Initialize loss functions
  rmse, mape, msle, logcosh, huber, mae, cosine_loss, mse = [LossFunction() for LossFunction in (RootMeanSquaredError, MeanAbsolutePercentageError, MeanSquaredLogarithmicError, LogCosh, Huber, MeanAbsoluteError, CosineSimilarity, MeanSquaredError)]

  # Define the weights for the custom loss
  custom_loss_weights = {
      'mean_diff_weight': 1.0,  # Weight for mean difference loss
  }

  # Calculate losses for the initial guess
  initial_guess_rmse = np.sum([loss(ground_truth_real, initial_guess_real).numpy() + loss(ground_truth_imag, initial_guess_imag).numpy() for loss in (rmse, mape, msle, logcosh, huber, mae, cosine_loss)])
  initial_guess_mse = np.mean((ground_truth_real - initial_guess_real)**2 + (ground_truth_imag - initial_guess_imag)**2)  # Calculate MSE manually
  initial_guess_mape = np.sum([mape(ground_truth_real, initial_guess_real).numpy() + mape(ground_truth_imag, initial_guess_imag).numpy() for loss in (rmse, mape, msle, logcosh, huber, mae, cosine_loss)])
  initial_guess_msle = np.sum([msle(ground_truth_real, initial_guess_real).numpy() + msle(ground_truth_imag, initial_guess_imag).numpy() for loss in (rmse, mape, msle, logcosh, huber, mae, cosine_loss)])
  initial_guess_logcosh = np.sum([logcosh(ground_truth_real, initial_guess_real).numpy() + logcosh(ground_truth_imag, initial_guess_imag).numpy() for loss in (rmse, mape, msle, logcosh, huber, mae, cosine_loss)])
  initial_guess_huber = np.sum([huber(ground_truth_real, initial_guess_real).numpy() + huber(ground_truth_imag, initial_guess_imag).numpy() for loss in (rmse, mape, msle, logcosh, huber, mae, cosine_loss)])
  initial_guess_mae = np.sum([mae(ground_truth_real, initial_guess_real).numpy() + mae(ground_truth_imag, initial_guess_imag).numpy() for loss in (rmse, mape, msle, logcosh, huber, mae, cosine_loss)])
  initial_guess_cosine = np.sum([cosine_loss(ground_truth_real, initial_guess_real).numpy() + cosine_loss(ground_truth_imag, initial_guess_imag).numpy() for loss in (rmse, mape, msle, logcosh, huber, mae, cosine_loss)])
  # Calculate the weighted mean difference loss
  initial_guess_mean_diff_loss = tf.reduce_mean(weighted_mean_difference_loss(ground_truth_real, initial_guess_real, custom_loss_weights) + weighted_mean_difference_loss(ground_truth_imag, initial_guess_imag, custom_loss_weights)).numpy()

  # Print results
  print("Initial Guess Similarities:")
  print(f"Weighted Mean Difference Loss: {initial_guess_mean_diff_loss:.6f}")
  print("RMSE:", initial_guess_rmse, "MAPE:", initial_guess_mape, "MSLE:", initial_guess_msle, "LogCosh:", initial_guess_logcosh, "Huber:", initial_guess_huber, "MAE:", initial_guess_mae, "Cosine Similarity:", initial_guess_cosine, "MSE:", initial_guess_mse)
  print(f"Centroid: {initial_guess_similarity[0]}, Bandwidth: {initial_guess_similarity[1]}, Contrast: {initial_guess_similarity[2]}, Flatness: {initial_guess_similarity[3]}, Chroma: {initial_guess_similarity[4]}, MFCC: {initial_guess_similarity[5]}")


# Define the custom loss function for weighted mean difference
def weighted_mean_difference_loss(y_true, y_pred, weights):
    mean_pred = tf.reduce_mean(y_pred, axis=-1)
    mean_true = tf.reduce_mean(y_true, axis=-1)
    mean_difference = tf.abs(mean_pred - mean_true)
    weighted_loss = weights['mean_diff_weight'] * mean_difference
    return weighted_loss

# Function to calculate similarity
def calculate_similarity(ground_truth, comparison, sr):
    ground_truth_signal = librosa.istft(ground_truth)
    comparison_signal = librosa.istft(comparison)
    
    centroid_ground, centroid_comparison = [librosa.feature.spectral_centroid(y=signal, sr=sr)[0] for signal in (ground_truth_signal, comparison_signal)]
    bandwidth_ground, bandwidth_comparison = [librosa.feature.spectral_bandwidth(y=signal, sr=sr)[0] for signal in (ground_truth_signal, comparison_signal)]
    contrast_ground, contrast_comparison = [librosa.feature.spectral_contrast(y=signal, sr=sr).mean(axis=1) for signal in (ground_truth_signal, comparison_signal)]
    flatness_ground, flatness_comparison = [librosa.feature.spectral_flatness(y=signal)[0] for signal in (ground_truth_signal, comparison_signal)]
    chroma_ground, chroma_comparison = [librosa.feature.chroma_stft(y=signal, sr=sr).mean(axis=1) for signal in (ground_truth_signal, comparison_signal)]
    mfcc_ground, mfcc_comparison = [librosa.feature.mfcc(y=signal, sr=sr).mean(axis=1) for signal in (ground_truth_signal, comparison_signal)]

    centroid_similarity = 1 - cosine(centroid_ground, centroid_comparison)
    bandwidth_similarity = 1 - cosine(bandwidth_ground, bandwidth_comparison)
    contrast_similarity = 1 - cosine(contrast_ground, contrast_comparison)
    flatness_similarity = 1 - cosine(flatness_ground, flatness_comparison)
    chroma_similarity = 1 - cosine(chroma_ground, chroma_comparison)
    mfcc_similarity = 1 - cosine(mfcc_ground, mfcc_comparison)

    return centroid_similarity, bandwidth_similarity, contrast_similarity, flatness_similarity, chroma_similarity, mfcc_similarity
