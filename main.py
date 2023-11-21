from initial_audio_processing_functions import load_audio, characterize_distortion, plot_gaps_and_lengths, extract_lengths, adjust_gaps, export_modified_audio,characterizing_original_audio,add_silences_to_audio,plot_comparison_of_audio,process_original_audio,split_audio_into_channels
from audio_transformations import extract_context_with_silent_gap,extract_truth_context,process_audio_files,audio_to_tf_parts,process_batch,process_single_sample,process_to_tf_representation,plot_spectrum,reshape_data,stft_to_audio,fill_gap_with_repeated_contexts
from models import weighted_mean_difference, mfcc_loss,custom_loss,train_and_save_model,build_unet,build_cnn,buil_encoder_decoder,predict_audio_and_plot,blend_transition
from testing_loss_functions import test_loss_functions
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io.wavfile import read
from array import array
from pydub import AudioSegment
import io

# Main Execution Script
if __name__ == "__main__":
    
    # -----------------------------------------------------------------------------------
    # Section 1: Importing Original Audio and Characterizing
    # -----------------------------------------------------------------------------------
    
    
    original_path = '/Users/benjamincolmey/Desktop/Ben_original_piano.wav'
    characterizing_original_audio(original_path)
    

    # -----------------------------------------------------------------------------------
    # Section 2: Creating Training and Testing Data
    # -----------------------------------------------------------------------------------
    
    
    test_path = '/Users/benjamincolmey/Desktop/piano/2A.wav'
    train_path = '//Users/benjamincolmey/Desktop/piano/2B.wav'
    silence_length_samples = 9599
    interval_between_silences_ms = 1000
    add_silences_to_audio(test_path, train_path, silence_length_samples, interval_between_silences_ms)
    original_audio = AudioSegment.from_file(test_path).set_channels(1)
    modified_audio = AudioSegment.from_file(train_path).set_channels(1)
    fs_train = 48000
    plot_comparison_of_audio(test_path, train_path, fs_train, context_samples=9599, duration_sec=3)
    
    
    # -----------------------------------------------------------------------------------
    # Section 3: Importing Training and Test Data
    # -----------------------------------------------------------------------------------
    
    filepaths_train = ['/Users/benjamincolmey/Desktop/Piano/1A.wav','/Users/benjamincolmey/Desktop/Piano/2A.wav','/Users/benjamincolmey/Desktop/Piano/4A.wav','/Users/benjamincolmey/Desktop/Piano/5A.wav','/Users/benjamincolmey/Desktop/Piano/6A.wav']
    filepaths_test = ['/Users/benjamincolmey/Desktop/Piano/1B.wav','/Users/benjamincolmey/Desktop/Piano/2B.wav','/Users/benjamincolmey/Desktop/Piano/4B.wav','/Users/benjamincolmey/Desktop/Piano/5B.wav','/Users/benjamincolmey/Desktop/Piano/6B.wav']
    
    X_train_all, y_train_all = [], []

    for train_path, test_path in zip(filepaths_train, filepaths_test):
        print(f"Processing: {train_path} and {test_path}")
        X_train, y_train = process_audio_files(train_path, test_path)
        X_train_all.append(X_train)
        y_train_all.append(y_train)

    X_train = np.concatenate(X_train_all)[..., np.newaxis, np.newaxis]
    y_train = np.concatenate(y_train_all)[..., np.newaxis, np.newaxis]
    print(f"X_train_all shape: {X_train.shape}")
    print(f"y_train_all shape: {y_train.shape}")
    
    
    # -----------------------------------------------------------------------------------
    # Section 4: Processing Training and Testing Data
    # -----------------------------------------------------------------------------------
    
    
    X_train_tf, y_train_tf = map(lambda data: np.array([process_single_sample(sample) for sample in data]).squeeze(axis=1), [X_train, y_train])
    print(f"X_train_tf shape: {X_train_tf.shape}")
    print(f"y_train_tf shape: {y_train_tf.shape}")
    X_train_tf, y_train_tf = X_train_tf.astype(np.float32), y_train_tf.astype(np.float32) 
    plot_spectrum(X_train_tf[9, :, :, 0], X_train_tf[9, :, :, 1])
    X_piano_train_reshaped, X_piano_val_reshaped, y_piano_train_reshaped, y_piano_val_reshaped = reshape_data(X_train_tf, y_train_tf)
    
    
    # -----------------------------------------------------------------------------------
    # Section 5: Testing Loss Functions Using Initial Guess of Gap as Reference
    # -----------------------------------------------------------------------------------
    
    filled_stft_repeated = fill_gap_with_repeated_contexts(X_train_tf, 1, 28, 55, context_size=14, plotting=True)
    test_loss_functions(1, X_piano_val_reshaped, y_piano_val_reshaped)
    

    # -----------------------------------------------------------------------------------
    # Section 6: Training the Model
    # -----------------------------------------------------------------------------------
    
    model = build_unet()
    model.compile(optimizer=Adam(learning_rate=1e-4), loss=custom_loss(loss_weights))
    history = train_and_save_model(
        model, 
        X_piano_train_reshaped, y_piano_train_reshaped,
        X_piano_val_reshaped, y_piano_val_reshaped,
        epochs=5,
        plot_history=True
    )
    

    # -----------------------------------------------------------------------------------
    # Section 7: Plotting Model Predictions and Saving Modified Audio File
    # -----------------------------------------------------------------------------------
    
    blend_bins = 3
    sample_index = 1
    predict_audio_and_plot(X_piano_val_reshaped, y_piano_val_reshaped, sample_index, blend_bins, sr=48000, hop_length=331, model=model)
    
    original_path = '/Users/benjamincolmey/Desktop/modified_piano.wav'
    X_real = process_original_audio(original_path)
    X_tf_reshaped, X_real_tf = split_audio_into_channels(X_real)
    fill_gaps_with_predictions(X_tf_reshaped, X_real_tf, original_path, n_fft=662, sr=48000, hop_length=331, model=model)
    
