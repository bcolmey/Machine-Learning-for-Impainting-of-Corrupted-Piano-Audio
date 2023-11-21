import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization, Dropout, ReLU
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.layers import Input, Reshape, ConvLSTM2D, BatchNormalization, Conv2DTranspose, Activation
from keras.regularizers import l2
from keras import backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.models import load_model
from keras.models import model_from_json
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from audio_transformations import plot_spectrum,stft_to_audio
from initial_audio_processing_functions import characterize_distortion,process_original_audio
import soundfile as sf

def weighted_mean_difference(y_true, y_pred, weights):
    """Calculate the weighted mean difference loss between true and predicted values."""
    mean_pred = K.mean(y_pred, axis=-1)
    mean_true = K.mean(y_true, axis=-1)
    mean_difference = K.abs(mean_pred - mean_true)
    weighted_loss = weights['mean_diff_weight'] * mean_difference
    return weighted_loss

def mfcc_loss(y_true, y_pred):
    """Calculate the MFCC loss between true and predicted audio signals."""
    # Convert audio signals to MFCCs using TensorFlow operations
    mfcc_true = tf.signal.mfccs_from_log_mel_spectrograms(y_true)
    mfcc_pred = tf.signal.mfccs_from_log_mel_spectrograms(y_pred)

    # Calculate the loss as the mean squared error between the MFCCs
    loss = tf.reduce_mean(tf.square(mfcc_pred - mfcc_true), axis=-1)
    return loss

def custom_loss(weights):
    """Create a custom loss function that combines weighted mean difference loss and MFCC loss."""
    def loss(y_true, y_pred):
        # Weighted mean difference loss
        mean_diff_loss = weighted_mean_difference(y_true, y_pred, weights)

        # MFCC loss
        mfcc_loss_val = mfcc_loss(y_true, y_pred)

        # Weighted sum of the losses
        total_loss = weights['mean_diff_weight'] * mean_diff_loss + weights['mfcc_weight'] * mfcc_loss_val
        return total_loss
    return loss

# Define weights for different components of the loss
loss_weights = {
    'mean_diff_weight': 1.0,  # Weight for mean difference loss
    'mfcc_weight': 1.0,       # Weight for MFCC loss
}

def scheduler(epoch, lr):
    """Define a learning rate scheduler for the training process."""
    if epoch < 20:
        return lr
    else:
        return lr * np.exp(-0.1)


def train_and_save_model(model, X_train, y_train, X_val, y_val, epochs=50, plot_history=False):
    checkpoint = ModelCheckpoint(
        'best_model_weights.h5', 
        monitor='val_loss', 
        verbose=1, 
        save_best_only=True, 
        mode='min',
        save_weights_only=True
    )
    lr_scheduler = LearningRateScheduler(scheduler)
    callbacks = [checkpoint, lr_scheduler]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    # Optionally save the model architecture
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    # Optionally plot the training history
    if plot_history:
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

    return history

def build_unet():
    # Input shape
    input_shape = (332, 28, 4)  # Your input shape
    # Encoder
    inputs = Input(input_shape)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # Bottleneck
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    # Decoder
    up4 = concatenate([UpSampling2D(size=(2, 2))(conv3), conv2], axis=-1)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(up4)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(conv4)
    up5 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv1], axis=-1)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(up5)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(conv5)
    # Output layer
    output = Conv2D(2, 1, activation='relu', padding='same')(conv5)
    # Create model
    model = Model(inputs=[inputs], outputs=[output])
    # Compile model
    model.compile(optimizer=Adam(learning_rate=1e-4), loss=custom_loss(loss_weights))
    # Print model summary
    model.summary()
    return model

def build_cnn():
    # L2 regularization factor
    reg_strength = 0.0001
    # Input shape (332, 28, 4)
    input_tensor = Input(shape=(332, 28, 4))
    # Convolutional and Batch Normalization layers
    x = Conv2D(32, (3, 3), strides=1, padding='same', kernel_regularizer=l2(reg_strength))(input_tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.2)(x)
    x = Conv2D(64, (3, 3), strides=1, padding='same', kernel_regularizer=l2(reg_strength))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.3)(x)
    x = Conv2D(128, (3, 3), strides=1, padding='same', kernel_regularizer=l2(reg_strength))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.4)(x)
    x = Conv2D(64, (3, 3), strides=1, padding='same', kernel_regularizer=l2(reg_strength))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.3)(x)
    x = Conv2D(32, (3, 3), strides=1, padding='same', kernel_regularizer=l2(reg_strength))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.2)(x)
    # Final Conv2D layer to get the desired output shape
    x = Conv2D(2, (1, 1), strides=1, padding='same', kernel_regularizer=l2(reg_strength))(x)
    # Create the model instance
    model = Model(inputs=input_tensor, outputs=x)
    # Initialize the optimizer with a defined learning rate
    lr = 0.001
    optimizer = Adam(learning_rate=lr)
    # Use MSE for model compilation
    model.compile(optimizer=optimizer, loss=custom_loss(loss_weights))
    # Print the model summary
    model.summary()
    return model


def buil_encoder_decoder():
    input_tensor = Input(shape=(332, 28, 4))
    # Reshape input to 5D for ConvLSTM2D - adding a temporal dimension
    reshaped_input = Reshape((1, 332, 28, 4))(input_tensor)  # 1 is the temporal dimension
    # Encoder - ConvLSTM2D
    encoder = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=False)(reshaped_input)
    encoder = BatchNormalization()(encoder)
    # Decoder - reshape and use Conv2DTranspose to reconstruct spatial dimensions
    decoder = Reshape((332, 28, 64))(encoder)  # Adjust the shape as needed
    # Adding Conv2DTranspose layers to get to the desired output shape
    decoder = Conv2DTranspose(32, (3, 3), strides=(1, 1), padding='same')(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = Activation('relu')(decoder)
    decoder = Conv2DTranspose(2, (3, 3), strides=(1, 1), padding='same')(decoder)  # Output channels: 2
    # Create the model instance
    model = Model(inputs=input_tensor, outputs=decoder)
    # Initialize the optimizer with a defined learning rate
    lr = 0.001
    optimizer = Adam(learning_rate=lr)
    # Use MSE for model compilation
    model.compile(optimizer=optimizer, loss=custom_loss(loss_weights))
    # Print the model summary
    model.summary()
    return model


def blend_transition(context_section, predicted_section, start=True):
    for i in range(blend_bins):
        weight = (i + 1) / blend_bins  # Gradual transition weight
        if start:
            predicted_section[:, i] = (1 - weight) * context_section[:, -blend_bins + i] + weight * predicted_section[:, i]
        else:
            predicted_section[:, -i - 1] = (1 - weight) * predicted_section[:, -i - 1] + weight * context_section[:, i]



def predict_audio_and_plot(X_piano_val_reshaped,y_piano_val_reshaped,sample_index,blend_bins, sr = 48000, hop_length = 331, model=model):
    # Load the model architecture from the JSON file
    
    sample_x = X_piano_val_reshaped[sample_index]
    sample_y = y_piano_val_reshaped[sample_index]
    # Make predictions using the loaded and compiled model
    predicted_y = model.predict(np.expand_dims(sample_x, axis=0))[0]
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


def fill_gaps_with_predictions(X_tf_reshaped,X_train_tf,audio_path, n_fft = 662,sr = 48000, hop_length = 331, model=model):

    # Now the model is ready for use
    audio_segments= []
    for sample_index in range(X_train_tf.shape[0]):
        sample_x = X_tf_reshaped[sample_index]
        # Make predictions using the loaded and compiled model
        predicted_y = model.predict(np.expand_dims(sample_x, axis=0))[0]
        # Number of bins for blending
        left_context_real = sample_x[:, :28, 0]
        left_context_imag = sample_x[:, :28, 1]
        right_context_real = sample_x[:, -28:, 2]
        right_context_imag = sample_x[:, -28:, 3]
        X_train_tf[sample_index, :, 28:56, 0]=predicted_y[:, :, 0]
        X_train_tf[sample_index, :, 28:56, 1]=predicted_y[:, :, 1]
        filled_gap_real = X_train_tf[sample_index, :, 27:57, 0]
        filled_gap_imag = X_train_tf[sample_index, :, 27:57, 1]
        audio_segments.append(stft_to_audio(filled_gap_real,filled_gap_imag,331))
        
    y_original, sr = librosa.load(audio_path, sr=None)    
    gaps= characterize_distortion(audio_path)   
    # Replace the gaps in the original audio with the new audio segments
    for i, (start, end) in enumerate(gaps):
        if i < len(audio_segments):
            y_original[start:end] = audio_segments[i][:-1]  # Trim the last sample and 
    

    # Plot the entire original audio
    plt.figure(figsize=(15, 4))
    librosa.display.waveshow(y_original, sr=sr)
    plt.title("Original Audio")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.show()

    output_path = "/Users/benjamincolmey/Desktop/audio_predicted.wav"  # Set your desired output path
    # Save the modified audio to the output file
    sf.write(output_path, y_original, sr)   

