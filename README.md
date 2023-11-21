# Machine-Learning-for-Impainting-of-Corrupted-Piano-Audio
I recently digitized some old VHS cassette tapes and stumbled upon a recording of my first piano recital. Unfortunately, the audio was significantly corrupted. To remedy this, I turned to machine learning for audio inpainting, aiming to restore the corrupted sections. The process started by analyzing the original piano recording to identify and characterize silences and gaps. I then pre-processed the piano recording to standardize the gaps sizes, before performing data acquisition to obtain training and testing datasets with similar gaps for a controlled study. 

By employing the Short-Time Fourier Transform (STFT) technique, the audio is transformed into a format conducive to deep learning analysis. I experimented with a variety of loss functions and tested different machine learning models, comparing their effectiveness in inpainting the audio. This endeavor was a fascinating journey into the intersection of digital audio restoration and machine learning. 

A full description of the code can be found here:https://medium.com/@benjamincolmey/using-machine-learning-for-impainting-of-corrupted-piano-audio-11907e121f6e

Original audio:
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/ifMcBhQmidI/0.jpg)](https://www.youtube.com/watch?v=ifMcBhQmidI)

Audio after impainting:
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/7tZy8RvBF5c/0.jpg)](https://www.youtube.com/watch?v=7tZy8RvBF5c)


Original audio waveform with gaps marked:
![alt text](https://github.com/bcolmey/Machine-Learning-for-Impainting-of-Corrupted-Piano-Audio/blob/main/Images/original_audio.jpg)

Train and test data with newly added gaps and relevant contexts shown:
![alt text](https://github.com/bcolmey/Machine-Learning-for-Impainting-of-Corrupted-Piano-Audio/blob/main/Images/train_test_data.jpg)

Spectrogram of first gap and contexts after performing STFT:
![alt text](https://github.com/bcolmey/Machine-Learning-for-Impainting-of-Corrupted-Piano-Audio/blob/main/Images/stft.jpg)

Spectrogram and waveform of the first gap predicted using UNET model:
![alt text](https://github.com/bcolmey/Machine-Learning-for-Impainting-of-Corrupted-Piano-Audio/blob/main/Images/cnn_gap.jpg)

Spectrogram and waveform of the first gap predicted using CNN model:
![alt text](https://github.com/bcolmey/Machine-Learning-for-Impainting-of-Corrupted-Piano-Audio/blob/main/Images/Unet.jpg)

Spectrogram and waveform of the first gap predicted using the encoder-decoder model:
![alt text](https://github.com/bcolmey/Machine-Learning-for-Impainting-of-Corrupted-Piano-Audio/blob/main/Images/encoder_decoder_gap.jpg)

Final audio filled in using model predictions:
![alt text](https://github.com/bcolmey/Machine-Learning-for-Impainting-of-Corrupted-Piano-Audio/blob/main/Images/final_audio.jpg)
