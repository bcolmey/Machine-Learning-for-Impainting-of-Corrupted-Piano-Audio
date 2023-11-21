# Machine-Learning-for-Impainting-of-Corrupted-Piano-Audio
I recently digitized some old VHS cassette tapes and stumbled upon a recording of my first piano recital. Unfortunately, the audio was significantly corrupted. To remedy this, I turned to machine learning for audio inpainting, aiming to restore the corrupted sections. The process started by analyzing the original piano recording to identify and characterize silences and gaps. I then pre-processed the piano recording to standardize the gaps sizes, before performing data acquisition to obtain training and testing datasets with similar gaps for a controlled study. 

By employing the Short-Time Fourier Transform (STFT) technique, the audio is transformed into a format conducive to deep learning. I experimented with a variety of loss functions and tested different machine learning models, comparing their effectiveness in inpainting the audio. This endeavor was a fascinating journey into the intersection of digital audio restoration and machine learning. 

A full description of the code can be found here:https://medium.com/@benjamincolmey/using-machine-learning-for-impainting-of-corrupted-piano-audio-11907e121f6e


Original audio:

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/ifMcBhQmidI/0.jpg)](https://www.youtube.com/watch?v=ifMcBhQmidI)


Audio after impainting:

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/7tZy8RvBF5c/0.jpg)](https://www.youtube.com/watch?v=7tZy8RvBF5c)


Original audio waveform with gaps marked:
<img src="https://github.com/bcolmey/Machine-Learning-for-Impainting-of-Corrupted-Piano-Audio/blob/main/Images/original_audio.jpg" width="700" height="500">

Train and test data with newly added gaps and relevant contexts shown:
<img src="https://github.com/bcolmey/Machine-Learning-for-Impainting-of-Corrupted-Piano-Audio/blob/main/Images/train_test_data.jpg" width="700" height="500">


Spectrogram of first gap and contexts after performing STFT:

<img src="https://github.com/bcolmey/Machine-Learning-for-Impainting-of-Corrupted-Piano-Audio/blob/main/Images/stft.jpg" width="400" height="400">


Spectrogram and waveform of the first gap predicted using UNET model:

<img src="https://github.com/bcolmey/Machine-Learning-for-Impainting-of-Corrupted-Piano-Audio/blob/main/Images/Unet.jpg" width="800" height="600">


Spectrogram and waveform of the first gap predicted using CNN model:

<img src="https://github.com/bcolmey/Machine-Learning-for-Impainting-of-Corrupted-Piano-Audio/blob/main/Images/cnn_gap.jpg" width="800" height="200">


Spectrogram and waveform of the first gap predicted using the encoder-decoder model:

<img src="https://github.com/bcolmey/Machine-Learning-for-Impainting-of-Corrupted-Piano-Audio/blob/main/Images/encoder_decoder_gap.jpg" width="800" height="200">


Final audio filled in using model predictions:

<img src="https://github.com/bcolmey/Machine-Learning-for-Impainting-of-Corrupted-Piano-Audio/blob/main/Images/final_audio.jpg" width="700" height="500">

