# Machine-Learning-for-Impainting-of-Corrupted-Piano-Audio
I recently digitized some old VHS cassette tapes and stumbled upon a recording of my first piano recital. Unfortunately, the audio was significantly corrupted. To remedy this, I turned to machine learning for audio inpainting, aiming to restore the corrupted sections. The process started by analyzing the original piano recording to identify and characterize silences and gaps. I then pre-processed the piano recording to standardize the gaps sizes, before creating training and testing datasets, artificially augmenting them with similar gaps for a controlled study. The training data was taken from hours of youtube videos of children's first recitals, which predictably altered the youtube recommendation algorithm. 

By employing the Short-Time Fourier Transform (STFT) technique, the audio is transformed into a format conducive for deep learning analysis. I experimented with a variety of loss functions and tested different machine learning models, comparing their effectiveness in inpainting the audio. This endeavor was a fascinating journey into the intersection of digital audio restoration and machine learning. A full description of the code can be found here:

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/ifMcBhQmidI/0.jpg)]([https://www.youtube.com/watch?v=ifMcBhQmidI)


![alt text](https://github.com/bcolmey/Machine-Learning-for-Impainting-of-Corrupted-Piano-Audio/blob/main/Images/original_audio.jpg)

![alt text](https://github.com/bcolmey/Machine-Learning-for-Impainting-of-Corrupted-Piano-Audio/blob/main/Images/train_test_data.jpg)

![alt text](https://github.com/bcolmey/Machine-Learning-for-Impainting-of-Corrupted-Piano-Audio/blob/main/Images/stft.jpg)

![alt text](https://github.com/bcolmey/Machine-Learning-for-Impainting-of-Corrupted-Piano-Audio/blob/main/Images/cnn_gap.jpg)

![alt text](https://github.com/bcolmey/Machine-Learning-for-Impainting-of-Corrupted-Piano-Audio/blob/main/Images/Unet.jpg)

![alt text](https://github.com/bcolmey/Machine-Learning-for-Impainting-of-Corrupted-Piano-Audio/blob/main/Images/encoder_decoder_gap.jpg)

![alt text](https://github.com/bcolmey/Machine-Learning-for-Impainting-of-Corrupted-Piano-Audio/blob/main/Images/smoothed.jpg)

![alt text](https://github.com/bcolmey/Machine-Learning-for-Impainting-of-Corrupted-Piano-Audio/blob/main/Images/final_audio.jpg)
