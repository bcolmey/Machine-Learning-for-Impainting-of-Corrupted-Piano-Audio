# Machine-Learning-for-Impainting-of-Corrupted-Piano-Audio
I recently digitized some old VHS cassette tapes and stumbled upon a recording of my first piano recital. Unfortunately, the audio was significantly corrupted. To remedy this, I turned to machine learning for audio inpainting, aiming to restore the corrupted sections. The process started by analyzing the original piano recording to identify and characterize silences and gaps. I then pre-processed the piano recording to standardize the gaps sizes, before creating training and testing datasets, artificially augmenting them with similar gaps for a controlled study. The training data was taken from hours of youtube videos of children's first recitals, which predictably altered the youtube recommendation algorithm. 

By employing the Short-Time Fourier Transform (STFT) technique, the audio is transformed into a format conducive for deep learning analysis. I experimented with a variety of loss functions and tested different machine learning models, comparing their effectiveness in inpainting the audio. This endeavor was a fascinating journey into the intersection of digital audio restoration and machine learning. A full description of the code can be found here:https://medium.com/@benjamincolmey/unveiling-the-aharonov-bohm-effect-simulating-quantum-electron-wavepacket-dynamics-from-scratch-in-7c54fea2193d

[![IMAGE ALT TEXT]([https://github.com/bcolmey/Machine-Learning-for-Impainting-of-Corrupted-Piano-Audio/Images
/youtube.png?raw=true "Title"])(https://www.youtube.com/watch?v=ifMcBhQmidI "Ben first piano recital with corrupted audio")


[<img src="https://github.com/bcolmey/Machine-Learning-for-Impainting-of-Corrupted-Piano-Audio/Images/youtube.png)" width="50%">]([https://www.youtube.com/watch?v=Hc79sDi3f0U "Now in Android: 55"](https://www.youtube.com/watch?v=ifMcBhQmidI "Ben first piano recital with corrupted audio"))
