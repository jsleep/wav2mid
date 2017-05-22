# wav2mid: Polyphonic Piano Music Transcription with Deep Neural Networks

### Thesis by Jonathan Sleep for MS in CSC @ CalPoly

## Abstract / Intro
There has been a multitude of recent research on using deep learning for music & audio generation and classification. In this paper, we plan to build on these works by implementing a novel system to automatically transcribe polyphonic music with an artificial neural network model. We show that by treating the transcription problem as an image classification problem we can use transformed audio data to predict the group of notes currently being played.

## Background
Digital Signal Processing: Fourier Transform, STFT, Constant-Q, Onset/Beat Tracking, Auto-correlation
Machine Learning: Artificial Neural Networks, Convolutional Neural Networks, Recurrent Neural Networks

## Related Work on AMT
*  Pre-Deep Learning Research
    * [Non-negative matrix factorization for polyphonic music transcription](http://ieeexplore.ieee.org/abstract/document/1285860/)
        * Really cool paper for transcribing music using NMF - very simple. I wish there were more results shown with metrics like accuracy, but the work seemed clear. It would be cool to see if/how I could extend this.
    * [YIN, a fundamental frequency estimator for speech and music](asa.scitation.org/doi/abs/10.1121/1.1458024) - building off autocorrelation which produces an f0 estimator with even less error.

* Research that use Deep Learning
    * [Modeling Temporal Dependencies in High-Dimensional Sequences: Application to Polyphonic Music Generation and Transcription](https://arxiv.org/abs/1206.6392) - Using a sequential model to aid in transcription.
    * [An End-to-End Neural Network for Polyphonic Piano Music Transcription](https://arxiv.org/abs/1508.01774) - Research on AMT that used an acoustic and language model, ~75% accuracy on MAPS
    * [On the Potential of Simple Framewise Approaches to Piano Transcription](https://arxiv.org/abs/1612.05153) - explains the current state-of-the-art and what the most effective architectures and input representations are for framewise transcription.
    * [An Experimental Analysis of the Entanglement Problem in Neural-Network-based Music Transcription Systems](https://arxiv.org/abs/1702.00025) - explains entanglement, which is the problem of learning to generalize note combinations that it may have not been trained with. Entanglement is the current glass ceiling problem for framewise neural network music transcription. They present a few (really just one) possible solutions that I could try to implement (a loss function that takes entanglement into account).

* Products
    * [Melodyne](http://www.celemony.com/en/melodyne/what-is-melodyne) - Popular plugin for transcription + pitch correction, costs up to $500
    * [AnthemScore](https://www.lunaverus.com/cnn) - A product for Music Transcription that uses deep learning.

## Design
The design for the system is as follows:
* Pre-process our data into an ingestible format, fourier-like transform of the audio and piano-roll conversion of midi files.
* Design a neural network model to estimate current notes from audio data
* Use frame-wise (simpler) or onsets (faster)
* Train on a large corpus of audio to midi
* Evaluate it's performance on audio/midi pairs we have not trained on

## Implementation
### Libraries
* Python - due to the abundance of music and machine learning libraries developed for it
* librosa - for digital signal processing methods
* pretty_midi - for midi manipulation methods
* TensorFlow - for neural networks

## Data
* [MAPS dataset](http://www.tsi.telecom-paristech.fr/aao/en/2010/07/08/maps-database-a-piano-database-for-multipitch-estimation-and-automatic-transcription-of-music/)
* [Large MIDI collection](https://www.reddit.com/r/WeAreTheMusicMakers/comments/3ajwe4/the_largest_midi_collection_on_the_internet/)
