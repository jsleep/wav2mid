# wav2mid: Automatic Music Transcription with Deep Neural Networks

### Thesis by Jonathan Sleep for MS in CSC @ CalPoly

## Abstract / Intro
There has been a multitude of recent research on using deep learning for music & audio generation and classification. In this paper, we plan to build on these works by implementing a novel system to automatically transcribe music with an artificial neural network model. We show that by treating the transcription problem as an image classification problem we can use transformed audio data to predict the group of notes currently being played.

## Background
Digital Signal Processing: Fourier Transform, STFT, Constant-Q, Onset, Beat Tracking
Machine Learning: Artificial Neural Networks, Convolutional Neural Networks, Recurrent Neural Networks

## Related Work on AMT
* Pre-Deep Learning Research / Product
    * [Melodyne](http://www.celemony.com/en/melodyne/what-is-melodyne) - Popular plugin for transcription + more that costs up to $500
* Research / Product that use Deep Learning
    * [AnthemScore](https://www.lunaverus.com/cnn) - A product for Music Transcription
    * [An End-to-End Neural Network for Polyphonic Piano Music Transcription](https://arxiv.org/abs/1508.01774) - Research on AMT that used an acoustic and language model, ~75% accuracy on MAPS

## Design
The design for the system is as follows:
* Pre-process our data into an ingestible format, fourier-like transform of the audio and piano-roll conversion of midi files.
* Design a neural network model to estimate current notes from audio data
* Train on a large corpus of audio to midi
* Evaluate it's performance on audio/midi pairs we have not trained on

## Implementation
* Python - due to the abundance of music and machine learning libraries developed for it
* librosa - for digital signal processing methods
* pretty_midi - for midi manipulation methods
* TensorFlow - for deep learning

## Data
* MAPS dataset
* Large MIDI collection - which can be synthesized to audio (sorta cheating)
