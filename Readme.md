# Convolutional Neural Networks with analytically determined Filters

This is the code used for running the experiments in our paper "Convolutional Neural Networks with analytically determined Filters" (Matthias Kissel and Klaus Diepold). The paper was accepted at the International Joint Conference on Neural Networks (IJCNN 2022) featured by the IEEE World Congress on Computational Intelligence 2022. 

## File Overview

The files in this repository are organized as follows.

- benchmark.py: This script benchmarks the models introduced in the paper. It shows how the models can be used. 
- ConvolutionalNeuralNetwork.py implements a standard convolutional neural network used as baseline in the paper.
- PseudoRegressorKeras.py implements the Convolutional Neural Network, which is trained using the algorithm proposed in the paper. The name is given by the pseudo-inverse, which plays a major role during training.
- KMeans_ELM.py implemnets a Convolutional Neural Network which is trained using k-means clustering (i.e. an unsupervised learning method).
- RandomPatch_ELM.py implements a Convolutional Neural Network with random convolutional filters, whereas the last layer is trained using the standard Extreme Learning Machine algorithm.

## Requirements

Several packages are required for using the provided code (for example tensorflow, numpy, etc.). The full requirements are listed in requirements.txt. We recomment using a virtual environment / conda environment:

	conda create --name analyticalconvfilters python=3.6

After activating the environment

	conda activate analyticalconvfilters

you can install the required packages by running

	python3 -m pip install -r requirements.txt

Finally, you can start the benchmark using the command

	python3 benchmark.py
