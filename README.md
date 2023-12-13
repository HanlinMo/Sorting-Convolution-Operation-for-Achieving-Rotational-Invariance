# Sorting Convolution Operation for Achieving Invariance to Arbitrary Rotations

Hanlin Mo and Guoying Zhao. "Sorting Convolution Operation for Achieving Invariance to Arbitrary Rotations".

Instruction

Inspired by some hand-crafted features of texture images, we propose a Sorting Convolution (SConv) which achieves invariance to arbitrary rotation angles without data augmentation. By substituting all standard convolutions in a CNN model with the corresponding SConv , we can obtain a Sorting Convolutional Neural Network (SCNN). We train SCNN on original MNIST training set without data augmentation, evaluate its performance on MNIST rot test set, and analyze the impact of convolution kernel size, sampling grid, and sorting method on its rotational invariance. In comparison to previous rotation-invariant CNN models, our SCNN achieves state-of-the-art result. We integrate SConv into widely used CNN models and perform classification experiments on popular texture and remote sensing image datasets. Our results show that SConv significantly increases the classification accuracy of these models, particularly when training data is limited.

The papper can be downloaded from: https://arxiv.org/abs/2305.14462

Usage

The code is tested under Pytorch 2.0.0, Python 3.10, and CUDA12.2 on Ubuntu16.04. 

Questions

If you have any questions, please do not hesitate to contact hanlin.mo@oulu.fi.   