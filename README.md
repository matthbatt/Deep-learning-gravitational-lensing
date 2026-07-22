# Strong Gravitational Lensing Modeling with Deep Learning

<img width="1280" height="854" alt="grav_lensing" src="https://github.com/user-attachments/assets/ab6c138e-7a1a-4f25-956a-df01016b6c42" />

Image credit: https://esahubble.org/images/viewall/?search=gravitational+lens

# Abstract

Gravitational lensing is a powerful probe of the dark matter distribution in galaxies and an independent means of constraining the Hubble constant, $(H_0)$. These applications require accurate inference of lens mass distributions from observed lens images. While traditional approaches based on Markov Chain Monte Carlo (MCMC) sampling are computationally intensive, deep-learning methods enable parameter estimation in milliseconds.
We model lensing galaxies as Singular Isothermal Ellipsoids (SIEs) with external shear and investigate several deep-learning architectures trained on simulated lens images representative of observations from the Hyper Suprimer-Cam (HSC) survey on the Subaru Telescope. The networks are designed to infer five lens parameters: the Einstein radius, the ellipticity components $e_x$ and $e_y$, and the external shear components $\gamma_{\text{ext, 1}}$ and $\gamma_{\text{ext, 2}}$. 
Staring from a standard Residual network (ResNet), we incorporate a U-Net segmentation model to enhance the extraction of lensing features such as arcs and Einstein rings. We further extend the architecture to a Bayesian Neural Network that predicts both parameter estimates and their associated $1\sigma$ uncertainties. The Einstein radius is recovered with high accuracy with 68\% of the predictions showing relative errors between -3\% and +3\%, and the ellipticity components are inferred robustly. In contrast, the external shear parameters remain more challenging to constrain, highlighting the need for further improvements in model architecture and training strategies.

# What does this repository contain?

This repository contains the code used to train and evaluate several deep-learning models for strong gravitational lensing parameter inference.

The main experiments are implemented in four Jupyter notebooks (`.ipynb` files), each dedicated to a different model architecture:

- **U-Net**: used for lensing feature extraction and image segmentation.
- **U-Net + ResNet**: combines image segmentation with a residual network for lens parameter prediction.
- **Vision Transformer (ViT)**: explores transformer-based architectures for lensing parameter inference.
- **Bayesian Neural Network (BNN)**: provides parameter predictions together with associated uncertainties.

The model architectures are defined in the Python files:

- `deep_learning_models.py`: contains the different neural network architectures.
- `process_training.py`: contains utility functions used for training, validation, and model optimization.

The repository also includes scripts and notebooks used to reproduce the experiments presented in the paper, including model training, evaluation, and performance analysis.

# How to run the models ?

The models are trained using one GPU and three CPUs.

# Hyperparameters

All models are trained using a similar setup. The number of training epochs is set to 100, with a batch size of 32. The learning rate is initialized at $10^{-3}$ and is adjusted during training using a cosine annealing scheduler. Early stopping is triggered when the validation loss does not improve for 20 consecutive epochs. Prior to training, the images are preprocessed by setting all negative pixel values to zero and applying a square-root transformation.