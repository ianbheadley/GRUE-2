# GRUE-2: Explorations in Feature (Concept) Detection and Extraction

This repository contains a series of experiments and tools developed to understand how neural networks encode information, starting from a very simple question: **"Can we subtract models?"**

Through trial, error, and empirical testing—mainly utilizing Small CNNs trained on CIFAR-10/100—this project evolved into a deep dive into latent space geometry, supervised concept extraction, and eventually unsupervised feature discovery. 

## Key Learnings & Observations

These experiments were instrumental in shaping an understanding of supervised vs. unsupervised learning and how concepts naturally coalesce within model weights:

1. **Features over Labels:** AI systems will learn *all* features of the dataset, regardless of how they are labeled. The loss function and labeling simply act as scaffolding to organize the output layer and make it easier to retrieve information. The network models reality; the labels just point to it.
2. **The Power of the Latent Space:** Augmenting labels can make minor differences in training dynamics, but inherently learned features are incredibly robust and will often override a label if the feature is a stronger predictive shortcut.
3. **Supervised Concept Extraction:** By identifying a specific feature visually in our training set (e.g., "Sky" or "Ocean"), we can label a small subset and compute a direct geometric vector in the activation space. This supervised vector can then be used to identify where that concept heavily influences the model across unseen data.
4. **Blind/Unsupervised Feature Discovery:** We can go a step further and abandon labels entirely. Because coherent concepts form distinct, non-Gaussian clusters in high-dimensional space, we can search for particular distribution bands (using Independent Component Analysis, Excess Kurtosis, Bimodality Coefficients, and KL-Divergence). This allows us to mathematically sift the latent space and discover features the network learned *completely unsupervised*.

## Project Overview

- **Model Subtraction & Arithmetic:** Initial tests on whether taking the literal weight differences of networks trained on slightly different datasets (e.g., one missing a specific color or class) yields a coherent "concept model."
- **Latent Concept Vectors:** Scripts to extract and measure Mean-Difference vectors in the feature space (`fc1` activations) for user-defined subsets.
- **Kurtosis & ICA Discovery:** Advanced pipelines using `scipy` and `sklearn` to run FastICA over activations, sorting the resulting axes by their non-Gaussianity to dynamically unearth hidden visual concepts (like lighting, backgrounds, or specific structural shapes) without any prior human labeling.
- **VLM Auto-Naming:** Integration with local Vision-Language Models (like Gemma) to automatically contrast the top and bottom activating images for a discovered latent axis and assign a human-readable name to the raw mathematical feature.

These tools demonstrate that neural networks are less "black box classifiers" and more "topographical reality modelers."
