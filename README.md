# Anime VAE
Small Tkinter app using a VAE to produce anime faces.

A Variational Auto-Encoder has been trained on multiple anime faces using this
[dataset](https://www.kaggle.com/splcher/animefacedataset).

From this, the decoder is able to produce images from random points in the latent space.
The tkinter application is an friendly interface to generate random images.
The application also propose to modify the first principal components in the latent space
to see the implications on the decoded images.
