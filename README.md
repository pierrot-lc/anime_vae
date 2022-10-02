# Anime VAE
Small tkinter app using a VAE to produce anime faces.

A Variational Auto-Encoder has been trained on multiple anime faces using this
[dataset](https://www.kaggle.com/splcher/animefacedataset).

From this, the decoder is able to produce images from random points in the latent space.
The tkinter application is an friendly interface to generate random images.
The application also propose to modify the first principal components in the latent space
and see the result on the decoded images.

The VAE model isn't stored in the repo as it is heavy (33M params).

## TODO
* Model store with git?
* Train a better model
* Add images to this README
* wandb link
* Add more transformations to the training images
* Tkinter app with a 4K display
* Save model to wandb during training
