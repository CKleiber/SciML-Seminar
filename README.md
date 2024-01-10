# SciML-Seminar
Repository of my seminar talk about variational autoencoders (VAE) in January 2024.

## Autoencoder.ipynb
This is the first file, where a very basic autoencoder is used to reproduce the two moons dataset. Furthermore, a convolutional autoencoder is applied on the MNIST dataset. The 2D latent space is visualised for each label, showing the irregularities of the autoencoder latent space. When trying to generate data, some problems occur, which are then treated in the next notebook.

## VariationalAutoencoder.ipynb
Here, the same datasets are used as above. Only the autoencoder is expanded to a VAE. The trade-off of reconstuction and generation can be observed in both datasets. The gaps in the latent space of the autoencoder are not appearing for the VAE, creating higher quality synthetic data. 

## PlayingWithVAE.ipynb
Notebook I used to play around with the concept of the VAE. I used a comic face dataset to train a quite large VAE to test the interpolation and reconstruction of more complicated data than e.g. MNIST.

### interpolate.py
Helper function for interpolation of two or more datapoints. 

### LatentspaceExplorer.py
Helper for a interface to explore the effect of each individual latent space dimension in the cartoon example. 

### requirements.txt
Includes all the requited packages to run the code. Indluces also some redundant packages like Tensorflow, which are not needed, but still are in the venv somehow.

Use `pip install -r requirements.txt` to install the requirements (for Windows 10).

### models
Folder where all the pretrained models and their respective loss histories are saved.

### gifs
Folder for saving interpolation gifs.


