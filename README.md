# Multi-scale Convolutional Neural Networks (MCNN) based Side-Channel Analysis
Repository code to support our paper Cryptology ePrint Archive: Report 2020/1134 "Back To The Basics: Seamless Integration of Side-Channel Pre-processing in Deep Neural Networks"

Link to the paper: https://eprint.iacr.org/2020/1134

Authors: Yoo-Seung Won (Nanyang Technological University), Xiaolu Hou (Nanyang Technological University), Dirmanto Jap (Nanyang Technological University), Jakub Breier (Graz University of Technology), and Shivam Bhasin (Nanyang Technological University)

# Datasets
The source code is prepared for four datasets: AES_HD, AES_HD_MM, AES_RD, ASCAD.
The datasets are included to our repository folder.

# Code
Our code has been performed the environment: Tensorflow Version 1.15 and Keras Version 2.1.6.

Each folder has corresponding code (*.py), guessing entropy result (*.npy or *.txt), and trained keras model we saved (*.hdf5).
A few folder don't have a python code.

You can perform our source code by using simple command.

If you want to run BN(org).py file, you type "python BN(org).py" or "python3 BN(org).py".

Note: You should check the trace,plaintext, and save path again to use it, even though we put the proper path in the source code.

# Show the model
Every folder has a trained model (*.hdf5).

If you want to graphically show it, you can download Netron program via https://github.com/lutzroeder/netron.

If you want to load it in your own source code, you can use tf.keras.model.load_model API function.

# Show the Guessing entropy (GE) result
Every folder has a GE result (*.npy).

If you want to load it. you can use numpy.load API function.

# Reference Code
When we have implemented our MCNN SCA, we have referred some source code as below.

https://github.com/KULeuven-COSIC/TCHES20V3_CNN_SCA

https://github.com/AISyLab/EnsembleSCA
