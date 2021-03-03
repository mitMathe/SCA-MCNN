# Multi-scale Convolutional Neural Networks (MCNN) based Side-Channel Analysis
Repository code to support our paper Cryptology ePrint Archive: Report 2020/1134 "Back To The Basics: Seamless Integration of Side-Channel Pre-processing in Deep Neural Networks"

Link to the paper: https://eprint.iacr.org/2020/1134

Authors: Yoo-Seung Won (Nanyang Technological University), Xiaolu Hou (Nanyang Technological University), Dirmanto Jap (Nanyang Technological University), Jakub Breier (Graz University of Technology), and Shivam Bhasin (Nanyang Technological University)

# Datasets
The source code is prepared for four datasets: AES_HD, AES_HD_MM, AES_RD, ASCAD.
The datasets are included to our repository folder.

# Code
Our code has been performed the below environment.
Tensorflow Version 1.15
Keras Version 2.1.6

Each folder has corresponding code (*.py), guessing entropy result (*.npy or *.txt), and keras model we saved (*.hdf5).
A few folder don't have a python code.

Note: You should check the trace and plaintext path again to use it, even though we put the proper path in the source code.
