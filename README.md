**Epileptic Seizures Detection Using Fusion Of Artificial Neural Network With Hybrid Deep Learning**
This project motive was to detect epileptic seizures in patients. A public dataset called CHB-MIT dataset was used in this project. This dataset has 23 patients of varying ages.

The algorithms that were used in this project are Convulutional neural network, Gated Recurrent Unit and Artificial Neural Network.

Dataset contained files in edf format. These files were converted into EEG signals by using MNE python library. These EEG signals were subjected to data preprocessing and feature extraction. Then that data was given to hybrid deep learning algorithms that is CNN-GRU.
CNN extracted spatial features i.e., which are effective in identifying patterns in different regions of brain. GRU extracts temporal features which are helpful in capturing temporal dependencies over time.

Then the output from the CNN-GRU were given to Artificial neural network which predicts the seizure in a patient by using inpute, hidden and output layers.
