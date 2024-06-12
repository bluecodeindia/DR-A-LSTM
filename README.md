# DR-A-LSTM Repository

## Overview
This repository contains various Python scripts and folders for implementing and evaluating a Dimension Reduction Autoencoder Long Short-Term Memory (DR-A-LSTM) model for landslide movement prediction. The project involves using autoencoders, LSTM models, Principal Component Analysis (PCA), and t-Distributed Stochastic Neighbor Embedding (t-SNE) for dimension reduction and classification.

## Repository Structure
- `encoder/`: Contains scripts related to the encoder models.
- `final/`: Final versions of the model scripts.
- `finalmodel/`: Final model implementation scripts.
- `model1/`: First set of model scripts.
- `model2/`: Second set of model scripts.
- `prefinal/`: Pre-final versions of the model scripts.
- `premodel/`: Preliminary model scripts.
- `weights/`: Folder containing model weights.
- `weights2-outputlstm/`: Additional model weights.

### Python Scripts
- `Auto Data.py`: Script for data preprocessing and handling.
- `Auto-LSTM.py`: LSTM model with automated data handling.
- `Autoencoder.py`: Implementation of a basic autoencoder.
- `Conv-Autoencoder.py`: Implementation of a Convolutional Autoencoder.
- `DR-A-LSTM.py`: Implementation of the Dimension Reduction Autoencoder LSTM.
- `GAN-Autoencoder-Noise.py`: GAN-based autoencoder with noise handling.
- `GAN-Autoencoder.py`: Implementation of a GAN-based Autoencoder.
- `LSTM-Autoencoder-Noisy.py`: LSTM Autoencoder with noise handling.
- `LSTM-Autoencoder.py`: Implementation of an LSTM Autoencoder.
- `LSTM_Classification-Autoencoder.py`: LSTM classification with autoencoder.
- `LSTM_Classification-PCA.py`: LSTM classification with PCA.
- `LSTM_Classification.py`: Implementation of LSTM classification.
- `MLP-LSTM.py`: Implementation of an MLP combined with LSTM.
- `Prediction LSTM Autoencoder.py`: Prediction using LSTM Autoencoder.
- `t-SNE Landslide.py`: t-SNE visualization for landslide data.

## Getting Started

### Prerequisites
Ensure you have the following libraries installed:
- TensorFlow
- Keras
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

You can install them using pip:
```bash
pip install tensorflow keras torch numpy pandas scikit-learn matplotlib seaborn
```

### Running the Scripts
1. **Clone the Repository**:
    ```bash
    git clone https://github.com/bluecodeindia/DR-A-LSTM.git
    cd DR-A-LSTM
    ```

2. **Run the Scripts**:
    Use Python to run the scripts. For example, to run the `DR-A-LSTM.py` script:
    ```bash
    python DR-A-LSTM.py
    ```

## Scripts Explanation

### Autoencoders and Dimension Reduction
- `Autoencoder.py`, `Conv-Autoencoder.py`, `GAN-Autoencoder.py`, `LSTM-Autoencoder.py`: Implementations of various autoencoder models for dimension reduction.
- `t-SNE Landslide.py`: Uses t-SNE for visualizing the reduced dimensions of the landslide data.

### LSTM Models
- `Auto-LSTM.py`, `LSTM_Classification.py`, `Prediction LSTM Autoencoder.py`: Implementations of LSTM models for classification and prediction tasks.

### PCA
- `LSTM_Classification-PCA.py`: Uses PCA for dimension reduction before LSTM classification.

### Combining Models
- `MLP-LSTM.py`: Combines Multi-Layer Perceptron (MLP) with LSTM for enhanced predictions.

## Contributing
Feel free to contribute by submitting a pull request. Please ensure your changes are well-documented and tested.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For any questions or inquiries, please contact [bluecodeindia@gmail.com](mailto:bluecodeindia@gmail.com).
