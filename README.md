# Neural-Reconstruction-of-STN-LFP-Signals-via-Rossler-Dynamics-and-Deep-Networks

## Overview
This project investigates the relationship between **chaotic dynamical systems (Rossler oscillator networks)** and **neural signals (Local Field Potentials, LFPs)**. By combining Rossler-based synthetic oscillations with deep learning architectures (Feed-forward, LSTM, and Flip-Flop networks), the work evaluates whether chaotic oscillatory dynamics can model and reconstruct real brain signals.

The LFP recordings used here are from **Parkinson’s disease (PD)** patients, where abnormal oscillatory activity is a hallmark. A particularly important phenomenon is **β synchronisation (13–30 Hz)** — an exaggerated beta-band activity that is strongly linked to motor impairments in PD.

The ultimate aim is to provide insights into:
- Computational neuroscience models of oscillatory activity.
- Neural signal reconstruction and prediction.
- Identification and modelling of disease-specific **biomarkers** (e.g., beta synchronisation in PD).
- Potential applications in Brain–Computer Interfaces (BCIs) for motor intent decoding and assistive technologies.

## Project Structure
- `main.py`: Main experimental pipeline. Handles simulation, preprocessing, training, and evaluation.
- `rossler_network.py`: Implements the Rossler oscillator system and Hermitian coupling matrix.
- `lfp_process.py`: Preprocesses Local Field Potential (LFP) recordings (from MATRIX_DBS.mat).
- `nn_model.py`: Defines Feed-forward and LSTM neural network architectures.
- `flipflop_nn.py`: Defines the Flip-Flop recurrent neural network model (TensorFlow-based).
- `train_model.py`: Contains training routines with MSE loss and Adam optimiser.
- `functions.py`: Utility functions for FFT plotting, preprocessing, and YAML handling.
- `parameters.yaml`: Stores all simulation, network, and training hyperparameters.
- `model_training.ipynb`: Jupyter notebook for interactive experiments and visualisation.
- `Flip-flop model/`: Folder containing the Flip-Flop RNN implementation and experiments.
