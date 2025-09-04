# Neural-Reconstruction-of-STN-LFP-Signals-via-Rossler-Dynamics-and-Deep-Networks

## Overview
This project investigates the relationship between **chaotic dynamical systems (Rossler oscillator networks)** and **neural signals (Local Field Potentials, LFPs)**. By combining Rossler-based synthetic oscillations with deep learning architectures (Feed-forward, LSTM, and Flip-Flop networks), the work evaluates whether chaotic oscillatory dynamics can model and reconstruct real brain signals.

The LFP recordings used here are from **Parkinson’s disease (PD)** patients, where abnormal oscillatory activity is a hallmark. A particularly important phenomenon is **β Synchronization (13–30 Hz)** — an exaggerated beta-band activity that is strongly linked to motor impairments in PD.

The ultimate aim is to provide insights into:
- Computational neuroscience models of oscillatory activity.
- Neural signal reconstruction and prediction.
- Identification and modelling of disease-specific **biomarkers** (e.g., beta synchronization in PD).
- Potential applications in Brain–Computer Interfaces (BCIs) for motor intent decoding and assistive technologies.

## Project Structure
- `main.py`: Main experimental pipeline. Handles simulation, preprocessing, training, and evaluation.
- `rossler_network.py`: Implements the Rossler oscillator system and Hermitian coupling matrix.
- `lfp_process.py`: Preprocesses Local Field Potential (LFP) recordings (from MATRIX_DBS.mat).
- `nn_model.py`: Defines Feed-forward and LSTM neural network architectures.
- `flipflop_nn.py`: Defines the Flip-Flop recurrent neural network model (TensorFlow-based).
- `train_model.py`: Contains training routines with MSE loss and Adam Optimizer.
- `functions.py`: Utility functions for FFT plotting, preprocessing, and YAML handling.
- `parameters.yaml`: Stores all simulation, network, and training hyperparameters.
- `model_training.ipynb`: Jupyter notebook for interactive experiments and visualisation.
- `Flip-flop model/`: Folder containing the Flip-Flop RNN implementation and experiments.

## Install Dependencies
Install the required packages via pip:
```bash
pip install torch numpy scipy matplotlib pyyaml tqdm hdf5storage tensorflow
```

## Data Processing Pipeline
### Rossler Network Simulation
- Generates synthetic chaotic oscillatory data using coupled Rossler equations.
- Incorporates Hermitian complex-weight coupling for network interactions.
- Outputs `x, y, z` oscillator dynamics.

### LFP Signal Preprocessing
- Loads experimental neural recordings from `MATRIX_DBS.mat`.
- Applies downsampling and low-pass filtering **(butter_lowpass)**.
- Normalizes the data into tensor form suitable for training.

### Feature Construction
- Rossler network outputs and LFP recordings are aligned into training, testing, and whole segments.
- FFT analysis enables frequency-domain comparison between synthetic and experimental signals.

### Model Training
- Neural networks are trained to map Rossler signals → LFP signals.
- Architectures supported:
  - Feed-forward NN (multi-layer dense network).
  - LSTM NN (temporal recurrent model).
  - Flip-Flop RNN (custom recurrent architecture for temporal memory).
- Loss function: MSE, Optimizer: Adam.

## Methodology
### Training Procedure
The training loop optimizes models to minimize reconstruction error between Rossler-generated input and LFP target signals.
```python
from main import run_code

# Train with Feed-forward network
results = run_code("Feed-forward")

# Train with LSTM network
results = run_code("LSTM model")
```

For the Flip-Flop model (TensorFlow-based):
```python
from flipflop_nn import flip_flop, training_mode

model = flip_flop(output_dim, output_size, timesteps, features, num_epochs, learning_rate).forward()
result = training_mode(model, data, target, batch_num, learning_rate, num_epochs)
```

### Signal Analysis
- **Time-domain**: Direct comparison of Rossler vs LFP segments.
- **Frequency-domain**: FFT reveals shared oscillatory components, including the beta-band peak (13–30 Hz) characteristic of PD patients.
- **Whole reconstruction**: Models tested on entire signal duration.

## Plots and Visualisations
### Feed-forward & LSTM Models

#### Time-domain Reconstruction
Visualizes how well the neural model maps Rossler inputs to experimental LFP outputs.

#### Training phase
<img width="1214" height="312" alt="Image" src="https://github.com/user-attachments/assets/4d963387-a2b7-455b-8776-e61080195539" />

#### Testing phase
<img width="1214" height="312" alt="Image" src="https://github.com/user-attachments/assets/3f3cfeee-2418-4fc9-aa7c-67393de1edea" />

#### Frequency-domain Analysis
FFT plots comparing synthetic and biological signals highlight overlapping spectral peaks.
- A notable observation is the beta synchronization peak (13–30 Hz) in LFP signals, a phenomenon linked to Parkinson’s disease pathology.
- Capturing this feature in reconstructed signals validates both the model and its ability to preserve clinically relevant biomarkers.

<img width="490" height="390" alt="Image" src="https://github.com/user-attachments/assets/686e5aa7-0093-497e-afc4-52617471d191" />

#### Whole-signal Evaluation
- Entire reconstructed LFP signals plotted alongside original LFP recordings.

<img width="990" height="490" alt="Image" src="https://github.com/user-attachments/assets/fb982f54-9524-43d4-980e-8f4b550dddf4" />
