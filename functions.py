import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt 
from scipy.signal import find_peaks
from numpy.fft import fft

def load_yaml(file_name):
    with open(file_name,'r') as file:
     data = yaml.safe_load(file)
    return data



def fft_plot(w1, w2, *args):
   sr = 500
   fft_output_computed = fft(w2.detach().numpy()[0,:,0])
   fft_output_actual = fft(w1.detach().numpy()[0,:,0])
   N_computed = len(fft_output_computed)
   n_computed = np.arange(N_computed)
   T_computed = N_computed/sr
   freq_computed = n_computed/T_computed

   N_actual = len(fft_output_actual)
   n_actual = np.arange(N_actual)
   T_actual = N_actual/sr
   freq_actual = n_actual/T_actual

   peak1,_ = find_peaks(fft_output_computed)
   peak2,_ = find_peaks(fft_output_actual)

   plt.figure(figsize = (5,4))
   plt.plot(freq_computed[peak1], np.abs(fft_output_computed[peak1]), label = args[1]) 
   plt.plot(freq_actual[peak2], np.abs(fft_output_actual[peak2]), label = args[0])
   plt.xlim(0,100)
   plt.ylim(0,100)
   plt.title(args[2])
   plt.xlabel('frequency',fontsize="12")
   plt.ylabel('FFT',fontsize="12")
   plt.xticks(fontsize='9')
   plt.yticks(fontsize = '9')
   plt.grid()
   plt.legend()
   plt.tight_layout()
   return None