import hdf5storage
import numpy as np
from scipy.signal import butter, lfilter
import torch
import tensorflow as tf

def butter_lowpass(cutoff, fs, order=20):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff=50, fs=500, order=20):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def lfp_data(mode, t_end , patient_num,  file = 'MATRIX_DBS.mat'):
    mat=hdf5storage.loadmat(file)
    sampling_rate_data=mat['MATRIX_DBS']['fs'].flatten()[0].flatten()
    signal_base_data=mat['MATRIX_DBS']['signal_base'].flatten()[0].flatten()
    signal_dbs_data=mat['MATRIX_DBS']['signal_dbs'].flatten()[0].flatten()

    sampling_rate=sampling_rate_data[patient_num]
    base_signal=signal_base_data[patient_num]
    base_signal=base_signal.flatten()

    sr=500
    if mode == 'training':
      end = t_end*sr
      data_signal=base_signal[0 : t_end*sampling_rate]
    elif mode == 'testing':
      end = t_end*sr
      data_signal=base_signal[t_end*sampling_rate : 2*t_end*sampling_rate]
    elif mode == 'whole':
      end = 2*t_end*sr
      data_signal=base_signal[0 : 2*t_end*sampling_rate]
    if sampling_rate == 2048:
         i=4
    else:
         i = 8
    reduced_arr = np.mean(data_signal.reshape(-1,i) , axis=1)
    reduced_data=np.array(reduced_arr[0:end])
    filtered_data = butter_lowpass_filter(reduced_data)
    target_data=torch.tensor(filtered_data,dtype=torch.float)
    processed_data = target_data.clone().detach().unsqueeze(0).unsqueeze(2)
    final_data = (processed_data - processed_data.min())/(processed_data.max()-processed_data.min())
    return tf.convert_to_tensor(final_data)

def reshape(file , mode):
    
    if mode == 'train_lfp':
        new_file = tf.reshape(file, [-1,500,1])
    elif mode == 'train_data':
        new_file = tf.reshape(file, [-1,500,5])    
    elif mode == 'test':
        new_file = tf.reshape(file, [1,-1, 1])  
    else:
        print('Cannot process')

    return new_file          