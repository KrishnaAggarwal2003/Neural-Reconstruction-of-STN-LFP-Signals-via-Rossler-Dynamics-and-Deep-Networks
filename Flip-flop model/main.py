from functions import load_yaml
import numpy as np
from rossler_network import RosslerSystem, ross_preprocess, hermitian_matrix
from lfp_process import lfp_data, reshape
from flipflop_nn import flip_flop, training_mode
import matplotlib.pyplot as plt

arguements = load_yaml('parameters.yaml')

patient_num = arguements['patient_num']
t_end = arguements['t_end']

# Model Parameters
lr = arguements['learning_rate'] 
output_dim = arguements['output_dim']
output_size = arguements['output_size']
num_epochs = arguements['num_epochs']
timesteps = arguements['timesteps']

# Rossler Parameters
N = arguements['num_osc']
const = arguements['const']
a = arguements['a']
b = arguements['b']
c = arguements['c']
d = arguements['d']
k = arguements['k']
Iext = arguements['Iext']
num_points = arguements['num_points']
t_span = (0,2200)
omega = np.random.uniform(0.8,1.3, size=(N,1))

def run_code():
    print('**********Running Rossler Network****************')
    rossler_osc = RosslerSystem(N, complex_wts=hermitian_matrix(N), omega=omega, a=a, b=b, c=c, const=const, d=d, Iext=Iext, k=k,
                                t_span=t_span, num_points=num_points)
    x, y, z = rossler_osc.solve()
    print('***********************Network running completed***************************')
    print('***********************Loading LFP and Rossler Data************************\n')
    train_lfp = lfp_data(mode='training', t_end=t_end, patient_num=patient_num)
    test_lfp = lfp_data(mode='testing', t_end=t_end, patient_num=patient_num)
    whole_lfp = lfp_data(mode='whole', t_end=t_end, patient_num=patient_num)

    train_data = ross_preprocess(x, mode='training', t_end=t_end)
    test_data = ross_preprocess(x, mode='testing', t_end=t_end)
    whole_data = ross_preprocess(x, mode='whole', t_end=t_end)
    
    print('****************************Loading Completed, Initiating FF_Model***************************************')
    ff_class = flip_flop(output_dim=output_dim , output_size=output_size , timesteps=timesteps, features=N, 
                      num_epochs=num_epochs, learning_rate=lr)
    print('**********************Loading the model, Initiating Training phase**************************')
    
    new_train_data = reshape(train_data, mode='train_data')
    new_train_lfp = reshape(train_lfp, mode='train_lfp')

    ff_model = ff_class.forward()
    print(ff_model.summary())

    training_mode(ff_model, new_train_data, new_train_lfp, batch_num = new_train_data.shape[0],
                  num_epochs= num_epochs, learning_rate=lr)

    keys = ['Model', 'train_lfp', 'test_lfp', 'whole_lfp', 'train_data', 'test_data', 'whole_data']
    values = [ff_model, train_lfp, test_lfp, whole_lfp, train_data, test_data, whole_data]

    output = dict(zip(keys, values))
    return output


def signal_plot(w1, w2, *args, t_end=t_end):
    """
    Plot signals w1 and w2 over time.

    Parameters:
    w1 (torch.Tensor): First signal data.
    w2 (torch.Tensor): Second signal data.
    *args (str): Variable length argument list containing:
                 args[0] (str): Label for the first signal.
                 args[1] (str): Label for the second signal.
                 args[2] (str, optional): Title for the plot (default: None).
    t_end (int): End time for the signals in seconds. Default value is t_end (assuming it's predefined).

    Returns:
    None
    """
    time = np.linspace(0, t_end, t_end * 500)
    
    plt.figure(figsize=(15, 3))
    plt.plot(time, w1[0,:,0], label=args[0])
    plt.plot(time, w2[0,:,0], label=args[1])
    plt.legend(fontsize=8, loc='upper right')
    plt.xlabel('Time (sec)', fontsize='8')
    if len(args) > 2:
        plt.title(args[2])
    plt.show()

def whole_plot(w1, w2, t_end=t_end):
    """
    Plot two signals w1 and w2 over an extended time period.

    Parameters:
    w1 (torch.Tensor): First signal data.
    w2 (torch.Tensor): Second signal data.
    t_end (int): End time for the signals in seconds. Default value is t_end (assuming it's predefined).

    Returns:
    None
    """
    time = np.linspace(0, 2 * t_end, 2 * t_end * 500)
    fig, ax = plt.subplots(2, 1, figsize=(10, 5))

    ax[0].plot(time, w2[0,:,0])
    ax[0].set_title('Reconstructed Signal')
    ax[0].set_xlabel('Time (sec)', fontsize='8')
    ax[0].set_ylim(0, 1)

    ax[1].plot(time, w1[0,:,0])
    ax[1].set_title('LFP Signal')
    ax[1].set_xlabel('Time (sec)', fontsize='8')
    ax[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.show()    
