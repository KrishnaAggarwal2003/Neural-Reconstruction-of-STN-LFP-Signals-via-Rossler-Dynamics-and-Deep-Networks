import numpy as np
from functions import load_yaml
from rossler_network import RosslerSystem , hermitian_matrix, ross_preprocess
from lfp_process import lfp_data
from nn_model import feed_forward , lstm_nn
from train_model import training_model
import matplotlib.pyplot as plt

arguements = load_yaml('parameters.yaml')

patient_num = arguements['patient_num']
t_end = arguements['t_end']

# Model Parameters
lr = arguements['learning_rate']
hidden_size = arguements['hidden_size']
num_layers = arguements['num_layers']
output_size = arguements['output_size']
lstm_epoch = arguements['lstm_num_epochs']
forward_epoch = arguements['forward_num_epochs']

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


def run_code(model_name):
    print('**********Running Rossler Network****************')
    rossler_osc = RosslerSystem(N, complex_wts=hermitian_matrix(N), omega=omega, a=a,b=b,c=c,const=const,d=d,Iext=Iext, k=k, 
                                t_span=t_span, num_points=num_points)
    x,y,z = rossler_osc.solve()
    print('***********************Network running completed***************************')
    print('***********************Loading LFP and Rossler Data************************')
    train_lfp = lfp_data(mode='training', t_end=t_end, patient_num=patient_num)
    test_lfp = lfp_data(mode='testing', t_end=t_end, patient_num=patient_num)
    whole_lfp = lfp_data(mode='whole', t_end=t_end, patient_num=patient_num)

    train_data = ross_preprocess(x, mode='training', t_end=t_end)
    test_data =  ross_preprocess(x, mode='testing', t_end=t_end)
    whole_data = ross_preprocess(x, mode='whole', t_end=t_end)
      
    
    if model_name == 'Feed-forward':
        nn_model = feed_forward(input_size=N , output_size=output_size)
        epoch_num = forward_epoch

    elif model_name == 'LSTM model':
        nn_model = lstm_nn(input_size=N, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)
        epoch_num = lstm_epoch
    else:
        print("Model Name Error") 

    
    print(f'Name of the Model used is:- {model_name}\n')  
    
    print('***********************Initiating Model Training***************************')       
        
    training_model(nn_model, train_data, train_lfp, num_epochs=epoch_num, learning_rate=lr)
    
    print('***********************************Training Completed********************************')    

    
    keys = ['model', 'train rossler', 'test rossler', 'whole rossler', 'train lfp', 'test lfp', 'whole lfp']
    values = [nn_model , train_data , test_data , whole_data , train_lfp , test_lfp , whole_lfp]
    
    output = dict(zip(keys,values))  
    return output  


def signal_plot(w1, w2, *args, t_end=t_end):
    time = np.linspace(0, t_end, t_end*500)
    
    plt.figure(figsize = (15,3))
    plt.plot(time , w1.detach().numpy()[0,:,0], label = args[0])
    plt.plot(time , w2  .detach().numpy()[0,:,0], label = args[1])
    plt.legend(fontsize = 8, loc = 'upper right')
    plt.xlabel('Time (sec)', fontsize='8')
    if len(args) > 2:
        plt.title(args[2])
    plt.show()


def whole_plot(w1, w2, t_end=t_end):
    time = np.linspace(0 , 2*t_end , 2*t_end*500)
    fig,ax =plt.subplots(2,1, figsize=(10,5))

    ax[0].plot(time , w2.detach().numpy()[0,:,0])
    ax[0].set_title('Reconstructed Signal')
    ax[0].set_xlabel('Time (sec)', fontsize='8')
    ax[0].set_ylim(0,1)

    ax[1].plot(time , w1.detach().numpy()[0,:,0])
    ax[1].set_title('LFP Signal')
    ax[1].set_xlabel('Time (sec)', fontsize='8') 
    ax[1].set_ylim(0,1)

    plt.tight_layout()
    plt.show()