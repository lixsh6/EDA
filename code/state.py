from collections import OrderedDict

def prototype_state():
    state = {}
    state['raw_dataPath'] = '../data/'
    state['dataSavePath'] = '../idata/data.pkl'
    state['max_length'] = 1000
    state['rnn_hidden_units'] = 512
    state['fc_hidden_units'] = 24       #full-connection layer
    state['bs'] = 64                    #batch size

    state['seed'] = 7
    state['epoch'] = 100
    return state


    
