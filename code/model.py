from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Masking, Embedding,LSTM
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import train_test_split

import numpy,os
import argparse
import cPickle
from state import *
import numpy as np

def create_rnn_model(state,optimizer='rmsprop'):
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(state['max_length'],1)))
    model.add(LSTM(units=state['rnn_hidden_units'], input_shape=(state['max_length'],1)))
    model.add(Dense(state['fc_hidden_units'], activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='rmsprop',metrics=['mse'])
    return model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prototype", type=str, help="Use the prototype", default='prototype_state')
    args = parser.parse_args()
    return args

def main(args):
    state = eval(args.prototype)()
    #X,Y are list
    (X,Y) = cPickle.load(open(state['dataSavePath']))

    seed = state['seed']
    np.random.seed(seed)

    assert len(X) = len(Y)
    
    X = np.array(X)
    Y = np.array(Y)


    print 'X shape: ',type(X),X.shape
    print 'Y shape: ',type(Y),Y.shape

    model = create_rnn_model(state)

    #split data
    train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size=0.2)
    
    #training
    model.fit(train_x, train_y, nb_epoch=state['epoch'], batch_size=state['bs'])

    #testing
    scores = model.evaluate(test_x, test_y, batch_size=state['bs'])

    #MSE
    print("%s: %.4f" % (model.metrics_names[1], scores[1]))

    #estimator = KerasRegressor(build_fn=model, nb_epoch=state['epoch'], batch_size=state['bs'], verbose=0)

    #kfold = KFold(n_splits=10, random_state=seed)
    #results = cross_val_score(estimator, X, Y, cv=kfold)

    #print("Results: %.3f (%.3f) MSE" % (results.mean(), results.std()))


if __name__ == '__main__':
    args = parse_args()
    main(args)