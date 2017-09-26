import os
import cPickle
import numpy as np
import argparse
from state import *


def readUserData(filename):
    dataX = {};dataY = {}
    for line in open(filename):
        terms = line.strip().split('\t')
        #EDA, class, queryId
        if len(terms) == 3:
            if not dataX.has_key(terms[2]):
                dataX[terms[2]] = []
            dataX[terms[2]].append(terms[0])
            dataY[terms[2]] = int(float(terms[1]))

    return dataX,dataY


def printExample(X,Y):
    print 'X: ',X
    print 'Y: ',Y

def paddingFeature(X,maxLength):
    seqLen = min(len(X),maxLength)
    X = np.array(X)[:,None]
    features = np.zeros((maxLength,1),dtype=np.float)
    features[:seqLen,:] = X[:seqLen,:]
    return features

def create_dataset(state):
    filepath = state['raw_dataPath']
    savepath = state['dataSavePath']
    X = [];Y = []
    pathDir =  os.listdir(filepath)
    userCount = 0

    first = 1
    for allDir in pathDir:
        child = os.path.join('%s%s' % (filepath, allDir))
        if child.endswith('.txt'): 
            dataX,dataY = readUserData(child)
            for (key,item) in dataX.items():
                features = paddingFeature(dataX[key],state['max_length']) #to (maxLen,1)
                if first:
                    first = 0
                    printExample(features,dataY[key])
                    print 'Shape: ',features.shape
                X.append(features)
                Y.append(dataY[key])
            userCount += 1

    #sizeList = [len(x) for x in X]

    #avg = np.mean(sizeList)
    #mid = sizeList[len(sizeList)/2]
    #var_ = np.var(sizeList)
    #max_ = max(sizeList)

    
    #print 'max: ',max_
    #print 'avg: ',avg
    #print 'mid: ',mid
    #print 'var_: ',var_

    print 'User number: ',userCount
    cPickle.dump((X,Y),open(savepath,'w'))
    return X,Y

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prototype", type=str, help="Use the prototype", default='prototype_state')

    args = parser.parse_args()
    return args

def main(args):
    state = eval(args.prototype)()
    create_dataset(state)

if __name__ == '__main__':
    args = parse_args()
    main(args)

