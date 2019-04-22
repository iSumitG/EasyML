import tensorflow as tf

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

x = unpickle("../../data/cifar10/cifar-10-batches-py/data_batch_1")

#for t in x[b'data']:
    #print(t)

#print(len(x[b'data'][1]))
