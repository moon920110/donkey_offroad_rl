import torch as tr
import torch.optim as optim

class BaseAgent(object):
    '''
    Base class for Keras models that will provide steering and throttle to guide a car.
    '''

    def __init__(self):
        self.lr = 0.002
        self.decay = None
    def save(self):
        pass
    def load(self, model_paths):
        pass

    def load_weights(self, model_path, by_name=True):
        pass

    def shutdown(self):
        pass

    def log_dir(self):
        '''about Tensorboard & Visualize'''
        pass

    def train(self):
        pass
    def run(self,img,speed,meter,train_state):
       pass