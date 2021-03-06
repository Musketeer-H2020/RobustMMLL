# -*- coding: utf-8 -*-
'''
Neural Network model
'''

__author__ = "Marcos Fernández Díaz"
__date__ = "May 2019"


# Code to ensure reproducibility in the results
from numpy.random import seed
seed(1)
from tensorflow.compat.v1 import set_random_seed
set_random_seed(2)

import numpy as np
from keras import backend as K
from keras import losses
from keras.models import model_from_json, Sequential
from keras.layers import Dense, Activation
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os
# Disables the warning "Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA", doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from RobustMMLL.models.POM1.CommonML.POM1_CommonML import POM1_CommonML_Master, POM1_CommonML_Worker



class model():

    def __init__(self, model_architecture, optimizer='Adam', loss='categorical_crossentropy', metric='accuracy'):
        """
        Initializes keras model

        Parameters
        ----------
        model_architecture: JSON
            JSON containing the neural network architecture as defined by Keras (in model.to_json())
        optimizer: String
            Type of optimizer to use (must be one from https://keras.io/api/optimizers/)
        loss: String
            Type of loss to use (must be one from https://keras.io/api/losses/)
        metric: String
            Type of metric to use (must be one from https://keras.io/api/metrics/)
        """
        self.keras_model = model_from_json(model_architecture)                        # Store the model architecture
        self.keras_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])    # Compile the model



    def predict(self, X_b):
        """
        Predicts outputs given the model and inputs

        Parameters
        ----------
        X_b: ndarray
            2-D numpy array containing the input patterns

        Returns
        -------
        preds: ndarray
            1-D array containing the predictions
        """
        preds = self.keras_model.predict(X_b) # One-hot encoding
        preds = np.argmax(preds, axis=-1)     # Labels

        return preds




class NN_Master(POM1_CommonML_Master):
    """
    This class implements Neural nets, run at Master node. It inherits from POM1_CommonML_Master.
    """
    def __init__(self, comms, logger, verbose=False, robust=None, model_architecture=None, Nmaxiter=10, learning_rate=0.0001, model_averaging='True', optimizer='adam', loss='categorical_crossentropy', metric='accuracy', batch_size=32, num_epochs=1):
        """
        Create a :class:`NN_Master` instance.

        Parameters
        ----------
        comms: comms object instance
            Object providing communication functionalities

        logger: class:`mylogging.Logger`
            Logging object instance

        verbose: boolean
            Indicates if messages are print or not on screen

        robust: robust_node object instance
            Object providing robust functionalities

        model_architecture: JSON
            JSON containing the neural network architecture as defined by Keras (in model.to_json())

        Nmaxiter: int
            Maximum number of iterations

        learning_rate: float
            Learning rate for training

        model_averaging: Boolean
            Wether to use model averaging (True) or gradient averaging (False)

        optimizer: String
            Type of optimizer to use (should be one from https://keras.io/api/optimizers/)

        loss: String
            Type of loss to use (should be one from https://keras.io/api/losses/)

        metric: String
            Type of metric to use (should be one from https://keras.io/api/metrics/)

        batch_size: Int
            Size of the batch to use for training in each worker locally

        num_epochs: Int
            Number of epochs to train in each worker locally before sending the result to the master
        """
        self.comms = comms    
        self.robust = robust
        self.logger = logger
        self.verbose = verbose
        self.model_architecture = model_architecture
        self.Nmaxiter = Nmaxiter
        self.learning_rate = learning_rate
        self.model_averaging = model_averaging.lower() # Convert to lowercase
        self.optimizer = optimizer
        self.loss = loss
        self.metric = metric
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.name = 'POM1_NN_Master'                # Name
        self.platform = comms.name                  # Type of comms to use: either 'pycloudmessenger' or 'local_flask'
        self.workers_addresses = comms.workers_ids  # Addresses of the workers
        self.Nworkers = len(self.workers_addresses) # Number of workers
        self.reset()                                # Reset local data
        self.model = model(model_architecture, self.optimizer, self.loss, self.metric)      # Keras model initialization
        self.display(self.name + ': Model architecture:')
        self.model.keras_model.summary(print_fn=self.display)
        self.iter = 0                               # Number of iterations
        self.is_trained = False                     # Flag to know if the model has been trained
        self.state_dict = {}                        # Dictionary storing the execution state
        for worker in self.workers_addresses:
            self.state_dict.update({worker: ''})



    def Update_State_Master(self):
        '''
        Function to control the state of the execution

        Parameters
        ----------
        None
        '''
        if self.state_dict['CN'] == 'START_TRAIN':
            self.state_dict['CN'] = 'INIT_MODEL'

        if self.model_averaging == 'true':
            if self.checkAllStates('ACK_INIT_MODEL', self.state_dict):
                for worker in self.workers_addresses:
                    self.state_dict[worker] = ''
                self.state_dict['CN'] = 'COMPILE_INIT'
            if self.checkAllStates('ACK_COMPILE_INIT', self.state_dict):
                for worker in self.workers_addresses:
                    self.state_dict[worker] = ''
                self.state_dict['CN'] = 'FIT_INIT'
            if self.checkAllStates('ACK_FIT_INIT', self.state_dict):
                for worker in self.workers_addresses:
                    self.state_dict[worker] = ''
                self.state_dict['CN'] = 'LOCAL_TRAIN'
            if self.checkAllStates('LOCAL_UPDATE', self.state_dict):
                for worker in self.workers_addresses:
                    self.state_dict[worker] = ''
                self.state_dict['CN'] = 'MODEL_AVERAGING'

        else:            
            if self.checkAllStates('ACK_INIT_MODEL', self.state_dict):
                for worker in self.workers_addresses:
                    self.state_dict[worker] = ''
                self.state_dict['CN'] = 'COMPUTE_GRADIENTS'

            if self.checkAllStates('UPDATE_GRADIENTS', self.state_dict):
                for worker in self.workers_addresses:
                    self.state_dict[worker] = ''
                self.state_dict['CN'] = 'UPDATE_MODEL'
    
    
    
    def TakeAction_Master(self):
        """
        Takes actions according to the state

        Parameters
        ----------
        None
        """
        # Send model to all workers
        if self.state_dict['CN'] == 'INIT_MODEL':
            action = 'INIT_MODEL'
            to = 'MLmodel'
            data = {'model_json': self.model_architecture}
            packet = {'to': to, 'action': action, 'data': data}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent ' + action + ' to all workers')
            self.state_dict['CN'] = 'wait'

        # Compute average of gradients and update model
        if self.state_dict['CN'] == 'UPDATE_MODEL':
            for index_layer in range(len(self.model.keras_model.get_weights())):
                layer_gradients = []
                for worker in range(self.Nworkers):
                    layer_gradients.append(self.list_gradients[worker][index_layer])                 
                mean_weights = np.mean(layer_gradients, axis=0) # Average layer gradients for all workers
                K.set_value(self.model.keras_model.trainable_weights[index_layer], K.get_value(self.model.keras_model.trainable_weights[index_layer]) - self.learning_rate*mean_weights) # Update model weights
                
            self.reset()
            self.state_dict['CN'] = 'CHECK_TERMINATION'
            self.iter += 1

        # Compute model averaging
        if self.state_dict['CN'] == 'MODEL_AVERAGING':
            if self.robust is not None:
                new_weights = self.robust.aggregate(self.list_weights) # Update model weights
            else:
                new_weights = []
                for index_layer in range(len(self.list_weights[0])):
                    layer_weights = []
                    for worker in range(len(self.list_weights)):
                        layer_weights.append(self.list_weights[worker][index_layer])                 
                    mean_weights = np.mean(layer_weights, axis=0) # Average layer weights for all workers
                    new_weights.append(mean_weights)

            self.model.keras_model.set_weights(new_weights)        
            self.reset()
            self.state_dict['CN'] = 'CHECK_TERMINATION'
            self.iter += 1

        # Check for termination of the training
        if self.state_dict['CN'] == 'CHECK_TERMINATION':
            if self.Xval is not None and self.yval is not None:
                [loss, accuracy] = self.model.keras_model.evaluate(self.Xval, self.yval, verbose=self.verbose)
                self.display(self.name + ': Iteration %d, loss: %0.4f val accuracy: %0.4f' %(self.iter, loss, accuracy))
            if self.iter == self.Nmaxiter:
                self.state_dict['CN'] = 'SEND_FINAL_MODEL'
                self.display(self.name + ': Stopping training, maximum number of iterations reached!')
            else:
                if self.model_averaging == 'true':
                    self.state_dict['CN'] = 'LOCAL_TRAIN' 
                else:
                    self.state_dict['CN'] = 'COMPUTE_GRADIENTS'           
                if self.Xval is None or self.yval is None:
                    self.display(self.name + ': Iteration %d' %self.iter)

        # Asking the workers to compute local gradients
        if self.state_dict['CN'] == 'COMPUTE_GRADIENTS':
            action = 'COMPUTE_LOCAL_GRADIENTS'
            to = 'MLmodel'
            data = {'model_weights': self.model.keras_model.get_weights()}
            packet = {'to': to, 'action': action, 'data': data}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent ' + action + ' to all workers')
            self.state_dict['CN'] = 'wait_gradients'

        # Asking the workers to compile the model
        if self.state_dict['CN'] == 'COMPILE_INIT':
            action = 'COMPILE_INIT'
            to = 'MLmodel'
            data = {'optimizer': self.optimizer, 'loss': self.loss, 'metric': self.metric}
            packet = {'to': to, 'action': action, 'data': data}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent ' + action + ' to all workers')
            self.state_dict['CN'] = 'wait'

        # Asking the workers to initialize fit parameters
        if self.state_dict['CN'] == 'FIT_INIT':
            action = 'FIT_INIT'
            to = 'MLmodel'
            data = {'batch_size': self.batch_size, 'num_epochs': self.num_epochs}
            packet = {'to': to, 'action': action, 'data': data}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent ' + action + ' to all workers')
            self.state_dict['CN'] = 'wait'

        # Asking the workers to update model with local data
        if self.state_dict['CN'] == 'LOCAL_TRAIN':
            action = 'LOCAL_TRAIN'
            to = 'MLmodel'
            data = {'model_weights': self.model.keras_model.get_weights()}
            packet = {'to': to, 'action': action, 'data': data}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent ' + action + ' to all workers')
            self.state_dict['CN'] = 'wait_weights'
        
        # Send final model to all workers
        if self.state_dict['CN'] == 'SEND_FINAL_MODEL':
            action = 'SEND_FINAL_MODEL'
            to = 'MLmodel'
            data = {'model_weights': self.model.keras_model.get_weights()}
            packet = {'to': to, 'action': action, 'data': data}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent %s to all workers' %action)
            self.is_trained = True
            self.state_dict['CN'] = 'wait'
            


    def ProcessReceivedPacket_Master(self, packet, sender):
        """
        Process the received packet at Master and take some actions, possibly changing the state

        Parameters
        ----------
        packet: Dictionary
            Packet received

        sender: Strings
            Id of the sender
        """
        if packet['action'][0:3] == 'ACK':
            self.state_dict[sender] = packet['action']
            if self.checkAllStates('ACK_FINAL_MODEL', self.state_dict): # Included here to avoid calling CheckNewPacket_Master after sending the final model (this call could imply significant delay if timeout is set to a high value)
                self.state_dict['CN'] = 'END'

        if self.state_dict['CN'] == 'wait_gradients':
            if packet['action'] == 'UPDATE_GRADIENTS':
                self.list_gradients.append(packet['data']['gradients'])
                self.state_dict[sender] = packet['action']

        if self.state_dict['CN'] == 'wait_weights':
            if packet['action'] == 'LOCAL_UPDATE':
                self.list_weights.append(packet['data']['weights'])
                self.state_dict[sender] = packet['action']
  
    
    

#===============================================================
#                 Worker   
#===============================================================

class NN_Worker(POM1_CommonML_Worker):
    '''
    Class implementing Neural nets, run at Worker

    '''

    def __init__(self, master_address, comms, logger, verbose=False, Xtr_b=None, ytr=None):
        """
        Create a :class:`NN_Worker` instance.

        Parameters
        ----------
        master_address: string
            Identifier of the master instance

        comms: comms object instance
            Object providing communication functionalities

        logger: class:`mylogging.Logger`
            Logging object instance

        verbose: boolean
            Indicates if messages are print or not on screen

        Xtr_b: np.ndarray
            2-D numpy array containing the inputs for training

        ytr: np.ndarray
            2-D numpy array containing the labels for training
        """
        self.master_address = master_address
        self.comms = comms
        self.logger = logger
        self.verbose = verbose
        self.Xtr_b = Xtr_b
        self.ytr = ytr

        self.name = 'POM1_NN_Worker'                           # Name
        self.worker_address = comms.id
        self.platform = comms.name
        self.num_classes = ytr.shape[1]
        self.sess = tf.compat.v1.InteractiveSession()          # Create TF session
        init = tf.compat.v1.global_variables_initializer()     # Initialize variables
        self.sess.run(init)                                    # Start TF session
        self.is_trained = False                                # Flag to know if the model has been trained
        
        

    def ProcessReceivedPacket_Worker(self, packet):
        """
        Take an action after receiving a packet

        Parameters
        ----------
            packet: packet object 
                packet received (usually a dict with various content)

        """
        self.terminate = False

        # Exit the process
        if packet['action'] == 'STOP':
            self.display(self.name + ' %s: terminated by Master' %self.worker_address)
            self.terminate = True
        
        if packet['action'] == 'INIT_MODEL':
            self.display(self.name + ' %s: Initializing local model' %self.worker_address)
            model_json = packet['data']['model_json']
            # Initialize local model
            self.current_index = 0
            self.model = model(model_json)
            self.display(self.name + ': Model architecture:')
            self.model.keras_model.summary(print_fn=self.display)
            self.label_placeholder = tf.compat.v1.placeholder(tf.float32, shape=[None, self.num_classes])
            self.loss = losses.categorical_crossentropy(self.label_placeholder, self.model.keras_model.output)
            self.gradients = K.gradients(self.loss, self.model.keras_model.trainable_weights)
            action = 'ACK_INIT_MODEL'
            packet = {'action': action}
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))

        if packet['action'] == 'COMPILE_INIT':
            self.display(self.name + ' %s: Initializing compiler' %self.worker_address)
            optimizer = packet['data']['optimizer']
            loss = packet['data']['loss']
            metric = packet['data']['metric']
            # Compile the model
            self.model.keras_model.compile(loss=loss, optimizer=optimizer, metrics=[metric])
            action = 'ACK_COMPILE_INIT'
            packet = {'action': action}
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))

        if packet['action'] == 'FIT_INIT':
            self.display(self.name + ' %s: Storing batch size and number of epochs' %self.worker_address)
            self.batch_size = packet['data']['batch_size']
            self.num_epochs =  packet['data']['num_epochs']
            action = 'ACK_FIT_INIT'
            packet = {'action': action}
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))

        if packet['action'] == 'LOCAL_TRAIN':
            self.display(self.name + ' %s: Updating model locally' %self.worker_address)
            weights = packet['data']['model_weights']
            self.model.keras_model.set_weights(weights)
            self.model.keras_model.fit(self.Xtr_b, self.ytr, epochs=self.num_epochs, batch_size=self.batch_size, verbose=1)
            action = 'LOCAL_UPDATE'
            data = {'weights': self.model.keras_model.get_weights()}
            packet = {'action': action, 'data': data}            
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))
            
        if packet['action'] == 'COMPUTE_LOCAL_GRADIENTS':
            self.display(self.name + ' %s: Computing local gradients' %self.worker_address)
            gradients = self.get_weight_grad(packet['data']['model_weights'], num_data=500)
            action = 'UPDATE_GRADIENTS'
            data = {'gradients': gradients}
            packet = {'action': action, 'data': data}            
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))
            
        if packet['action'] == 'SEND_FINAL_MODEL':            
            model_weights = packet['data']['model_weights']
            self.model.keras_model.set_weights(model_weights)
            self.display(self.name + ' %s: Final model stored' %self.worker_address)
            [_, accuracy] = self.model.keras_model.evaluate(self.Xtr_b, self.ytr, verbose=self.verbose)
            self.display(self.name + ' %s: Accuracy in training set: %0.4f' %(self.worker_address, accuracy))
            self.is_trained = True

            action = 'ACK_FINAL_MODEL'
            packet = {'action': action}            
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))



    def get_weight_grad(self, model_weights, num_data=None):
        """ Gets gradient of model for given inputs and outputs for all weights"""
        if num_data==None or num_data>self.Xtr_b.shape[0]:
            x_batch = self.Xtr_b
            y_batch = self.ytr
        else:
            data_indexes = (np.array(range(num_data)) + self.current_index) %self.Xtr_b.shape[0]
            self.current_index = (self.current_index + num_data) %self.Xtr_b.shape[0]
            x_batch = np.take(self.Xtr_b, data_indexes, axis=0) # Slice along first axis
            y_batch = np.take(self.ytr, data_indexes, axis=0)

        self.model.keras_model.set_weights(model_weights)        
        output_grad = self.sess.run(self.gradients, feed_dict={self.label_placeholder: y_batch, self.model.keras_model.input: x_batch})

        return output_grad
