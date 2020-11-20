# -*- coding: utf-8 -*-
'''
Common ML operations to be used by all algorithms in POM1

'''

__author__ = "Marcos Fernandez Diaz"
__date__ = "May 2020"

import sys
import numpy as np

from RobustMMLL.models.POM1.CommonML.POM1_ML import POM1ML



class POM1_CommonML_Master(POM1ML):
    """
    This class implements the Common ML operations, run at Master node. It inherits from POM1ML.
    """

    def __init__(self, workers_addresses, comms, logger, verbose=False):
        """
        Create a :class:`POM1_CommonML_Master` instance.

        Parameters
        ----------
        workers_addresses: list of strings
            list of the addresses of the workers

        comms: comms object instance
            object providing communications

        logger: class:`logging.Logger`
            logging object instance

        verbose: boolean
            indicates if messages are print or not on screen

        """
        self.workers_addresses = workers_addresses
        self.comms = comms
        self.logger = logger
        self.verbose = verbose

        self.name = 'POM1_CommonML_Master'          # Name
        self.platform = comms.name                  # Type of comms to use (either 'pycloudmessenger' or 'local_flask')



    def terminate_Workers(self, workers_addresses_terminate=None):
        """
        Send order to terminate Workers

        Parameters
        ----------
        users_addresses_terminate: List of strings
            Addresses of the workers to be terminated

        """
        packet = {'action': 'STOP', 'to': 'MLmodel'}
        # Broadcast packet to all workers
        self.comms.broadcast(packet, self.workers_addresses)
        self.display(self.name + ' sent STOP to all Workers')



    def reset(self):
        """
        Create/reset some empty variables needed by the Master Node
        """
        self.display(self.name + ': Resetting local data')
        self.list_centroids = []
        self.list_counts = []
        self.list_dists = []
        self.list_public_keys = []
        self.list_gradients = []
        self.list_weights = []
        self.list_costs = []
    
    

    def checkAllStates(self, condition, state_dict):
        """
        Checks if all worker states satisfy a given condition

        Parameters
        ----------
        condition: String
            Condition to check
        state_dict: Dictionary
            Dictionary whose values need to be compared against condition

        Returns
        ----------
        all_active: Boolean
            Flag indicating if all values inside dictionary are equal to condition
        """
        all_active = True
        for worker in self.workers_addresses:
            if state_dict[worker] != condition:
                all_active = False
                break
        return all_active



    def train_Master(self):
        """
        This is the main training loop, it runs the following actions until the stop condition is met:
            - Update the execution state
            - Perform actions according to the state
            - Process the received packets

        Parameters
        ----------
        None
        """        
        self.state_dict.update({'CN': 'START_TRAIN'})
        self.display(self.name + ': Starting training')

        while self.state_dict['CN'] != 'END':
            self.Update_State_Master()
            self.TakeAction_Master()
            self.CheckNewPacket_Master()
            
        self.display(self.name + ': Training is done')



    def CheckNewPacket_Master(self):
        """
        Checks if there is a new message in the Master queue

        Parameters
        ----------
            None
        """
        if self.platform == 'pycloudmessenger':
            packet = None
            sender = None
            try:
                packet = self.comms.receive_poms_123(timeout=10) # We only receive a dictionary at a time even if there are more than 1 workers
                try:  # For the pycloudmessenger cloud
                    sender = packet.notification['participant']
                except Exception: # For the pycloudmessenger local
                    self.counter = (self.counter + 1) % self.Nworkers
                    sender = self.workers_addresses[self.counter]
                    
                packet = packet.content
                self.display(self.name + ': Received %s from worker %s' %(packet['action'], sender))
                self.ProcessReceivedPacket_Master(packet, sender)
            except KeyboardInterrupt:
                self.display(self.name + ': Shutdown requested by Keyboard...exiting')
                sys.exit()
            except Exception as err:
                if 'pycloudmessenger.ffl.fflapi.TimedOutException' in str(type(err)):
                    pass
                else:
                    self.display(self.name + ': Error %s' %err)
                    raise
        else: # Local flask
            packet = None
            sender = None
            for sender in self.workers_addresses:
                try:
                    packet = self.comms.receive(sender, timeout=0.1)
                    self.display(self.name + ': Received %s from worker %s' %(packet['action'], sender))
                    self.ProcessReceivedPacket_Master(packet, sender)
                except KeyboardInterrupt:
                    self.display(self.name + ': Shutdown requested by Keyboard...exiting')
                    sys.exit()
                except Exception as err:
                    if str(err).startswith('Timeout when receiving data'): # TimedOutException
                        pass
                    else:
                        self.display(self.name + ': Error %s' %err)
                        raise

       
        

#===============================================================
#                 Worker   
#===============================================================

class POM1_CommonML_Worker(POM1ML):
    '''
    Class implementing the POM1 Common operations, run at Worker

    '''

    def __init__(self, logger, verbose=False):
        """
        Create a :class:`POM1_CommonML_Worker` instance.

        Parameters
        ----------
        logger: class:`logging.Logger`
            logging object instance

        verbose: boolean
            Indicates if messages are print or not on screen
        """
        self.logger = logger
        self.verbose = verbose

        self.name = 'POM1_CommonML_Worker'      # Name



    def run_worker(self):
        """
        This is the training executed at every Worker

        Parameters
        ----------
        None
        """
        self.display(self.name + ' %s: READY and waiting instructions' %(self.worker_address))
        self.terminate = False

        while not self.terminate:
            self.CheckNewPacket_worker()



    def CheckNewPacket_worker(self):
        """
        Checks if there is a new message in the Worker queue

        Parameters
        ----------
        None
        """
        if self.platform == 'pycloudmessenger':
            packet = None
            sender = None
            try:
                packet = self.comms.receive_poms_123(timeout=10)
                packet = packet.content
                sender = 'Master'
                self.display(self.name + ' %s: Received %s from %s' % (self.worker_address, packet['action'], sender))

                if packet['to'] == 'Preprocessing':
                    self.ProcessPreprocessingPacket(packet)
                else: # Message for training the ML model
                    self.ProcessReceivedPacket_Worker(packet)
            except KeyboardInterrupt:
                self.display(self.name + '%s: Shutdown requested by Keyboard...exiting' %self.worker_address)
                sys.exit()
            except Exception as err:
                if 'pycloudmessenger.ffl.fflapi.TimedOutException' in str(type(err)):
                    pass
                else:
                    self.display(self.name + ': Error %s' %err)
                    raise
        else: # Local flask
            packet = None
            sender = None
            try:
                packet = self.comms.receive(self.master_address, timeout=0.1)
                sender = 'Master'
                self.display(self.name + ' %s: Received %s from %s' % (self.worker_address, packet['action'], sender))

                if packet['to'] == 'Preprocessing':
                    self.ProcessPreprocessingPacket(packet)
                else: # Message for training the ML model
                    self.ProcessReceivedPacket_Worker(packet)
            except KeyboardInterrupt:
                self.display(self.name + '%s: Shutdown requested by Keyboard...exiting' %self.worker_address)
                sys.exit()
            except Exception as err:
                if str(err).startswith('Timeout when receiving data'): # TimedOutException
                    pass
                else:
                    self.display(self.name + ': Error %s' %err)
                    raise



    def ProcessPreprocessingPacket(self, packet):
        """
        Take an action after receiving a packet for the preprocessing

        Parameters
        ----------
        packet: Dictionary
            Packet received
        """
        self.terminate = False

        # Exit the process
        if packet['action'] == 'STOP':
            self.display(self.name + ' %s: terminated by Master' %self.worker_address)
            self.terminate = True            
        
        if packet['action'] == 'SEND_MEANS':
            self.display(self.name + ' %s: Obtaining means' %self.worker_address)
            self.data_description = np.array(packet['data']['data_description'])
            means = np.mean(self.Xtr_b, axis=0)
            counts = self.Xtr_b.shape[0]

            action = 'COMPUTE_MEANS'
            data = {'means': means, 'counts':counts}
            packet = {'action': action, 'data': data}            
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))            

        if packet['action'] == 'SEND_STDS':
            self.display(self.name + ' %s: Obtaining stds' %self.worker_address)
            self.global_means = np.array(packet['data']['global_means'])
            X_without_mean = self.Xtr_b-self.global_means                              
            var = np.mean(X_without_mean*X_without_mean, axis=0)
            counts = self.Xtr_b.shape[0]

            action = 'COMPUTE_STDS'
            data = {'var': var, 'counts':counts}
            packet = {'action': action, 'data': data}            
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))
    
        if packet['action'] == 'SEND_MIN_MAX':
            self.display(self.name + ' %s: Obtaining means' %self.worker_address)
            self.data_description = np.array(packet['data']['data_description'])
            mins = np.min(self.Xtr_b, axis=0)
            maxs = np.max(self.Xtr_b, axis=0)

            action = 'COMPUTE_MIN_MAX'
            data = {'mins': mins, 'maxs':maxs}
            packet = {'action': action, 'data': data}            
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))
            
        if packet['action'] == 'SEND_PREPROCESSOR':
            self.display(self.name + ' %s: Receiving preprocessor' %self.worker_address)

            # Store the preprocessing object
            self.prep_model = packet['data']['prep_model']
            self.display(self.name + ' %s: Final preprocessor stored' %self.worker_address)

            # Apply the received object to Xtr_b and store back the result
            Xtr = np.copy(self.Xtr_b)
            X_prep = self.prep_model.transform(Xtr)
            self.Xtr_b = np.copy(X_prep)
            self.display(self.name + ' %s: Training set transformed using preprocessor' %self.worker_address)

            action = 'ACK_SEND_PREPROCESSOR'
            packet = {'action': action}            
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))
            self.preprocessor_ready = True
