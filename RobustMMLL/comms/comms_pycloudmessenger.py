# -*- coding: utf-8 -*-
'''
Comms interface to pycloudmessenger
@author:  Angel Navia Vázquez
'''
__author__ = "Angel Navia Vázquez, UC3M."

import time


def get_current_task_name(self):
    task_available = False
    while not task_available:
        try:
            with open('current_taskname.txt', 'r') as f:
                self.task_name = f.read()
            task_available = True
        except:
            print('No available task yet...')
            time.sleep(1)
            pass
    return self.task_name


class Comms_master:
    """
    """

    def __init__(self, commsffl):
        """
        """
        self.name = 'pycloudmessenger'
        self.commsffl = commsffl
        workers = self.commsffl.get_participants()
        self.workers_ids = list(workers.keys())

    def send(self, message, destiny):

        try:
            with self.commsffl:
                # self.send_to maps between worker_id and pseudo_id 
                self.commsffl.send(message, destiny)
        except Exception as err:
            print('\n')
            print('*' * 80)
            print('Pycloudmessenger ERROR at send: %s' % err)
            print('*' * 80)
            print('\n')
            raise

    def broadcast(self, message, receivers_list=None):
        # receivers_list are not used here, pycloudmessenger already knows all the recipients
        try:
            with self.commsffl:
                self.commsffl.send(message)
        except Exception as err:
            print('\n')
            print('*' * 80)
            print('Pycloudmessenger ERROR at broadcast: %s' % err)
            print('*' * 80)
            print('\n')
            raise

    def receive(self, timeout=1):
        try:
            with self.commsffl:
                packet = self.commsffl.receive(timeout)
            message = packet.content
            pseudo_id = packet.notification['participant']
            message.update({'sender': pseudo_id})
        except Exception as err:
            if 'pycloudmessenger.ffl.fflapi.TimedOutException' not in str(type(err)): # we skip the normal timeouts
                print('\n')
                print('*' * 80)
                print('Pycloudmessenger ERROR at receive: %s' % err)
                print('*' * 80)
                print('\n')
            else:
                message = None
            raise
        return message


    def receive_poms_123(self, timeout=10):
        with self.commsffl:
            packet = self.commsffl.receive(timeout)            
        return packet


class Comms_worker:
    """
    """

    def __init__(self, commsffl, worker_real_name='Anonymous'):
        """
        """
        self.id = worker_real_name  # unused by now...
        self.name = 'pycloudmessenger'
        self.commsffl = commsffl

    def send(self, message, address=None):
        try:
            # address is not used here, a worker can only send to the master        
            with self.commsffl:
                self.commsffl.send(message)
        except Exception as err:
            print('\n')
            print('*' * 80)
            print('Pycloudmessenger ERROR at send: %s' % err)
            print('*' * 80)
            print('\n')
            raise

    def receive(self, timeout=1):
        try:
            with self.commsffl:
                packet = self.commsffl.receive(timeout)
            message = packet.content
        except Exception as err:
            if 'pycloudmessenger.ffl.fflapi.TimedOutException' not in str(type(err)): # we skip the normal timeouts
                print('\n')
                print('*' * 80)
                print('Pycloudmessenger ERROR at receive: %s' % err)
                print('*' * 80)
                print('\n')
            else:
                message = None
            raise
        return message


    def receive_poms_123(self, timeout=10):
        with self.commsffl:
            packet = self.commsffl.receive(timeout)            
        return packet
