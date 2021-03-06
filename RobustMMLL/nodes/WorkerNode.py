# -*- coding: utf-8 -*-
'''
Worker node object
'''

__author__ = "Angel Navia-Vázquez, Marcos Fernández"
__date__ = "May 2020"

import numpy as np
import pickle
from RobustMMLL.Common_to_all_objects import Common_to_all_objects
import time
import json

class WorkerNode(Common_to_all_objects):
    """
    This class represents the main process associated to every Worker Node, and it responds to the commands sent by the master to carry out the training procedure under POMs 4, 5 and 6
    """

    def __init__(self, pom, comms, logger, verbose=False, **kwargs):

        """
        Create a :class:`WorkerNode` instance.

        Parameters
        ----------
        pom: integer
            the selected POM

        comms: comms object instance
            object providing communications

        logger: class:`logging.Logger`
            logging object instance

        verbose: boolean
            indicates if messages are print or not on screen

        **kwargs: Arbitrary keyword arguments.

       """

        self.pom = pom                  # Selected POM
        self.comms = comms              # The comms library for this worker
        self.logger = logger            # logger
        self.verbose = verbose          # print on screen when true
        self.master_address = 'ma'
        
        self.process_kwargs(kwargs)

        # The id of this worker (string), received through the comms library...
        # self.worker_address = worker_address          
        self.worker_address = self.comms.id
        self.cryptonode_address = 'ca'          # The id of the cryptonode (string), default value

        self.terminate = False          # used to terminate the process
        self.model_type = None          # ML model to be used, to be defined later
        self.display('WorkerNode %s: Loading comms' % str(self.worker_address))
        self.display('WorkerNode %s: Loading Data Connector' % str(self.worker_address))
        self.display('WorkerNode %s: Initiated' % str(self.worker_address))
        self.data_is_ready = False

    def set_training_data_OLD(self, dataset_name, Xtr=None, ytr=None):
        """
        Set data to be used for training.

        Parameters
        ----------
        dataset_name: (string): dataset name
        Xval: Input data matrix: row = No. patterns , col = No. features
        yval: target vector: row = No. patterns
        add_bias: boolean. If true, it adds a column of ones to the input data matrix
        """
        self.dataset_name = dataset_name
        try:
            self.NPtr = Xtr.shape[0]
            self.NI = Xtr.shape[1]                # Number of inputs
            self.Xtr_b = Xtr
            self.ytr = ytr

            if self.Xtr_b.shape[0] != self.ytr.shape[0] and ytr is not None:
                self.display('ERROR: different number of patterns in Xtr and ytr (%s vs %s)' % (str(self.Xval_b.shape[0]), str(self.yval.shape[0])))
                self.Xtr_b = None
                self.ytr = None
                self.NPtr = 0
                self.display('WorkerNode: ***** Train data NOT VALID. *****')
                return
            else:
                self.display('WorkerNode got train data: %d patterns, %d features' % (self.NPtr, self.NI))
                self.data_is_ready = True
        except:
            self.display('WorkerNode: ***** Training data NOT available. *****')
            pass

    def set_training_data(self, dataset_name, Xtr=None, ytr=None):
        """
        Set data to be used for training

        *****  List of lists... ****

        Parameters
        ----------
        dataset_name: (string): dataset name
        Xtr: Input data: list of lists
        ytr: target vector: list of lists 
        """
        self.dataset_name = dataset_name
        try:
            self.Xtr_b = np.array(Xtr)            
            self.ytr = np.array(ytr)
            self.NPtr, self.NI = self.Xtr_b.shape
            
            if self.Xtr_b.shape[0] != self.ytr.shape[0] and ytr is not None:
                self.display('ERROR: different number of patterns in Xtr and ytr (%s vs %s)' % (str(self.Xval_b.shape[0]), str(self.yval.shape[0])))
                self.Xtr_b = None
                self.ytr = None
                self.NPtr = 0
                self.display('WorkerNode: ***** Train data NOT VALID. *****')
                return
            else:
                self.display('WorkerNode got train data: %d patterns, %d features' % (self.NPtr, self.NI))

        except:
            self.display('WorkerNode: ***** Training data NOT available. *****')
            pass



    def set_validation_data(self, dataset_name, Xval=None, yval=None):
        """
        Set data to be used for validation.

        Parameters
        ----------
        dataset_name: (string): dataset name
        Xval: Input data matrix: row = No. patterns , col = No. features
        yval: target vector: row = No. patterns
        add_bias: boolean. If true, it adds a column of ones to the input data matrix
        """
        self.dataset_name = dataset_name
        try:
            self.NPval = Xval.shape[0]
            self.NI = Xval.shape[1]                # Number of inputs
            self.yval = yval
            self.Xval_b = Xval

            if self.Xval_b.shape[0] != self.yval.shape[0] and yval is not None:
                self.display('ERROR: different number of patterns in Xval and yval (%s vs %s)' % (str(self.Xval_b.shape[0]), str(self.yval.shape[0])))
                self.Xval_b = None
                self.yval = None
                self.NPval = 0
                self.display('WorkerNode: ***** Validation data NOT VALID. *****')
                return
            else:
                self.display('WorkerNode got validation data: %d patterns, %d features' % (self.NPval, self.NI))
        except:
            self.display('WorkerNode: ***** Validation data NOT available. *****')
            pass


    def set_test_data(self, dataset_name, Xtst=None, ytst=None):
        """
        Set data to be used for testing.

        Parameters
        ----------
        dataset_name: (string): dataset name
        Xtst: Input data matrix: row = No. patterns , col = No. features
        ytst: target vector: row = No. patterns
        add_bias: boolean. If true, it adds a column of ones to the input data matrix
        """
        self.dataset_name = dataset_name
        try:
            self.NPtst = Xtst.shape[0]
            self.NI = Xtst.shape[1]                # Number of inputs
            self.ytst = ytst
            self.Xtst_b = Xtst
            if self.Xtst_b.shape[0] != self.ytst.shape[0]  and ytst is not None:
                self.display('ERROR: different number of patterns in Xtst and ytst (%s vs %s)' % (str(self.Xval_b.shape[0]), str(self.yval.shape[0])))
                self.Xtst_b = None
                self.ytst = None
                self.NPtst = 0
                self.display('WorkerNode: ***** Test data NOT VALID. *****')
                return
            else:
                self.display('WorkerNode got test data: %d patterns, %d features' % (self.NPtst, self.NI))
        except:
            self.display('WorkerNode: ***** Test data NOT available. *****')
            pass



    def create_model_worker(self, model_type):
        """
        Create the model object to be used for training at the Worker side.

        Parameters
        ----------
        model_type: str
            Type of model to be used

        """
        self.model_type = model_type

        if self.pom == 1:
            if model_type == 'Kmeans':
                from RobustMMLL.models.POM1.Kmeans.Kmeans import Kmeans_Worker
                self.workerMLmodel = Kmeans_Worker(self.master_address, self.comms, self.logger,  self.verbose, self.Xtr_b)

            elif model_type == 'NN':
                from RobustMMLL.models.POM1.NeuralNetworks.neural_network import NN_Worker
                self.workerMLmodel = NN_Worker(self.master_address, self.comms, self.logger,  self.verbose, self.Xtr_b, self.ytr)

            elif model_type == 'SVM':
                from RobustMMLL.models.POM1.SVM.SVM import SVM_Worker
                self.workerMLmodel = SVM_Worker(self.master_address, self.comms, self.logger,  self.verbose, self.Xtr_b, self.ytr)

            self.display('WorkerNode %s: Created %s model under POM %d' % (str(self.worker_address), model_type, self.pom))

        if self.pom == 2:
            if model_type == 'Kmeans':
                from RobustMMLL.models.POM2.Kmeans.Kmeans import Kmeans_Worker
                self.workerMLmodel = Kmeans_Worker(self.master_address, self.comms, self.logger,  self.verbose, self.Xtr_b)

            elif model_type == 'NN':
                from RobustMMLL.models.POM2.NeuralNetworks.neural_network import NN_Worker
                self.workerMLmodel = NN_Worker(self.master_address, self.comms, self.logger,  self.verbose, self.Xtr_b, self.ytr)

            elif model_type == 'SVM':
                from RobustMMLL.models.POM2.SVM.SVM import SVM_Worker
                self.workerMLmodel = SVM_Worker(self.master_address, self.comms, self.logger,  self.verbose, self.Xtr_b, self.ytr)

            self.display('WorkerNode %s: Created %s model under POM %d' % (str(self.worker_address), model_type, self.pom))

        if self.pom == 3:
            if model_type == 'Kmeans':
                from RobustMMLL.models.POM3.Kmeans.Kmeans import Kmeans_Worker
                self.workerMLmodel = Kmeans_Worker(self.master_address, self.comms, self.logger,  self.verbose, self.Xtr_b)

            elif model_type == 'NN':
                from RobustMMLL.models.POM3.NeuralNetworks.neural_network import NN_Worker
                self.workerMLmodel = NN_Worker(self.master_address, self.comms, self.logger,  self.verbose, self.Xtr_b, self.ytr)

            elif model_type == 'SVM':
                from RobustMMLL.models.POM3.SVM.SVM import SVM_Worker
                self.workerMLmodel = SVM_Worker(self.master_address, self.comms, self.logger,  self.verbose, self.Xtr_b, self.ytr)

            self.display('WorkerNode %s: Created %s model under POM %d' % (str(self.worker_address), model_type, self.pom))


        if self.pom == 4:
            # This object includes general ML tasks, common to all algorithms in POM4
            from RobustMMLL.models.POM4.CommonML.POM4_CommonML import POM4_CommonML_Worker
            self.workerCommonML = POM4_CommonML_Worker(self.master_address, self.worker_address, self.model_type, self.comms, self.logger, self.verbose, self.Xtr_b, self.ytr, cryptonode_address=self.cryptonode_address)
            self.display('WorkerNode %s: Created CommonML model' % str(self.worker_address))

            if model_type == 'LR':
                from RobustMMLL.models.POM4.LR.LR import LR_Worker
                self.workerMLmodel = LR_Worker(self.master_address, self.worker_address, self.model_type, self.comms, self.logger, self.verbose, self.Xtr_b, self.ytr, cryptonode_address=self.cryptonode_address)
                self.display('WorkerNode %s: Created CommonML model' % str(self.worker_address))
                self.model_type = model_type
                self.display('WorkerNode %s: Created %s model' % (str(self.worker_address), model_type))

            if model_type == 'Kmeans':
                from RobustMMLL.models.POM4.Kmeans.Kmeans import Kmeans_Worker
                self.workerMLmodel = Kmeans_Worker(self.master_address, self.worker_address, self.model_type, self.comms, self.logger, self.verbose, self.Xtr_b, self.ytr, cryptonode_address=self.cryptonode_address)
                self.display('WorkerNode %s: Created CommonML model' % str(self.worker_address))
                self.model_type = model_type
                self.display('WorkerNode %s: Created %s model' % (str(self.worker_address), model_type))


        if self.pom == 5:
            from RobustMMLL.models.POM5.CommonML.POM5_CommonML import POM5_CommonML_Worker
            self.workerCommonML = POM5_CommonML_Worker(self.master_address, self.worker_address, self.model_type, self.comms, self.logger, self.verbose, self.Xtr_b, self.ytr)
            self.display('WorkerNode %s: Created CommonML model' % str(self.worker_address))

            if model_type == 'LR':
                from RobustMMLL.models.POM5.LR.LR import LR_Worker
                self.workerMLmodel = LR_Worker(self.master_address, self.worker_address, self.model_type, self.comms, self.logger, self.verbose, self.Xtr_b, self.ytr)
                self.model_type = model_type
                self.display('WorkerNode %s: Created %s model' % (str(self.worker_address), model_type))

            if model_type == 'Kmeans':
                from RobustMMLL.models.POM5.Kmeans.Kmeans import Kmeans_Worker
                self.workerMLmodel = Kmeans_Worker(self.master_address, self.worker_address, self.model_type, self.comms, self.logger, self.verbose, self.Xtr_b)
                self.model_type = model_type
                self.display('WorkerNode %s: Created %s model' % (str(self.worker_address), model_type))

        if self.pom == 6:
            # This object includes general ML tasks, common to all algorithms in POM6
            from RobustMMLL.models.POM6.CommonML.POM6_CommonML import POM6_CommonML_Worker
            self.workerCommonML = POM6_CommonML_Worker(self.master_address, self.worker_address, model_type, self.comms, self.logger, self.verbose, self.Xtr_b, self.ytr)
            self.display('WorkerNode %s: Created CommonML model' % str(self.worker_address))

            if model_type == 'XC':
                from RobustMMLL.models.POM6.XC.XC import XC_Worker
                self.workerMLmodel = XC_Worker(self.master_address, self.worker_address, model_type, self.comms, self.logger, self.verbose, self.Xtr_b, self.ytr)

            if model_type == 'RR':
                from RobustMMLL.models.POM6.RR.RR import RR_Worker
                self.workerMLmodel = RR_Worker(self.master_address, self.worker_address,  model_type, self.comms, self.logger, self.verbose, self.Xtr_b, self.ytr)

            if model_type == 'KR_pm':
                from RobustMMLL.models.POM6.KR_pm.KR_pm import KR_pm_Worker
                self.workerMLmodel = KR_pm_Worker(self.master_address, self.worker_address, model_type, self.comms, self.logger, self.verbose, self.Xtr_b, self.ytr)

            if model_type == 'LC_pm':
                from RobustMMLL.models.POM6.LC_pm.LC_pm import LC_pm_Worker
                self.workerMLmodel = LC_pm_Worker(self.master_address, self.worker_address, model_type, self.comms, self.logger, self.verbose, self.Xtr_b, self.ytr)

            if model_type == 'MLC_pm':
                from RobustMMLL.models.POM6.MLC_pm.MLC_pm import MLC_pm_Worker
                self.workerMLmodel = MLC_pm_Worker(self.master_address, self.worker_address, model_type, self.comms, self.logger, self.verbose, self.Xtr_b, self.ytr)

            if model_type == 'Kmeans_pm':
                from RobustMMLL.models.POM6.Kmeans_pm.Kmeans_pm import Kmeans_pm_Worker
                self.workerMLmodel = Kmeans_pm_Worker(self.master_address, self.worker_address, model_type, self.comms, self.logger, self.verbose, self.Xtr_b)

            self.display('WorkerNode_' + self.model_type + ' %s: Created %s model' % (str(self.worker_address), model_type))



    def run(self):
        """
        Run the main execution loop at the worker
        """
        self.display('WorkerNode_' + self.model_type + ' %s: running %s ...' % (str(self.worker_address), self.model_type))

        if self.pom == 1 or self.pom==2 or self.pom==3:
            self.workerMLmodel.run_worker()

        if self.pom in [4, 5, 6]:
            # the worker checks both workerMLmodel and workerCommonML
            self.workerMLmodel.terminate = False
            self.workerCommonML.terminate = False
            self.workerMLmodel.Xtr_b = self.workerCommonML.Xtr_b
            self.workerMLmodel.ytr = self.workerCommonML.ytr

            while not (self.workerMLmodel.terminate or self.workerCommonML.terminate):  # The worker can be terminated from the MLmodel or CommonML
                # We receive one packet, it could be for Common or MLmodel
                #print('I'm alive!)
                packet, sender = self.workerCommonML.CheckNewPacket_worker()
                if packet is not None:
                    try:
                        #if packet['action'] == 'do_local_prep':
                        #    print('STOP AT ')
                        #    import code
                        #    code.interact(local=locals())
                        if packet['to'] == 'CommonML':
                            #print('worker run CommonML', packet['action'])
                            self.workerCommonML.ProcessReceivedPacket_Worker(packet, sender)
                            if packet['action'] == 'send_encrypter':
                                # Passing the encrypter to workerMLmodel
                                self.workerMLmodel.encrypter = self.workerCommonML.encrypter
                            if packet['action'] == 'do_local_prep':
                                # Passing the transformed training data to workerMLmodel
                                self.workerMLmodel.Xtr_b = self.workerCommonML.Xtr_b
                                self.workerMLmodel.ytr = self.workerCommonML.ytr
                                time.sleep(0.1)                               
                        if packet['to'] == 'MLmodel':
                            #print('worker run MLmodel', packet['action'])
                            self.workerMLmodel.ProcessReceivedPacket_Worker(packet, sender)
                    except Exception as err:
                        #print('ERROR at Workernode', err, str(err), str(type(err)))
                        raise
                        pass


    def get_normalizer(self):
        """
        Returns the normalizer parameters and transform the training data in the workers

        Returns
        ----------
        prep_model: Object
            Preprocessing object
        """
        if not self.workerMLmodel.preprocessor_ready:
            self.display('WorkerNode: Error - Preprocessor not available at the worker')
            return None
        else:
            return self.workerMLmodel.prep_model


    def get_model(self):
        """
        Returns the ML model as an object, if it is trained, returns None otherwise

        Parameters
        ----------
        None
        """

        try:
            model_is_trained = self.workerMLmodel.is_trained
            if not model_is_trained:
                self.display('WorkerNode: Error - Model not trained yet')
                return None
            else:
                return self.workerMLmodel.model
        except:
            self.display('In this POM, the model is not available at WorkerNode.')
            return None



    def save_model(self, output_filename_model=None):
        """
        Saves the ML model using pickle if it is trained, prints an error otherwise

        Parameters
        ----------
        None
        """

        if not self.workerMLmodel.is_trained:
            self.display('WorkerNode: Error - Model not trained yet, nothing to save.')
        else:
            '''
            if output_filename_model is None:
                output_filename_model = './POM' + str(self.pom) + '_' + self.model_type + '_' + self.dataset_name + '_worker_model.pkl'
            '''
            try:
                with open(output_filename_model, 'wb') as f:
                    pickle.dump(self.workerMLmodel.model, f)
            except:
                output_filename_model = './POM' + str(self.pom) + '_' + self.model_type + '_' + self.dataset_name + '_model.pkl'
                with open(output_filename_model, 'wb') as f:
                    pickle.dump(self.workerMLmodel.model, f)

            self.display('WorkerNode: Model saved at %s' %output_filename_model)
