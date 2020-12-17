import numpy as np
import networkx as nx
import GPyOpt
from GPyOpt.optimization.acquisition_optimizer import AcquisitionOptimizer
from myAcquisitionModular import MyAcquisitionModular
from myGPModel import MyGPModel
from acquisition_optimizer import MPAcquisitionOptimizer
from GPyOpt.core.evaluators.sequential import Sequential
from GPyOpt.core.task.objective import SingleObjective
from GPyOpt.core.task.cost import CostModel

import networkx as nx

class MyBOModular(GPyOpt.core.BO):
    
    def __init__(self, domain, initial_design, graph_function,
        normalize_Y=False, max_eval=-1,
        fn=None, fn_optimizer=None, noise_var=None, exact_feval=None, exploration_weight_function=None, learnDependencyStructureRate=None, learnParameterRate=None,
        acq_opt_restarts=1):

        #self.design_space = Design_space(domain.get_gpy_domain())

        self.fn = fn
        self.objective = SingleObjective(self.fn, 1, "no name", space=domain)
        self._init_design(initial_design)

        self.domain = domain

        self.acquisition_optimizer = MPAcquisitionOptimizer(domain, graph_function, [], self.fn.mlflow_logging, max_eval=max_eval, acq_opt_restarts=acq_opt_restarts)
        #self.acquisition_optimizer = AcquisitionOptimizer(domain)

        # model needed for LCB
        self.model = MyGPModel(noise_var=noise_var, exact_feval=exact_feval, optimize_restarts=0, 
            exploration_weight_function=exploration_weight_function, learnDependencyStructureRate=learnDependencyStructureRate, 
            learnParameterRate=learnParameterRate, graph_function=graph_function, mlflow_logging=self.fn.mlflow_logging, fn=self.fn)

        ## !!! models inside acqu1 must be the same as models in MyModel !!! -> Ok in Python, the object are references, not copied
        self.acquisition = MyAcquisitionModular(self.model, self.acquisition_optimizer, domain)
        self.evaluator = Sequential(self.acquisition)
        
        self.modular_optimization = False
        
        self.cost = CostModel(None)
        self.fn_optimizer = fn_optimizer

        super(MyBOModular, self).__init__(model = self.model, space = domain, objective = self.objective,
            acquisition = self.acquisition, evaluator = self.evaluator, X_init = self.X, Y_init = self.Y,
            cost = self.cost, normalize_Y = normalize_Y, model_update_interval = 1)

    def _init_design(self, initial_design):
        self.X = initial_design
        self.Y, _ = self.objective.evaluate(self.X)

    def _update_model(self, normalization_type):
        """
        Updates the model and saves the parameters (if available).
        """
        self.model.update_structure(self.acquisition, self.X, self.Y, self.fn_optimizer, self.fn)
        super(MyBOModular, self)._update_model()
    
    def _save_model_parameter_values(self):
        return
