from function_optimizer import GraphFunction
from GPyOpt.models.gpmodel import GPModel
import numpy as np
import logging
from datasets import BattackFn

# This is the prior model
class MyGPModel(GPModel):
    def __init__(self, noise_var, exact_feval, optimize_restarts, exploration_weight_function, learnDependencyStructureRate, learnParameterRate, mlflow_logging, graph_function, fn):
        self.graph_function = graph_function
        self.fn = fn
        self.has_logged_inital = False
        _, kernel_full, self.cfn = graph_function.make_decomposition(self)
        self.t = 0
        self.exploration_weight_function = exploration_weight_function
        self.learnDependencyStructureRate = learnDependencyStructureRate
        if learnParameterRate == None:
            self.learnParameterRate = learnDependencyStructureRate
        else:
            self.learnParameterRate = learnParameterRate
        self.mlflow_logging = mlflow_logging
        super(MyGPModel, self).__init__(kernel=kernel_full, noise_var=noise_var, exact_feval=exact_feval, optimize_restarts=optimize_restarts)
    
    def predict(self, X, with_noise=True):
        raise Exception
    def predict_withGradients(self, X):
        raise Exception

    def predict_with_kernel(self, X, kernel):
        if X.ndim == 1:
            X = X[None,:]
        # self.model -> GPRegression
        #self.model.kern = kernel
        #m, v = self.model.predict(X, full_cov=False, include_likelihood=True)
        m, v = self.model.predict(X, kern=kernel, full_cov=False, include_likelihood=True)
        # Stability issues?
        v = np.clip(v, 1e-10, np.inf)
        # We can take the square root because v is just a diagonal matrix of variances
        return m, np.sqrt(v)

    def predict_withGradients_with_kernel(self, X, kernel):
        if X.ndim==1: X = X[None,:]
        m, v = self.model.predict(X, kern=kernel, full_cov=False, include_likelihood=True)
        v = np.clip(v, 1e-10, np.inf)

        dmdx, dvdx = self.model.predictive_gradients(X, kern=kernel)
        dmdx = dmdx[:,:,0]
        dsdx = dvdx / (2*np.sqrt(v))
        return m, np.sqrt(v), dmdx, dsdx


    def exploration_weight(self):
        return self.exploration_weight_function(self.t)
    
    # Update t when the model is updated
    def updateModel(self, X_all, Y_all, X_new, Y_new):
        super(MyGPModel, self).updateModel(X_all, Y_all, X_new, Y_new)

        if self.t == 0:
            assert(len(self.fn.history_y) == len(Y_all))
            # Attempt to return f instead of y if that exist
            self.mlflow_logging.log_init_y(np.min(self.fn.history_y))
        else:
            assert(len(self.fn.history_y) == len(Y_all))
            self.mlflow_logging.log_y(np.min(self.fn.history_y[-1]))
        self.t+=1

    def update_structure(self, acquisition, X_all, Y_all, fn_optimizer, fn):

        if self.learnDependencyStructureRate < 0:
            return

        # bayes attack, mimicking their run behavior of running the 
        bayes_attack_style = (isinstance(fn, BattackFn) and self.t == 1)

        # Decide when to learn new structure
        if bayes_attack_style or self.t % self.learnDependencyStructureRate == 0:
            if not self.has_logged_inital:
                # Log the inital graph
                fn.mlflow_logging.log_graph_metrics(self.graph_function.graph)
                self.has_logged_inital = True

            # Optimal never reaches here...

            # We do not need to learn the following
            # best_tree, lengthscales, dimensional_lengthscale
            Y_vect = Y_all.flatten()

            fn_optimizer.optimize(X_all, Y_vect, self.graph_function)
            fn.mlflow_logging.log_graph_metrics(self.graph_function.graph)

            # Make acquisitions
            # ========================================================================
            # Update the decomposition used
            logging.debug("Dim Param: {}".format(self.graph_function.dimensional_parameters))
            _, self.kernel, self.cfn = self.graph_function.make_decomposition(self)
            self.model = None
        elif self.t % self.learnParameterRate == 0:
            # learn the parameters
            
            Y_vect = Y_all.flatten()
            fn_optimizer.optimize_parameters(X_all, Y_vect, self.graph_function)

            logging.debug("Dim Param: {}".format(self.graph_function.dimensional_parameters))
            _, self.kernel, self.cfn = self.graph_function.make_decomposition(self)
            self.model = None