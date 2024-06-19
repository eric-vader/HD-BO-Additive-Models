#!/usr/bin/env python3
import subprocess
import json
import re
import logging
import itertools
import collections
import math
from collections import defaultdict
from itertools import combinations
from functools import partial

import networkx as nx
import numpy as np
import GPy
import matplotlib.pyplot as plt
import pickle
import mlflow
import os.path
import h5py
import json
import sys
import lpsolve_config
from common import Config
from hpolib.benchmarks import synthetic_functions
from GPyOpt.core.task.space import Design_space

def getDecompositionFromGraph(graph):
    cliques = nx.find_cliques(graph)
    decomp = []
    for c in cliques:
        decomp.append(sorted(c))
    return decomp

class MetaLoader(type):
    registry = {}
    loader_ids = []
    def __new__(cls, cls_name, bases, attrs):
        new_class = super(cls, MetaLoader).__new__(cls, cls_name, bases, attrs)
        MetaLoader.registry[cls_name] = new_class
        MetaLoader.loader_ids.append(cls_name)
        return new_class
    @staticmethod
    def get_loader_constructor(loader_id):
        logging.info("Load loader[%s].", loader_id)
        return MetaLoader.registry[loader_id]

class NAS(type, metaclass=MetaLoader):
    registry = {}
    loader_ids = []
    def __new__(cls, cls_name, bases, attrs):
        new_class = super(cls, NAS).__new__(cls, cls_name, bases, attrs)
        NAS.registry[cls_name] = new_class
        NAS.loader_ids.append(cls_name)
        return new_class
    @staticmethod
    def get_loader_constructor(bench_type, **kwargs):
        logging.info("Using NAS dataset loader with bench_type[%s].", bench_type)
        return NAS.registry[bench_type]

class Synthetic(type, metaclass=MetaLoader):
    registry = {}
    loader_ids = []
    def __new__(cls, cls_name, bases, attrs):
        new_class = super(cls, Synthetic).__new__(cls, cls_name, bases, attrs)
        Synthetic.registry[cls_name] = new_class
        Synthetic.loader_ids.append(cls_name)
        return new_class
    @staticmethod
    def get_loader_constructor(graph_type, **kwargs):
        logging.info("Using synthetic dataset loader with graph_type[%s].", graph_type)
        return Synthetic.registry[graph_type]

class LPSolve(type, metaclass=MetaLoader):
    registry = {}
    loader_ids = []
    def __new__(cls, cls_name, bases, attrs):
        new_class = super(cls, LPSolve).__new__(cls, cls_name, bases, attrs)
        LPSolve.registry[cls_name] = new_class
        LPSolve.loader_ids.append(cls_name)
        return new_class
    @staticmethod
    def get_loader_constructor(problem_type, **kwargs):
        logging.info("Using LPSolve problem type[%s].", problem_type)
        return LPSolve.registry[problem_type]

class Hpolib(type, metaclass=MetaLoader):
    registry = {}
    loader_ids = []
    def __new__(cls, cls_name, bases, attrs):
        new_class = super(cls, Hpolib).__new__(cls, cls_name, bases, attrs)
        Hpolib.registry[cls_name] = new_class
        Hpolib.loader_ids.append(cls_name)
        return new_class
    @staticmethod
    def get_loader_constructor(hpo_fn, **kwargs):
        logging.info("Using Hpolib function[%s].", hpo_fn)
        return Hpolib.registry[hpo_fn]

class BayesianAttack(type, metaclass=MetaLoader):
    registry = {}
    loader_ids = []
    def __new__(cls, cls_name, bases, attrs):
        new_class = super(cls, BayesianAttack).__new__(cls, cls_name, bases, attrs)
        BayesianAttack.registry[cls_name] = new_class
        BayesianAttack.loader_ids.append(cls_name)
        return new_class
    @staticmethod
    def get_loader_constructor(nn_data, **kwargs):
        logging.info("Using BayesianAttack function[%s].", nn_data)
        return BayesianAttack.registry[nn_data]

class Simple(type, metaclass=MetaLoader):
    registry = {}
    loader_ids = []
    def __new__(cls, cls_name, bases, attrs):
        new_class = super(cls, Simple).__new__(cls, cls_name, bases, attrs)
        Simple.registry[cls_name] = new_class
        Simple.loader_ids.append(cls_name)
        return new_class
    @staticmethod
    def get_loader_constructor(simple_fn, **kwargs):
        logging.info("Using Simple function[%s].", simple_fn)
        return Simple.registry[simple_fn]

class Domain(Design_space):
    def __init__(self, dimension, combined_domain):
        self.dimension = dimension
        self.combined_domain = combined_domain
        super(Domain, self).__init__(self.get_gpy_domain())
    def get_gpy_domain(self):
        gpy_domain = [{'name': 'x_{}'.format(i), 'type': 'discrete', 'domain': tuple(d), 'dimensionality' : 1 } for i, d in enumerate(self.combined_domain)]
        return gpy_domain
    def get_opt_domain(self):
        space = {}
        space['type'] = 'discrete'
        space['domain'] = self.combined_domain
        return space
    def none_value(self):
        return np.array([-1] * self.dimension, dtype=np.float)
    def random_X(self, rs, n_rand):
        # Pick from each dimension's domain 
        X_T = []
        for ea_d in self.combined_domain:
            X_T.append(rs.choice(ea_d, n_rand, replace=True))
        return np.array(X_T).T
        
class SyntheticDomain(Domain):
    # fingerprints are the binary version of the fingerprints output by the Chem package
    def __init__(self, dimension, grid_size, domain_lower, domain_upper):
        self.grid_size = grid_size
        
        # The actual discretized domain in any dimension
        self.X_domain = np.linspace(domain_lower, domain_upper, grid_size)
        self.index_domain = list(range(self.grid_size))
        super(SyntheticDomain, self).__init__(dimension, [self.X_domain] * dimension)
    def generate_grid(self, dim):
        # This generates a N-Dim Grid
        return np.array(np.meshgrid(*[self.X_domain] * dim)).T.reshape(-1, dim)
    def translate(self, X_indices):
        return self.X_domain[X_indices]

from functools import reduce
# Barebones component function
class ComponentFunction(dict):
    def __init__(self, fn_decomp_lookup):
        self.__dict__ = fn_decomp_lookup
        
    def __call__(self, x):
        # Does not matter the sequence
        '''
        for ea_cfn_f in self.__dict__.values():
            f_parts.append(ea_cfn_f(x))
        return np.sum(f_parts,axis=0)
        '''
        #return np.array([ ea_cfn(x) for ea_cfn in self.__dict__.values() ]).sum(axis=0)
        return reduce(np.add, map(lambda ea: ea(x), self.__dict__.values()))
    def acq_f_df(self, x):
        
        f_parts = []
        g_parts = []
        for ea_cfn in self.__dict__.values():
            f_part, g_part = ea_cfn.acquisition_function_withGradients(x)
            f_parts.append(f_part)
            g_parts.append(g_part)
        return np.sum(f_parts,axis=0), np.sum(g_parts,axis=0)
        #return np.array([ ea_cfn.acquisition_function_withGradients(x) for ea_cfn in self.__dict__.values() ]).sum(axis=0)
    def __setitem__(self, key, item):
        self.__dict__[key] = item
    def __getitem__(self, key):
        return self.__dict__[key]
    def __repr__(self):
        return repr(self.__dict__)
    def __len__(self):
        return len(self.__dict__)
    def __delitem__(self, key):
        del self.__dict__[key]
    def clear(self):
        return self.__dict__.clear()
    def copy(self):
        return self.__dict__.copy()
    def has_key(self, k):
        return k in self.__dict__
    def update(self, *args, **kwargs):
        return self.__dict__.update(*args, **kwargs)
    def keys(self):
        return self.__dict__.keys()
    def values(self):
        return self.__dict__.values()
    def items(self):
        return self.__dict__.items()
    def pop(self, *args):
        return self.__dict__.pop(*args)
    def __cmp__(self, dict_):
        return self.__cmp__(self.__dict__, dict_)
    def __contains__(self, item):
        return item in self.__dict__
    def __iter__(self):
        return iter(self.__dict__)
    def __unicode__(self):
        return unicode(repr(self.__dict__))


# Synthetic Component function
class SyntheticComponentFunction(ComponentFunction):
    def __init__(self, graph, fn_decomp_lookup):

        # Find all maximal cliques
        cfns = [ tuple(sorted(c_vertices)) for c_vertices in nx.find_cliques(graph)] 
        cfns = sorted(cfns, key=lambda x : -len(x))

        all_fns = set()
        
        cfn_dict = {}
        for cfn in cfns:
            if len(cfn) == 1:
                cfn_decomposition = set( [ fn_decomp_lookup[cfn] ] )
            else:
                cfn_decomposition = set( [ fn_decomp_lookup[tuple(sorted(edge))] for edge in combinations(cfn, 2) ])

            # Make sure we do not have repeated cliques, or empty functions
            cfn_decomposition = cfn_decomposition - all_fns
            if len(cfn_decomposition) == 0:
                continue

            all_fns.update(cfn_decomposition)

            
            def cfn_eval(x, _cfn_decomposition = cfn_decomposition):
                return np.array([ ea_cfn(x) for ea_cfn in _cfn_decomposition ]).sum(axis=0)

            cfn_dict[cfn] = cfn_eval
        super(SyntheticComponentFunction, self).__init__(cfn_dict)

class Function(object):
    def __init__(self, domain):
        self.domain = domain
        self.history_y = []
    def eval(self, x):
        raise NotImplementedError
    def __call__(self, x):
        # This call is with noise added
        y = self.eval(x)
        self.history_y.append(y)
        return y
    def get_emb_dim(self):
        return max(2, int(np.sqrt(self.domain.dimension)))
    def has_synthetic_noise(self):
        return False

class NoisyFunction(Function):
    def __init__(self, domain, rs, fn_noise_var):
        Function.__init__(self, domain)
        self.rs = rs
        self.fn_noise_sd = np.sqrt(fn_noise_var)
    def __call__(self, x):
        y = self.eval(x)
        self.history_y.append(y)
        return self.rs.normal(0, self.fn_noise_sd) + y
    def has_synthetic_noise(self):
        return True

# Function for NAS test
class ConfigLosses(Function):
    # fingerprints are the binary version of the fingerprints output by the Chem package

    def __init__(self, parameters, key_map, domain, data, rs):
        Function.__init__(self, domain)
        self.parameters = parameters
        self.dim = len(parameters)
        self.data = data
        self.key_map = key_map
        self.rs = rs
        self.graph = None
    def eval(self, x):
        x = x[0]

        index = self.rs.randint(4)
        config_dict = {}
        for i in range(len(self.parameters)):
            config_dict[self.parameters[i]] = self.key_map[i][x[i]]
        k = json.dumps(config_dict, sort_keys=True)
        valid = self.data[k]["valid_mse"][index]
        return valid[-1]

# https://stackoverflow.com/questions/47370718/indexing-numpy-array-by-a-numpy-array-of-coordinates
def ravel_index(b, shp):
    return np.concatenate((np.asarray(shp[1:])[::-1].cumprod()[::-1],[1])).dot(b)

# Synthetic
class FunctionValues(NoisyFunction):
    # fingerprints are the binary version of the fingerprints output by the Chem package
    def __init__(self, f_list, v_list, domain, fn_decomposition, graph, kernel_params, rs, fn_noise_var):
        NoisyFunction.__init__(self, domain, rs, fn_noise_var)
        self.f_list = f_list
        self.v_list = v_list
        self.fn_decomposition = fn_decomposition
        # True graph and true lengthscale
        self.graph = graph
        self.kernel_params = kernel_params
        self.v_flat = [ list(v) for v in v_list ]
    def eval(self, x):
        x_i = np.searchsorted(self.domain.X_domain, x)
        # This actually evaluates the function
        return self.eval_indexed(x_i)
    def eval_indexed(self, x_i):
        return sum([ self._part_eval(ea_v, ea_f, x_i) for ea_v, ea_f in zip(self.v_flat, self.f_list) ])
    def part_eval(self,index_f, x):
        x_i = np.searchsorted(self.domain.X_domain, x)
        return [[ self._part_eval(self.v_flat[index_f], self.f_list[index_f], ea_x_i) ] for ea_x_i in x_i ]
    # Evaluate only that edge, internal function
    def _part_eval(self, ea_v, ea_f, x):
        return np.take(ea_f, ravel_index(np.take(x, ea_v), ea_f.shape))
    def make_component_function(self):
        fn_decomp_lookup = {}
        for i, decomp in enumerate(self.fn_decomposition):
            fn_decomp_lookup[decomp] = partial(self.part_eval, i)
        return SyntheticComponentFunction(self.graph, fn_decomp_lookup)

class Loader(object):
    def __init__(self, dataID, hash_data, **kwargs):
        self.dataID = dataID
        self.kwargs = kwargs
        self.hash_data = hash_data
    def get_dataset_id(self):
        return self.__class__.__name__ + self.dataID
    def load(self):
        # Save the ground truth network, if it exist
        self.log_true_graph()

        cached_file_path = self.cached_file_path()
        if os.path.isfile(cached_file_path):
            logging.info("Loading pre-computed function at {}.".format(cached_file_path))
            with open(cached_file_path, 'rb') as handle:
                fn, soln = pickle.load(handle)
                if isinstance(self, NetworkxGraph):
                    logging.info("Checking consistency of pre-compute.")
                    assert(nx.is_isomorphic(fn.graph, self.get_nx_graph()) )

            # Super hacks, for compatibility purposes
            # TODO
            if not hasattr(fn.domain, 'model_dimensionality'):
                super(Domain, fn.domain).__init__(fn.domain.get_gpy_domain())

            return fn, soln
        else:
            logging.info("No pre-computed function, computing.")
            return self._load(), None
    def cached_file_path(self):
        return Config().cache_file('{}.pkl'.format(self.hash_data))
    def save(self, fn, soln):
        cached_file_path = self.cached_file_path()
        logging.info("Saving pre-computed function at {}.".format(cached_file_path))
        with open(cached_file_path, 'wb') as handle:
            pickle.dump((fn, soln), handle, protocol=pickle.HIGHEST_PROTOCOL)
    def _load(self):
        raise NotImplementedError
    def log_true_graph(self):
        pass

class NASLoader(Loader):
    def __init__(self, dataID, dimension, data_random_seed, hyper_values, key_map, parameters, **kwargs):
        Loader.__init__(self, '{}Nas-DRS{}-D{}'.format(dataID, data_random_seed, dimension), **kwargs)
        self.rs = np.random.RandomState(data_random_seed)
        self.hyper_values = hyper_values
        self.key_map = key_map
        self.parameters = parameters
        self.dimension = dimension
    def load(self):
        tabular_benchmark_path = self.tabular_benchmark_path()
        # No precompute, we load directly
        if os.path.isfile(tabular_benchmark_path):
            logging.info("Found fcnet benchmark at {}".format(tabular_benchmark_path))
        else:
            logging.fatal("Required fcnet benchmark file - {} is not found".format(tabular_benchmark_path))

        data = h5py.File(tabular_benchmark_path, "r")

        best_k = None
        best_validation_error = np.inf
        for k in data.keys():
            validation_error = np.min(data[k]["valid_mse"][:, -1])
            if best_validation_error > validation_error:
                best_k = k
                best_validation_error = validation_error
                
        soln = (best_k, best_validation_error, len(data.keys()))
        config_losses = ConfigLosses(parameters=self.parameters, key_map=self.key_map, domain=Domain(self.dimension, self.hyper_values), data=data, rs=self.rs)

        return config_losses, soln

class FcnetLoader(NASLoader, metaclass=NAS):
    def __init__(self, fcnet_filename, **kwargs):
        dimension = 9
        hyper_values = [
            np.array([0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0]),
            np.array([0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0]),
            np.array([0.0, 0.5, 1.0]),
            np.array([0.0, 0.5, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.005, 0.01, 0.05, 0.1, 0.5, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.125, 0.25, 0.5, 1.0])
        ]
        key_map = [
            {0.03125:16, 0.0625:32, 0.125:64, 0.25:128, 0.5:256, 1.0:512},
            {0.03125:16, 0.0625:32, 0.125:64, 0.25:128, 0.5:256, 1.0:512},
            {0.0:0.0, 0.5:0.3, 1.0:0.6},
            {0.0:0.0, 0.5:0.3, 1.0:0.6},
            {0.0: 'relu', 1.0: 'tanh'},
            {0.0: 'relu', 1.0: 'tanh'},
            {0.005:0.0005, 0.01:0.001, 0.05:0.005, 0.1:0.01, 0.5:0.05, 1.0:0.1},
            {0.0: 'cosine', 1.0: 'const'},
            {0.125:8, 0.25:16, 0.5:32, 1.0:64}
        ]
        parameters = ["n_units_1", "n_units_2", "dropout_1", "dropout_2", "activation_fn_1", "activation_fn_2", "init_lr", "lr_schedule", "batch_size"]
        NASLoader.__init__(self, "", dimension=dimension, hyper_values=hyper_values, key_map=key_map, parameters=parameters, **kwargs)
        self.fcnet_filename = fcnet_filename
    def tabular_benchmark_path(self):
        return Config().fcnet_file("{}.hdf5".format(self.fcnet_filename))

class SyntheticLoader(Loader):
    def __init__(self, dataID, dimension, kernel_params, data_random_seed, grid_params, fn_noise_var, **kwargs):
        
        self.fn_noise_var = fn_noise_var
        self.kernel_params = kernel_params

        # Unpack kernel parameter for easy use
        lengthscale = kernel_params["lengthscale"]
        variance = kernel_params["variance"]

        grid_size = grid_params["grid_size"]
        domain_lower = grid_params["domain_lower"]
        domain_upper = grid_params["domain_upper"]

        Loader.__init__(self, '{}Syn-DRS{}-D{}-Grid{}[{},{}]-L{}V{}'.format(dataID, data_random_seed, dimension, grid_size, domain_lower, domain_upper, lengthscale, variance), **kwargs)
        self.dimension = dimension
        self.rs = np.random.RandomState(data_random_seed)
        self.domain = SyntheticDomain(dimension, grid_size, domain_lower, domain_upper)

        # We will not compute the functions as per cliques as its computationally intractable
        self.cliques = list(nx.find_cliques(self.get_nx_graph()))
        
        # We will instead decompose it to 1D and 2D functions
        # TODO REFACTOR
        self.fn_decomposition = [ tuple(sorted([v])) for v in nx.isolates(self.get_nx_graph())] + [ tuple(sorted(e)) for e in self.get_nx_graph().edges() ]

        # Lengthscale belongs to each function
        # ground truth lengthscale
        self.lengthscale = lengthscale
        if type(lengthscale) == float or type(lengthscale) == int:
            self.dimensional_lengthscale = [lengthscale] * len(self.fn_decomposition)
        else:
            raise NotImplementedError

        self.variance = variance
        if type(variance) == float or type(variance) == int:
            self.dimensional_variance = [variance] * len(self.fn_decomposition)
        else: 
            raise NotImplementedError

    def get_nx_graph(self):
        raise NotImplementedError
    def generate_functions(self):
        # Group GPs by dimension and lengthscale
        # We do this so we can compute really quickly
        dim_ls_dict = defaultdict(list)
        for v, ls, variance in zip(self.fn_decomposition, self.dimensional_lengthscale, self.dimensional_variance):
            dim_ls_dict[( len(v), ls, variance )].append(v)

        v_list = []
        f_list = []
        for k in dim_ls_dict:
            v_dim, ls, variance = k
            variables = dim_ls_dict[k]
            f_list += list(self.generate_functions_same_distribution(len(variables), ls, variance, v_dim))
            v_list += list(variables)

        # Generate for all length 2, with given lengthscale
        return f_list, v_list
    # Generates n_functions of GP with dim dimensions with the same lengthscale
    def generate_functions_same_distribution(self, n_functions, lengthscale, variance, v_dim):
        N = self.domain.grid_size
        grid = self.domain.generate_grid(v_dim)
        ker = GPy.kern.RBF(input_dim=v_dim, lengthscale=lengthscale, variance=variance)
        mu = np.zeros(N**v_dim) #(N*N)
        C = ker.K(grid, grid) #(N*N)
        # The following function will generate n_functions * (N*N)
        fun = self.rs.multivariate_normal(mu, C, (n_functions), check_valid='raise')
        target_shape = (n_functions,) + (N,) * v_dim
        # Which will need to be reshaped to n_functions * N * N
        return fun.reshape(target_shape)
    def log_true_graph(self):

        # self.layout = dict()
        # for i in range(3):
        #     for j in range(3):
        #         self.layout[i*3+j] = (i,j)
        # print(self.layout)
        self.layout = nx.spring_layout(self.get_nx_graph(), iterations=10000, seed=6)
        # print(self.layout)
        nx.draw(self.get_nx_graph(), cmap = plt.get_cmap('jet'), with_labels=True, pos=self.layout)
        plt.savefig(Config().data_file('ground_truth_graph.png'))
        plt.clf()
    def _load(self):
        f_list, v_list = self.generate_functions()
        self.function = FunctionValues(f_list, v_list, self.domain, self.fn_decomposition, self.get_nx_graph(), self.kernel_params, self.rs, self.fn_noise_var)

        return self.function

class NetworkxGraph(SyntheticLoader):
    def __init__(self, dimension, data_random_seed, **kwargs):
        self.data_random_seed = data_random_seed
        G = self.make_graph(dimension)
        G = nx.freeze(nx.convert_node_labels_to_integers(G))
        self.true_dependency_graph = G
        logging.info("Graph Edges: {}".format(G.edges()))
        SyntheticLoader.__init__(self, "-NetX", dimension, data_random_seed=data_random_seed, **kwargs)
    def make_graph(self, dimension):
        raise NotImplementedError
    def get_nx_graph(self):
        return self.true_dependency_graph

class EmptyGraph(NetworkxGraph, metaclass=Synthetic):
    def make_graph(self, dimension):
        return nx.empty_graph(dimension)

class PathGraph(NetworkxGraph, metaclass=Synthetic):
    def make_graph(self, dimension):
        return nx.path_graph(dimension)

class TestGraph(NetworkxGraph, metaclass=Synthetic):
    def make_graph(self, dimension):
        return nx.random_partition_graph([1,1,2,2,3],1,0)
        
class ErdosRenyiGraph(NetworkxGraph, metaclass=Synthetic):
    def make_graph(self, dimension):
        return nx.erdos_renyi_graph(dimension, np.random.RandomState(self.data_random_seed).rand(), seed=self.data_random_seed)

class StarGraph(NetworkxGraph, metaclass=Synthetic):
    def make_graph(self, dimension):
        #Return the Star graph with n+1 nodes: one center node, connected to n outer nodes.
        return nx.star_graph(dimension-1)

class GridGraph(NetworkxGraph, metaclass=Synthetic):
    def make_graph(self, dimension):
        n = int(np.sqrt(dimension))
        # Ensure perfect square
        assert(np.isclose(n**2, dimension))
        return nx.grid_2d_graph(n, n)

class GridLargeGraph(GridGraph, metaclass=Synthetic):
    def load(self):
        # Save the ground truth network, if it exist
        self.log_true_graph()

        cached_file_path = self.cached_file_path()
        if os.path.isfile(cached_file_path):
            logging.info("Loading pre-computed function at {}.".format(cached_file_path))
            with open(cached_file_path, 'rb') as handle:
                fn, soln = pickle.load(handle)
                if isinstance(self, NetworkxGraph):
                    logging.info("Checking consistency of pre-compute.")
                    assert(nx.is_isomorphic(fn.graph, self.get_nx_graph()) )

            return fn, soln
        else:
            logging.info("Pre-computed function has no answer")
            fn, soln = super().load()
            soln = (None, 0, 0)
            self.save(fn, soln)
            return fn, soln

class GridGraph34(NetworkxGraph, metaclass=Synthetic):
    def make_graph(self, dimension):
        assert(dimension == 12)
        return nx.grid_2d_graph(4, 3)

class PartitionGraph(NetworkxGraph, metaclass=Synthetic):
    def make_graph(self, dimension):
        assert(dimension == 12)
        return nx.random_partition_graph([3,3,3,3],1,0)

class SparseErdosRenyiGraph(NetworkxGraph, metaclass=Synthetic):
    def make_graph(self, dimension):
        return nx.erdos_renyi_graph(dimension, 2.0/dimension, seed=self.data_random_seed)

class PowerlawTree(NetworkxGraph, metaclass=Synthetic):
    def make_graph(self, dimension):
        sys.setrecursionlimit(1500)
        return nx.random_powerlaw_tree(dimension, seed=self.data_random_seed, tries=dimension**2)

class AncestryGraph(NetworkxGraph, metaclass=Synthetic):
    def __init__(self, dimension, data_random_seed, **kwargs):
        self.data_random_seed = data_random_seed
        G = self.make_graph(dimension)
        G = nx.freeze(G)
        self.true_dependency_graph = G
        logging.info("Graph Edges: {}".format(G.edges()))
        SyntheticLoader.__init__(self, "-NetX", dimension, data_random_seed=data_random_seed, **kwargs)

    def make_graph(self, dimension):
        # recursion limit
        assert(dimension == 132)
        G, self.shells = pickle.load(open("data/ancestry.pkl", 'rb'))
        return G
    def log_true_graph(self):
        # plt.rcParams['figure.figsize'] = [15, 15]
        pos = nx.shell_layout(self.get_nx_graph(), self.shells)
        nx.draw(self.get_nx_graph(), cmap = plt.get_cmap('jet'), with_labels=True, pos=pos)
        plt.savefig(Config().data_file('ground_truth_graph.png'))
        plt.clf()

class DebugGraph(NetworkxGraph, metaclass=Synthetic):
    def __init__(self, dimension, data_random_seed, **kwargs):
        self.data_random_seed = data_random_seed
        G = self.make_graph(dimension)
        G = nx.freeze(G)
        self.true_dependency_graph = G
        logging.info("Graph Edges: {}".format(G.edges()))
        SyntheticLoader.__init__(self, "-NetX", dimension, data_random_seed=data_random_seed, **kwargs)
    def make_graph(self, dimension):
        # recursion limit
        G = nx.read_gpickle("graph.pkl")
        assert(dimension == len(G.nodes()))
        return G

# =======================================================================
class MpsLoader(Loader, metaclass=LPSolve):
    def __init__(self, mps_filename, infinite, time_limit, max_floor, **kwargs):
        
        self.dimension = lpsolve_config.dimension
        print(lpsolve_config.dimension)
        self.hyper_values = lpsolve_config.hyper_values
        self.key_map = lpsolve_config.key_map
        self.parameters = lpsolve_config.parameters
        
        self.mps_filename = mps_filename
        self.infinite = infinite
        self.time_limit = time_limit
        self.max_floor = max_floor
        Loader.__init__(self, '{}-Mps-D{}'.format(mps_filename, self.dimension), **kwargs)
    def mps_path(self):
        return Config().mps_file("{}.mps".format(self.mps_filename))
    def load(self):
        mps_path = self.mps_path()
        # No precompute, we load directly
        if os.path.isfile(mps_path):
            logging.info("Found MPS File at {}".format(mps_path))
        else:
            logging.fatal("Required MPS File - {} is not found".format(mps_path))

        config_losses = ExecuteLPSolve(parameters=self.parameters, key_map=self.key_map, domain=Domain(self.dimension, self.hyper_values), mps_path=mps_path, infinite=self.infinite, time_limit=self.time_limit, max_floor=self.max_floor)

        return config_losses, (None, 0, 0)

# Function to execute LP Solve
class ExecuteLPSolve(Function):
    def __init__(self, parameters, key_map, domain, mps_path, infinite, time_limit, max_floor):
        Function.__init__(self, domain)
        self.parameters = parameters
        self.dim = len(parameters)
        self.mps_path = mps_path
        self.key_map = key_map
        self.graph = None
        self.infinite = infinite
        self.time_limit = time_limit
        self.max_floor = max_floor
    def eval(self, x):
        print(x)
        args = json.dumps({
            "x":x.tolist(),
            "mps_path":self.mps_path,
            "infinite":self.infinite,
            "time_limit":self.time_limit})
        
        #os.system('python ./hdbo/lpsolve.py \'{}\' '.format(args))
        cmd = 'python ./hdbo/lpsolve.py \'{}\' '.format(args)
        logging.debug(cmd)
        try:
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True, timeout=self.time_limit*5)
            logging.info(output)
            obj_val_str = re.findall(r"RETURN_OBJECTIVE_VALUE:\((\d*(?:\.\d*)?(?:e[\+|-]{0,1}\d+){0,1})\)", str(output))
            objective = min(float(obj_val_str[0]), self.max_floor)
        except Exception as e:
            #logging.exception("LPSOLVE Exception")
            #logging.error("cmd: {}".format(cmd))
            objective = self.max_floor
        
        logging.info("Objective Value: {}".format(objective))
        
        return objective
# =======================================================================
class HpolibLoader(Loader):
    def __init__(self, data_random_seed, grid_size, fn_noise_var, **kwargs):
        Loader.__init__(self, "Hpolib", **kwargs)
        self.hpo_fn = self.make_hpo_fn()
        self.grid_size = grid_size
        self.rs = np.random.RandomState(data_random_seed)
        self.fn_noise_var = fn_noise_var
    def load(self):
        
        info = self.hpo_fn.get_meta_information()
        soln = (info['optima'], info['f_opt'], None)
        domain = HpoDomain(self.grid_size, info['bounds'])
        hpo_fn_wrapper = HpolibWrapper(domain, self.hpo_fn, self.rs, self.fn_noise_var)
        
        return hpo_fn_wrapper, soln

# Function to execute LP Solve
class HpolibWrapper(NoisyFunction):
    def __init__(self, domain, hpo_fn, rs, fn_noise_var):
        NoisyFunction.__init__(self, domain, rs, fn_noise_var)
        self.hpo_fn = hpo_fn
        self.graph = None
    def eval(self, X):
        x = X[0]
        return self.hpo_fn(x)

class HpoDomain(SyntheticDomain):
    # Note that it does not work well when the grid too uneven
    # Uniformly distribute the grid up
    def __init__(self, grid_size, hpo_bounds):
        self.grid_size = grid_size
        dimension = len(hpo_bounds)
        
        joint_domain = []
        all_lower, all_upper = hpo_bounds[0]
        for domain_lower, domain_upper in hpo_bounds:
            joint_domain.append(np.linspace(domain_lower, domain_upper, grid_size))

        self.index_domain = list(range(self.grid_size))
        Domain.__init__(self, dimension, joint_domain)

# Some popular functions
# ========================
class Rosenbrock20D(HpolibLoader, metaclass=Hpolib):
    def make_hpo_fn(self):
        return synthetic_functions.rosenbrock.Rosenbrock20D()

class Hartmann6(HpolibLoader, metaclass=Hpolib):
    def make_hpo_fn(self):
        return synthetic_functions.Hartmann6()

class Camelback(HpolibLoader, metaclass=Hpolib):
    def make_hpo_fn(self):
        return synthetic_functions.Camelback()

# Some popular functions
# ========================

# ================================================================================
# Permute the domain but
class HpolibAugLoader(HpolibLoader):
    def __init__(self, aug_dimension, **kwargs):
        HpolibLoader.__init__(self, **kwargs)
        self.aug_dimension = aug_dimension

        info = self.hpo_fn.get_meta_information()
        self.actual_dimension = len(info['bounds'])
        total_dimension = self.actual_dimension + self.aug_dimension

        # Compute the permutations
        self.per = self.rs.permutation(total_dimension)
        self.inv_per = np.argsort(self.per)

    def load(self):
        
        info = self.hpo_fn.get_meta_information()

        opt_x = np.concatenate([info['optima'][0], np.zeros(self.aug_dimension)])[self.per]
        soln = (opt_x, info['f_opt'], None)

        # This is fine because the bounds are uniform
        bounds = info['bounds']
        lowers, uppers = zip(*bounds)
        print(bounds)

        self.aug_lower = min(lowers)
        self.aug_upper = max(uppers)

        bounds = bounds + [ [self.aug_lower, self.aug_upper] for i in range(self.aug_dimension) ]
        bounds = np.array(bounds)
        
        domain = HpoDomain(self.grid_size, bounds[self.per])
        hpo_fn_wrapper = HpolibAugWrapper(domain, self.actual_dimension, self.inv_per, self.hpo_fn, self.rs, self.fn_noise_var)
        
        return hpo_fn_wrapper, soln

class Hartmann6Aug(HpolibAugLoader, metaclass=Hpolib):
    def make_hpo_fn(self):
        return synthetic_functions.Hartmann6()

class CamelbackAug(HpolibAugLoader, metaclass=Hpolib):
    def make_hpo_fn(self):
        return synthetic_functions.Camelback()

# Function to execute LP Solve
class HpolibAugWrapper(HpolibWrapper):
    def __init__(self, domain, actual_dimension, inv_per, hpo_fn, rs, fn_noise_var):
        HpolibWrapper.__init__(self, domain, hpo_fn, rs, fn_noise_var)
        self.inv_per = inv_per
        self.actual_dimension = actual_dimension
    def eval(self, X):
        X = X[0] # Compatibility
        X = X[self.inv_per]  # undo permutation
        X = X[:self.actual_dimension]  # take active dimensions
        return self.hpo_fn(X)
    def get_emb_dim(self):
        return self.actual_dimension

# ================================================================================
class Gaussian(Loader, metaclass=Simple):
    def __init__(self, dimension, grid_size, initial_value, **kwargs):
        Loader.__init__(self, "Gaussian", **kwargs)
        self.domain = SyntheticDomain(dimension, grid_size, -1.0, 1.0)
        self.initial_value = initial_value
        self.soln = (np.zeros(dimension), -1.0, None)
    def load(self):
        return SimpleSyntheticFn(self.domain), self.soln

class SimpleSyntheticFn(Function):
    def __init__(self, domain):
        Function.__init__(self, domain)
        self.graph = None
    def eval(self, X):
        X = np.atleast_2d(X[0])
        Y = np.exp(-4*np.sum(np.square(X), axis=1))[0]
        return -Y

class Stybtang(Loader, metaclass=Simple):
    def __init__(self, dimension, grid_size, data_random_seed, fn_noise_var, **kwargs):
        Loader.__init__(self, "Stybtang", **kwargs)
        self.rs = np.random.RandomState(data_random_seed)
        self.fn_noise_var = fn_noise_var
        self.domain = SyntheticDomain(dimension, grid_size, -4.0, 4.0)
        self.soln = (np.array([ -2.903534 ] * dimension), -39.16599*dimension, None)
    def load(self):
        return StybtangFn(self.domain, self.rs, self.fn_noise_var), self.soln

class StybtangFn(NoisyFunction):
    def __init__(self, domain, rs, fn_noise_var):
        NoisyFunction.__init__(self, domain, rs, fn_noise_var)
        self.graph = None
    def eval(self, X):
        X = np.atleast_2d(X[0])
        Y = np.sum(X**4 -16.*X**2 + 5.*X, axis=1)/2.
        return Y

# Bayesian Attack ================================================================
from boattack.bayesopt import Bayes_opt
from boattack.objective_func.objective_functions_tf import CNN
from boattack.utilities.upsampler import upsample_projection
from boattack.utilities.utilities import get_init_data

class BattackFn(NoisyFunction):
    def __init__(self, f_adapted, x_bounds_adapted, results_file_name, failed_file_name, nchannel, low_dim, high_dim, dim_reduction, obj_func, cnn, **kwargs):
        NoisyFunction.__init__(self, **kwargs)
        self.f_adapted = f_adapted
        self.x_bounds_adapted = x_bounds_adapted
        self.results_file_name = results_file_name
        self.failed_file_name = failed_file_name
        self.nchannel = nchannel
        self.low_dim = low_dim
        self.high_dim = high_dim
        self.dim_reduction = dim_reduction
        self.obj_func = obj_func
        self.cnn = cnn
        self.graph = None
    def eval(self, X):
        return self.f_adapted(X)

    # super hacks to get the mlflow_logging to cnn
    # TODO
    @property
    def mlflow_logging(self):
        return self._mlflow_logging
    @mlflow_logging.setter
    def mlflow_logging(self,value):
        self.cnn.mlflow_logging = value
        self._mlflow_logging = value

class BattackDomain(SyntheticDomain):
    # fingerprints are the binary version of the fingerprints output by the Chem package
    def __init__(self, dimension, grid_size, domain_lower, domain_upper, x_bounds_adapted):
        super().__init__(dimension, grid_size, domain_lower, domain_upper)
        self.x_bounds_adapted = x_bounds_adapted
    def random_X(self, rs, n_rand):
        inital_design, y_init = get_init_data(obj_func=None, n_init=n_rand, bounds=self.x_bounds_adapted)
        return inital_design

class BayesianAttackLoader(Loader):
    def __init__(self, obj_func, model_type, seed, tg,
                img_offset, low_dim=2304, batch_size=1, acq_type='LCB', num_iter=40, target_label=0, dim_reduction='BILI',
                cost_metric=None, obj_metric=2, update_freq=10, high_dim=None, 
                nchannel=None, epsilon=None, grid_size=None, fn_noise_var=None, **kwargs):
        
        Loader.__init__(self, f'BayesianAttack-{obj_func}', **kwargs)
        self.rs = np.random.RandomState(seed)
        self.fn_noise_var = fn_noise_var

        # This is to prevent some strange error.
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
                except RuntimeError as e:
                    print(e)

        # Specify directory to store all the trash
        #directory = './'
        directory = Config().ba_path + '/'
        model_directory = Config().ba_models + '/'

        if 'LDR' in model_type:
            low_dim = high_dim

        if dim_reduction == 'NONE':
            x_bounds = np.vstack([[-1, 1]] * high_dim * nchannel)
        else:
            x_bounds = np.vstack([[-1, 1]] * low_dim * nchannel)

        # Specify the experiment results saving directory
        results_data_folder = f'{directory}exp_results/{obj_func}_tf_{model_type}_ob{obj_metric}_' \
                            f'_freq{update_freq}_ld{low_dim}_{dim_reduction}/'
        if not os.path.exists(results_data_folder):
            os.makedirs(results_data_folder)

        # Define the model and the original images to be attacked
        cnn = CNN(dataset_name=obj_func, img_offset=img_offset, epsilon=epsilon,
                dim_reduction=dim_reduction, low_dim=low_dim, high_dim=high_dim,
                obj_metric=obj_metric, results_folder=results_data_folder,
                directory=model_directory, rs=self.rs)
        
        # For each image, define the target class
        '''
        if ntargets > 1:
            target_list = list(range(ntargets))
        else:
            target_list = [target_label]
        '''
        # Start attack each target in sequence
        cnn.get_data_sample(tg)
        input_label = cnn.input_label
        img_id = cnn.orig_img_id
        target_label = cnn.target_label[0]

        '''
        from collections import Counter
        cnt = Counter()
        for i in range(10):
            self.rs = np.random.RandomState(seed)
            cnn = CNN(dataset_name=obj_func, img_offset=i, epsilon=epsilon,
                dim_reduction=dim_reduction, low_dim=low_dim, high_dim=high_dim,
                obj_metric=obj_metric, results_folder=results_data_folder,
                directory=model_directory, rs=self.rs)
            cnn.get_data_sample(tg)
            print(cnn.input_label)
            cnt[cnn.input_label] += 1
        print(cnt)
        input()
        '''
        
        logging.info(f'id={img_offset}, origin={input_label}, target={target_label}, eps={epsilon}, dr={low_dim}')

        # Define the BO objective function
        if obj_func == 'imagenet':
            if 'LDR' in model_type or dim_reduction == 'NONE':
                f = lambda x: cnn.np_evaluate_bili(x)
            else:
                f = lambda x: cnn.np_upsample_evaluate_bili(x)
        else:
            if 'LDR' in model_type or dim_reduction == 'NONE':
                f = lambda x: cnn.np_evaluate(x)
            else:
                f = lambda x: cnn.np_upsample_evaluate(x)

        # Define the name of results file and failure fail(for debug or resume)
        results_file_name = os.path.join(results_data_folder,
                                            f'{model_type}{acq_type}{batch_size}_{dim_reduction}_d{low_dim}_i{input_label}_t{target_label}_id{img_id}')
        failed_file_name = os.path.join(results_data_folder,
                                        f'failed_{model_type}{acq_type}{batch_size}_{dim_reduction}_d{low_dim}_i{input_label}_t{target_label}_id{img_id}')
        # low_dim is the pixel dimension, need to add the channels.
        self.ba_fn = BattackFn(f, x_bounds, results_file_name, failed_file_name, nchannel, low_dim, high_dim, dim_reduction, obj_func, cnn, 
            domain=BattackDomain(len(x_bounds), grid_size, -1.0, 1.0, x_bounds), rs=self.rs, fn_noise_var=self.fn_noise_var)

    def load(self):
        return self.ba_fn, (None, 0, 0)


class Mnist(BayesianAttackLoader, metaclass=BayesianAttack):
    def __init__(self, **kwargs):
        high_dim = 784
        nchannel = 1
        epsilon = 0.3
        super().__init__(high_dim=high_dim, nchannel=nchannel, epsilon=epsilon, obj_func='mnist', **kwargs)

class Cifar10(BayesianAttackLoader, metaclass=BayesianAttack):
    def __init__(self, **kwargs):
        high_dim = int(32 * 32)
        nchannel = 3
        epsilon = 0.05
        super().__init__(high_dim=high_dim, nchannel=nchannel, epsilon=epsilon, obj_func='cifar10', **kwargs)

class Imagenet(BayesianAttackLoader, metaclass=BayesianAttack):
    def __init__(self, **kwargs):
        high_dim = int(96 * 96)
        nchannel = 3
        epsilon = 0.05
        super().__init__(high_dim=high_dim, nchannel=nchannel, epsilon=epsilon, obj_func='imagenet', **kwargs)