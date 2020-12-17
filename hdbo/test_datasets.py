import datasets
import GPy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pytest

from acquisition_optimizer import MPAcquisitionOptimizer
from GPyOpt.core.task.space import Design_space
from itertools import product
from scipy import optimize

# Complete testing (all domain)
# =============================
# Compare answer with bruteforce
# ------------------------------
# Tests general computation
@pytest.mark.parametrize("i", range(100))
def test_optimization_correctness_large(i):
    compare_bruteforce(i, 10, 2)

# Test computation with more data 
@pytest.mark.parametrize("i", range(10))
def test_optimization_correctness_small(i):
    compare_bruteforce(i, 5, 6)

# Run basic sanity checks only on larger data
# -------------------------------------------
# Tests if the function decompositon is decomposed correctly into the respective parts
@pytest.mark.parametrize("i", range(1000))
def test_optimization_decomposition(i):
    basic_logic_checks_optimizer(i, 30, 1)

@pytest.mark.parametrize("i", range(10))
def test_optimization_compute_large(i):
    basic_logic_checks_optimizer(i, 6, 7)

# Reduced testing (partial domain)
# ================================
@pytest.mark.parametrize("i", range(100))
def test_optimization_reduced_compute_large(i):
    compare_bruteforce_reduced(i, 10, 2, 100)

@pytest.mark.parametrize("i", range(10))
def test_optimization_reduced_compute_small(i):
    compare_bruteforce_reduced(i, 7, 5, 100)

@pytest.mark.parametrize("i", range(1000))
def test_optimization_reduced_decomposition(i):
    compare_bruteforce_reduced(i, 15, 1, 100)

def basic_logic_checks_optimizer(i, dimension, grid_size, max_eval=-1, dataset_name='ErdosRenyiGraph'):
    args = dict(
        dimension=dimension,
        data_random_seed=i,
        grid_params=dict(grid_size=grid_size,
            domain_lower=0.,
            domain_upper=1.),
        kernel_params=dict(
            lengthscale=0.1,
            variance=1.0),
        fn_noise_var= 0.15,
        hash_data=None
        )

    Syn_Loader = datasets.Synthetic.get_loader_constructor(dataset_name)
    syn_loader = Syn_Loader(**args)
    fn = syn_loader._load()

    # Modifies graph
    optimizer = MPAcquisitionOptimizer(fn.domain, fn, X=[], mlflow_logging=None, max_eval=max_eval)
    cfn = fn.make_component_function()

    x_best, f_min, cost = optimizer.optimize(cfn)
    
    # Make sure the component function is the same as reported f_min
    assert(np.isclose(f_min, cfn(x_best)))
    # Make sure that the ground truth is the same as reported f_min
    assert(np.isclose(fn.eval(x_best), f_min))

    print("x_best_opt", x_best[0])
    print("f_min_opt", f_min)
    
    return args, fn, x_best, optimizer._domains

def compare_bruteforce(i, dimension, grid_size, dataset_name='ErdosRenyiGraph'):
    args, fn, x_best, _ = basic_logic_checks_optimizer(i, dimension, grid_size, dataset_name=dataset_name)
    
    # We now brute force 
    rranges = tuple([ slice(0, args['grid_params']['grid_size'], 1) ] * args['dimension'] )
    x_bruteforce = optimize.brute(fn.eval_indexed, rranges, full_output=False, finish=None)
    x_bruteforce = fn.domain.translate(x_bruteforce.astype(int))
    
    # Make sure that the brute force solution is the same as the solution found by optimizer
    assert(np.all(np.isclose(x_bruteforce, x_best[0])))

def compare_bruteforce_reduced(i, dimension, grid_size, max_eval, dataset_name='ErdosRenyiGraph'):
    np.random.seed(i)
    args, fn, x_best, domain = basic_logic_checks_optimizer(i, dimension, grid_size, max_eval, dataset_name=dataset_name)

    rranges = np.array([ domain[k] for k in range(len(domain))])
    
    # Additional check to ensure that the domain is respected
    for xi, di in zip(x_best[0], rranges):
        assert(xi in di)

    # We now brute force on the reduced domain
    f_min_bf = np.inf
    x_bruteforce = None
    for x in product(*rranges):
        x = np.array(x)
        f_min_curr = fn.eval(x)
        if f_min_curr < f_min_bf:
            f_min_bf = f_min_curr
            x_bruteforce = x
    
    print("x_bruteforce", x_bruteforce)
    print("f_min_bf", f_min_bf)
    
    # Make sure that the brute force solution is the same as the solution found by optimizer
    assert(np.all(np.isclose(x_bruteforce, x_best[0])))

#compare_bruteforce(0, 5, 1)
#compare_bruteforce_reduced(0, 5, 1, 100)
compare_bruteforce_reduced(0, 196, 150, 1000, 'DebugGraph')