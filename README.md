# High-Dimensional Bayesian Optimization via Tree-Structured Additive Models (AAAI 2021)

Authors:

1. Eric Han
2. Ishank Arora
3. Jonathan Scarlett

This repository is accompanied by the AAAI 2021 publication -- [arXiv Pre-Print](https://arxiv.org/abs/2012.13088).
Cite using our AAAI2021 article:
```
@article{Han_Arora_Scarlett_2021, 
title={High-Dimensional Bayesian Optimization via Tree-Structured Additive Models}, 
volume={35}, 
url={https://ojs.aaai.org/index.php/AAAI/article/view/16933},
abstractNote={Bayesian Optimization (BO) has shown significant success in tackling expensive low-dimensional black-box optimization problems. Many optimization problems of interest are high-dimensional, and scaling BO to such settings remains an important challenge. In this paper, we consider generalized additive models in which low-dimensional functions with overlapping subsets of variables are composed to model a high-dimensional target function. Our goal is to lower the computational resources required and facilitate faster model learning by reducing the model complexity while retaining the sample-efficiency of existing methods. Specifically, we constrain the underlying dependency graphs to tree structures in order to facilitate both the structure learning and optimization of the acquisition function. For the former, we propose a hybrid graph learning algorithm based on Gibbs sampling and mutation. In addition, we propose a novel zooming-based algorithm that permits generalized additive models to be employed more efficiently in the case of continuous domains. We demonstrate and discuss the efficacy of our approach via a range of experiments on synthetic functions and real-world datasets.}, 
number={9}, 
journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
author={Eric Han and Ishank Arora and Jonathan Scarlett}, 
year={2021}, 
month={May}, 
pages={7630-7638}}
```
## Acknowledgements

1. The code in this repository are derived from the code base from [High-Dimensional Bayesian Optimization via Additive Models with Overlapping Groups](https://arxiv.org/pdf/1802.07028.pdf), supplied by Paul Rolland.
2. The code included in `hdbo/febo` is taken from [LineBO](https://github.com/jkirschner42/LineBO). The paper accompanying the code is [Adaptive and Safe Bayesian Optimization in High Dimensions via One-Dimensional Subspaces](https://arxiv.org/abs/1902.03229).
3. The code included in `hdbo/boattack` is taken from [BayesOpt_Attack](https://github.com/rubinxin/BayesOpt_Attack). The paper accompanying the code is [BayesOpt Adversarial Attack](https://openreview.net/pdf?id=Hkem-lrtvH).
4. The NAS-Bench-101 datasets included in `data/fcnet` is taken from [nas_benchmarks](https://github.com/automl/nas_benchmarks). The paper accompanying the code is [NAS-Bench-101: Towards Reproducible Neural Architecture Search](https://arxiv.org/pdf/1902.09635.pdf).
5. The lpsolve datasets included in `data/mps` is taken from the benchmark dataset in [MIPLIB 2017](https://miplib.zib.de/download.html). 

## Repository Structure

The code repository is structured as follows:

* `data`: Folder containing the data files from NAS-Bench-101 and lpsolve.
* `hdbo`: Folder containing our code
* `sample`: Folder containing some sample configurations of the experiments are given below; Note that We ran many configurations with different random seeds, algorithms, and datasets.
   * `default.yml`: Sample experiment to check if your environment is installed correctly.
   * `syn-add-d.yml`: Sample synthetic additive (discrete) experiment
   * `syn-add-c.yml`: Sample synthetic additive (continuous) experiment
   * `syn-fn.yml`: Sample synthetic function experiment
   * `nas.yml`: Sample NAS-Bench-101 experiment
   * `lp.yml`: Sample lpsolve experiment
   * `ba-addgp.yml`: Sample BayesOpt Attack experiment
* `test`: Folder related to some of our unit tests
* `MLproject`: MLflow project file, we are using mlflow to help organize our experiments
* `README.md`: This readme
* `hdbo.yml`: Conda file that is used internally by the MLflow project

## Setup

We implemented all algorithms in Python 3.8.3. 
The Python environments are managed using Conda, and experiments are managed using [MLflow](https://www.mlflow.org), which allows convenient management of experiments.
We have included code from data from external sources in our repository for the ease of setup. 

Minimum System requirements:

* `Linux 5.4.12-100`
* 5GB of space (Because of NAS-Bench-101 and lpsolve, if not just 100MB)

Prepare your environment:

1. Run `data/setup.sh`, it will download data from [NAS Benchmarks](https://github.com/automl/nas_benchmarks) and  into your home directory. You may skip this step if you are not running lpsolve and NAS-Bench-101 datasets.
2. [Install Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)
3. [Install MLflow](https://mlflow.org/) either on your system or in the base environment of Conda - `pip install mlflow`

## Running

Check your installation by running `mlflow run .` in the base directory, to run with the test configuration `default.yml`, you should see something similar as output:
```
2020/02/06 15:16:12 INFO mlflow.projects: === Created directory /tmp/tmpya1pguie for downloading remote URIs passed to arguments of type 'path' ===
2020/02/06 15:16:12 INFO mlflow.projects: === Running command 'source activate mlflow-69104b8ea3ca24a4e45a47bc7581b825cfe12300 && LOGGING_TYPE=local python hdbo/main.py /home/X/Workspace/HD-BO-Additive-Models/sample/default.yml' in run with ID '9ca30276ec574aefbae92a2befc53d60' === 
06/02/2020 15:16:13 [INFO    ] [main.py:22] Platform: uname_result(system='Linux', node='X', release='5.4.12-100.fc30.x86_64', version='#1 SMP Wed Jan 15 00:38:53 UTC 2020', machine='x86_64', processor='x86_64')
06/02/2020 15:16:13 [INFO    ] [main.py:23] Processor: x86_64
06/02/2020 15:16:13 [INFO    ] [main.py:24] Python: 3.6.8/CPython
06/02/2020 15:16:13 [INFO    ] [main.py:27] Blas Library: ['blas', 'cblas', 'lapack', 'blas', 'cblas', 'lapack', 'blas', 'cblas', 'lapack']
06/02/2020 15:16:13 [INFO    ] [main.py:28] Lapack Library: ['blas', 'cblas', 'lapack', 'blas', 'cblas', 'lapack', 'blas', 'cblas', 'lapack']
06/02/2020 15:16:13 [INFO    ] [main.py:31] temp_dir: /tmp/tmph4iceyxt
06/02/2020 15:16:13 [INFO    ] [main.py:32] Host Name: X
06/02/2020 15:16:13 [INFO    ] [datasets.py:44] Load loader[Synthetic].
06/02/2020 15:16:13 [INFO    ] [datasets.py:70] Using synthetic dataset loader with graph_type[StarGraph].
06/02/2020 15:16:13 [INFO    ] [datasets.py:493] Graph Edges: [(0, 1), (0, 2), (0, 3)]
06/02/2020 15:16:13 [INFO    ] [datasets.py:321] Loading pre-computed function at /home/X/cache/2525f51218fc4e2827eab0aec1776a94abf895ca.pkl.
06/02/2020 15:16:13 [INFO    ] [datasets.py:325] Checking consistency of pre-compute.
06/02/2020 15:16:13 [INFO    ] [main.py:71] f_min = -5.562785863192289
06/02/2020 15:16:13 [INFO    ] [main.py:72] x_best = [[0.56 0.   0.16 0.74]]
06/02/2020 15:16:13 [INFO    ] [main.py:73] cost = 7803
06/02/2020 15:16:13 [INFO    ] [algorithms.py:41] Load algorithm loader[Algorithm].
06/02/2020 15:16:13 [INFO    ] [algorithms.py:54] Using algorithm with algorithm_id[Tree].
06/02/2020 15:16:13 [INFO    ] [gp.py:49] initializing Y
06/02/2020 15:16:13 [INFO    ] [gp.py:98] initializing inference method
06/02/2020 15:16:13 [INFO    ] [gp.py:107] adding kernel and likelihood as parameters
06/02/2020 15:16:17 [INFO    ] [function_optimizer.py:481] Running Tree
06/02/2020 15:16:18 [INFO    ] [function_optimizer.py:59] New graph : [(1, 3)]
06/02/2020 15:16:22 [INFO    ] [function_optimizer.py:481] Running Tree
06/02/2020 15:16:23 [INFO    ] [function_optimizer.py:59] New graph : [(0, 1), (0, 2), (0, 3)]
06/02/2020 15:16:29 [INFO    ] [function_optimizer.py:481] Running Tree
06/02/2020 15:16:30 [INFO    ] [function_optimizer.py:59] New graph : [(0, 1), (0, 2), (0, 3)]
06/02/2020 15:16:36 [INFO    ] [function_optimizer.py:481] Running Tree
06/02/2020 15:16:36 [INFO    ] [function_optimizer.py:59] New graph : [(0, 1), (0, 2), (0, 3)]
06/02/2020 15:16:43 [INFO    ] [function_optimizer.py:481] Running Tree
06/02/2020 15:16:43 [INFO    ] [function_optimizer.py:59] New graph : [(0, 1), (0, 2), (0, 3)]
06/02/2020 15:16:50 [INFO    ] [function_optimizer.py:481] Running Tree
06/02/2020 15:16:51 [INFO    ] [function_optimizer.py:59] New graph : [(0, 1), (0, 2), (0, 3)]
2020/02/06 15:16:57 INFO mlflow.projects: === Run (ID '9ca30276ec574aefbae92a2befc53d60') succeeded ===
```

You may run the configurations by using the command to specify the configuration, for example to run the synthetic additive (continuous) experiment: 
```
mlflow run . -P param_file=sample/syn-add-c.yml
```

In order to visualize the experiment, you can run `mlflow ui` and click on the experiments to visualize the metrics.
