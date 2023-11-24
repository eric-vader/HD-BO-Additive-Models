#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
# =============

import logging
import os
import numpy as np
from common import Config

logging_types = ['local', 'server']
# Determines if we are logging on the server or not
def logger(env_key='LOGGING_TYPE'):
    
    logging_type = os.getenv(env_key, 'local')

    log_formatter = logging.Formatter('%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    info_file_handler = logging.FileHandler(Config().log_file("info.log"))
    info_file_handler.setFormatter(log_formatter)
    info_file_handler.setLevel(logging.INFO)
    root_logger.addHandler(info_file_handler)
    
    error_file_handler = logging.FileHandler(Config().log_file("error.log"))
    error_file_handler.setFormatter(log_formatter)
    error_file_handler.setLevel(logging.ERROR)
    root_logger.addHandler(error_file_handler)
    
    if logging_type == 'local':
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(log_formatter)
        consoleHandler.setLevel(logging.INFO)
        root_logger.addHandler(consoleHandler)
        
# =====================================================================
import datasets
import algorithms
import yaml
import sys
import hashlib
import json
import boto3

# Argument Constants
syn_graphtype_list = datasets.Synthetic.loader_ids
nas_bench_list = datasets.NAS.loader_ids
# Check avaliable fcnet benchmark files
fcnet_benchmark_list = Config().list_fcnet()

MLFLOW_S3_ENDPOINT_URL = os.getenv('MLFLOW_S3_ENDPOINT_URL')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

# Must return a 
def args_parse():


    # We guess if config is a file or not
    exp_config = sys.argv[1]
    extension = os.path.splitext(exp_config)[-1]

    if extension == ".yml":
        if not os.path.isfile(exp_config):
            raise FileNotFoundError(f"Cannot experiment file at {exp_config}")
        logging.info(f"Load exp file from {exp_config}")
        args_dict = yaml.load(open(exp_config), yaml.FullLoader)
    else:
        endpoint_url = MLFLOW_S3_ENDPOINT_URL
        aws_access_key_id = AWS_ACCESS_KEY_ID
        aws_secret_access_key = AWS_SECRET_ACCESS_KEY
        jobs_port='9000'
        region_name='us-east-1'
        signature_version='s3v4'

        if not endpoint_url:
            endpoint_url = "http://scarlettgpu2.d2.comp.nus.edu.sg"
            aws_access_key_id = "miniomlflow"
            aws_secret_access_key = "R9RqzmC1"

        s3 = boto3.client('s3', 
            endpoint_url=endpoint_url+':'+jobs_port,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            config=boto3.session.Config(signature_version=signature_version),
            region_name=region_name)
        
        obj = s3.get_object(Bucket='mlflow-jobs', Key=exp_config)
        args_dict = yaml.load(obj['Body'].read().decode('utf-8'), yaml.FullLoader)

    # args_dict = yaml.load(open(sys.argv[1]), yaml.FullLoader)
    args_dict["hash_exe"] = hashlib.sha1(json.dumps(args_dict, sort_keys=True).encode()).hexdigest()

    which_datas = list(args_dict["data_type"].keys())
    if len(which_datas) != 1:
        logging.error('There should be only one data type declared.')

    which_algorithms = list(args_dict["algorithm_type"].keys())
    if len(which_algorithms) != 1:
        logging.error('There should be only one algorithm type declared.')

    # Process the 2 types, data and algo type
    args_dict["which_data"] = which_datas[0]
    args_dict["which_algorithm"] = which_algorithms[0]

    args_dict["hash_data"] = hashlib.sha1(json.dumps(args_dict["data_type"], sort_keys=True).encode()).hexdigest()

    # Flatten to the parameters
    args_dict.update(args_dict["data_type"][args_dict["which_data"]])
    del args_dict["data_type"]
    args_dict.update(args_dict["algorithm_type"][args_dict["which_algorithm"]])
    del args_dict["algorithm_type"]

    # turn it into lambda if string
    if type(args_dict["exploration_weight"]) == str:
        args_dict["exploration_weight"] =  eval(args_dict["exploration_weight"])

    return args_dict
