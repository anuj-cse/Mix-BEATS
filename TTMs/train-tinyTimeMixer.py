import os
import random 


os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

import time
import math 
import tempfile 
import torch 
import pickle 
import logging 
import warnings
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

import argparse


from transformers import Trainer, TrainingArguments, set_seed, EarlyStoppingCallback


from tinytimemixer.configuration_tinytimemixer import TinyTimeMixerConfig
from tinytimemixer.modeling_tinytimemixer import TinyTimeMixerForPrediction

from tsfm_public.toolkit.dataset import PretrainDFDataset, ForecastDFDataset
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
from tsfm_public.toolkit.util import select_by_index

from torch.utils.data import ConcatDataset


warnings.filterwarnings("ignore")
SEED = 42
set_seed(SEED)


def load_dataset_from_folders(folder_path, args):
    datasets = []
    for dataset in os.listdir(folder_path):
        building_path = folder_path + '/' + dataset
        for building in os.listdir(building_path):
            building_id = building.rsplit(".csv",1)[0]

            data = pd.read_csv(building_path + '/' + building, parse_dates=[args["timestamp_column"]])

            tsp = TimeSeriesPreprocessor(
                context_length=args["context_length"],
                timestamp_column=args["timestamp_column"],
                id_columns=args["id_columns"],
                target_columns=args["forecast_columns"],
                scaling=args["is_scaling"]
            )
            tsp.train(data)

            dataset = ForecastDFDataset(
                tsp.preprocess(data),
                timestamp_column=args["timestamp_column"],
                id_columns=args["id_columns"],
                target_columns=args["forecast_columns"],
                context_length=args["context_length"],
                prediction_length=args["prediction_length"],
                stride=args["patch_stride"],
            )
            
            
            datasets.append(dataset)

    combinedDataset = ConcatDataset(datasets)
    return combinedDataset





def dataset_split(args):

    train_folder = os.path.join(args["dataset_path"], "train")
    val_folder = os.path.join(args["dataset_path"], "val")

    train_dataset = load_dataset_from_folders(train_folder, args)
    val_dataset = load_dataset_from_folders(val_folder, args)

    return train_dataset, val_dataset




def model_config(args):

    config = TinyTimeMixerConfig(
        context_length=args["context_length"],
        patch_length=args["patch_length"],
        num_input_channels=args["num_input_channels"],
        patch_stride=args["patch_stride"],
        d_model=args["d_model"],
        num_layers=args["num_layers"],
        expansion_factor=args["expansion_factor"],
        dropout=args["dropout"],
        head_dropout=args["head_dropout"],
        mode=args["mode"][0],
        scaling=args["scaling"],
        prediction_length=args["prediction_length"],
        is_scaling=args["is_scaling"],
        gated_attn=args["gated_attn"],
        norm_mlp=args["norm_mlp"],
        self_attn=args["self_attn"],
        self_attn_heads=args["self_attn_heads"],
        use_positional_encoding=args["use_positional_encoding"],
        positional_encoding_type=args["positional_encoding_type"],
        loss=args["loss"],
        init_std=args["init_std"],
        post_init=args["post_init"],
        norm_eps=args["norm_eps"],
        adaptive_patching_levels=args["adaptive_patching_levels"],
        resolution_prefix_tuning=args["resolution_prefix_tuning"],
        frequency_token_vocab_size=args["frequency_token_vocab_size"],
        distribution_output=args["distribution_output"],
        num_parallel_samples=args["num_parallel_samples"],
        decoder_num_layers=args["decoder_num_layers"],
        decoder_d_model=args["decoder_d_model"],
        decoder_adaptive_patching_levels=args["decoder_adaptive_patching_levels"],
        decoder_raw_residual=args["decoder_raw_residual"],
        decoder_mode=args["decoder_mode"],
        use_decoder=args["use_decoder"],
        enable_forecast_channel_mixing=args["enable_forecast_channel_mixing"],
        fcm_gated_attn=args["fcm_gated_attn"],
        fcm_context_length=args["fcm_context_length"],
        fcm_use_mixer=args["fcm_use_mixer"],
        fcm_mix_layers=args["fcm_mix_layers"],
        fcm_prepend_past=args["fcm_prepend_past"], 
        init_linear=args["init_linear"],
        init_embed=args["init_embed"],

    )

    pretraining_model = TinyTimeMixerForPrediction(config)
    return pretraining_model

def train(args, model, train_dataset, eval_dataset):

    training_arguments  = TrainingArguments(
        output_dir = args["output_dir"],
        logging_dir = args["logging_dir"],
        overwrite_output_dir = args["overwrite_output_dir"],
        do_eval = args["do_eval"],
        dataloader_num_workers = args["num_workers"],
        report_to = args["report_to"],
        per_device_train_batch_size=args["per_device_train_batch_size"],
        per_device_eval_batch_size=args["per_device_eval_batch_size"],
        num_train_epochs=args["num_epochs"],
        evaluation_strategy=args["evaluation_strategy"],
        save_strategy=args["save_strategy"],
        save_total_limit=args["save_total_limit"],
        logging_strategy=args["logging_strategy"],
        load_best_model_at_end=args["load_best_model_at_end"],
        label_names=args["label_names"],
        learning_rate=args["learning_rate"],
        metric_for_best_model = args["metric_for_best_model"],
        greater_is_better = args["greater_is_better"],
    )

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=args["early_stopping_patience"],
        early_stopping_threshold=args["early_stopping_threshold"],
    )


    pretrainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[early_stopping_callback],
    )


    pretrainer.train()
    os.makedirs(args["model_save_dir"],exist_ok=True)
    pretrainer.save_model(args["model_save_dir"])
    model.config.to_json_file(args["config_file"])


if __name__ == '__main__':

    # get arguments
    parser = argparse.ArgumentParser(description='Time Series Forecasting')
    parser.add_argument('--config-file', type=str, default='./config/tinyTimeMixer.json', help='Input config file path', required=True)
    file_path_arg = parser.parse_args()
    config_file = file_path_arg.config_file
    # config_file = './config/tinyTimeMixer.json'
    with open(config_file, 'r') as f:
        args = json.load(f)
    
    # split dataset
    train_dataset, eval_dataset = dataset_split(args)

    # configure model
    model = model_config(args)

    # train model
    train(args, model, train_dataset, eval_dataset)

   

































# def dataset_split(args):

#     trainDataset = []
#     testDataset = []
#     evalDataset = []


#     for dataset in os.listdir(args["dataset_path"]):
#         building_path = args["dataset_path"] + '/' + dataset
#         for building in os.listdir(building_path):
#             building_id = building.rsplit(".csv",1)[0]

#             df = pd.read_csv(building_path + '/' + building, parse_dates=[args["timestamp_column"]])
            
#             n_samples = df.shape[0]

#             train_sample = int(0.7*n_samples)
#             eval_sample = int(0.8*n_samples)

#             train_data = df.iloc[:train_sample]
#             eval_data = df.iloc[train_sample: eval_sample]
#             test_data = df.iloc[eval_sample:]

#             tsp = TimeSeriesPreprocessor(
#                 context_length=args["context_length"],
#                 timestamp_column=args["timestamp_column"],
#                 id_columns=args["id_columns"],
#                 target_columns=args["forecast_columns"],
#                 scaling=args["is_scaling"]
#             )
#             tsp.train(train_data)

#             train_dataset = ForecastDFDataset(
#                 tsp.preprocess(train_data),
#                 timestamp_column=args["timestamp_column"],
#                 id_columns=args["id_columns"],
#                 target_columns=args["forecast_columns"],
#                 context_length=args["context_length"],
#                 prediction_length=args["prediction_length"],
#                 stride=args["patch_stride"],
#             )
            
#             eval_dataset = ForecastDFDataset(
#                 tsp.preprocess(eval_data),
#                 timestamp_column=args["timestamp_column"],
#                 id_columns=args["id_columns"],
#                 target_columns=args["forecast_columns"],
#                 context_length=args["context_length"],
#                 prediction_length=args["prediction_length"],
#                 stride=args["patch_stride"],
#             )

#             test_dataset = ForecastDFDataset(
#                 tsp.preprocess(test_data),
#                 timestamp_column=args["timestamp_column"],
#                 id_columns=args["id_columns"],
#                 target_columns=args["forecast_columns"],
#                 context_length=args["context_length"],
#                 prediction_length=args["prediction_length"],
#                 stride=args["patch_stride"],
#             )
            
#             trainDataset.append(train_dataset)
#             evalDataset.append(eval_dataset)
#             testDataset.append(test_dataset)

#     combinedTrainDataset = ConcatDataset(trainDataset)
#     combinedEvalDataset = ConcatDataset(evalDataset)
#     combinedTestDataset = ConcatDataset(testDataset)

#     return combinedTrainDataset, combinedEvalDataset, combinedTestDataset

