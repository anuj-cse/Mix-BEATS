import os
import random 

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1" 

import time
import math 
import tempfile 
import pickle 
import logging 
import warnings
import json
import torch
import argparse
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from math import sqrt

from transformers import Trainer, TrainingArguments, set_seed, EarlyStoppingCallback

from tinytimemixer.configuration_tinytimemixer import TinyTimeMixerConfig
from tinytimemixer.modeling_tinytimemixer import TinyTimeMixerForPrediction

from tsfm_public.toolkit.dataset import PretrainDFDataset, ForecastDFDataset
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
from tsfm_public.toolkit.util import select_by_index

from torch.utils.data import ConcatDataset

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# metrics used for evaluation
def cal_cvrmse(pred, true, eps=1e-8):
    pred = np.array(pred)
    true = np.array(true)
    return np.power(np.square(pred - true).sum() / pred.shape[0], 0.5) / (true.sum() / pred.shape[0] + eps)

def cal_mae(pred, true):
    pred = np.array(pred)
    true = np.array(true)
    return np.mean(np.abs(pred - true))

def cal_nrmse(pred, true, eps=1e-8):
    true = np.array(true)
    pred = np.array(pred)

    M = len(true) // 24
    y_bar = np.mean(true)
    NRMSE = 100 * (1/ (y_bar+eps)) * np.sqrt((1 / (24 * M)) * np.sum((true - pred) ** 2))
    return NRMSE

def denormalize(load, load_max, load_min):
    return load * (load_max - load_min) + load_min



# check that window have all zeros or not
def is_zero_only_window(context):
    return np.all(context == 0)

# Filter function to remove zero-only context windows
def filter_zero_context_windows(dataset):
    filtered_data = []
    for data_point in dataset:
        if not is_zero_only_window(data_point["past_values"]):
            filtered_data.append(data_point)
    return filtered_data




# test dataset          
def dataset_split(args, building_path, building):

    datasets = []

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
    return combinedDataset, tsp





# evaluate model and calculate metrics
def eval_model(args, trainer, test_dataset, tsp):

    true_list = []
    pred_list = []
    input_list = []

    output = trainer.predict(test_dataset)
    output = output[0][0]

    for i in range(0, len(test_dataset)):

        input = np.array(test_dataset[i]["past_values"])
        true = np.array(test_dataset[i]["future_values"])
        pred = np.array(output[i, :, :])



        # inverse scale
        df_true = pd.DataFrame(true, columns=['energy'])
        df_pred = pd.DataFrame(pred, columns=['energy'])
        df_input = pd.DataFrame(input, columns=['energy'])
        df_inv_true = tsp.inverse_scale_targets(df_true)
        df_inv_pred = tsp.inverse_scale_targets(df_pred)
        df_inv_input = tsp.inverse_scale_targets(df_input)
        input_list.append(np.array(df_inv_input))
        true_list.append(np.array(df_inv_true))
        pred_list.append(np.array(df_inv_pred))

    # 2. show overall evaluation results

    true = np.concatenate(true_list, axis=0)
    pred = np.concatenate(pred_list, axis=0)


    cv_rmse, mae, nrmse = cal_cvrmse(pred, true), cal_mae(pred, true), cal_nrmse(pred, true)

    return input_list, true_list, pred_list, cv_rmse, mae, nrmse


# save some sample time series plots
def sample_plots(args, input_list, true_list, pred_list, dataset_name, path):

    n_display = 5

    for i in range(min(len(input_list), n_display)):
        index = random.randint(0, len(input_list)-1)
        true = true_list[index]
        pred = pred_list[index]
        input = input_list[index]

        # print(true.shape, pred.shape, input.shape)

        # visualize
        plt.figure(figsize=(16, 8))
        plt.plot(range(input.shape[0]), input, color='black', label='input')
        plt.plot(range(input.shape[0], input.shape[0] + true.shape[0]), true, color="grey", label="true")
        plt.plot(range(input.shape[0], input.shape[0] + pred.shape[0]), pred, color="blue", label="pred")
        plt.axvline(x=args['context_length'], color='black', linestyle='--', linewidth=3)
        plt.legend()
        plt.title('{} Dataset, NRMSE: {:.4f}'.format(dataset_name, cal_nrmse(pred, true)))
        plt.savefig('{}/{}.png'.format(path, i))
        # plt.show()





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Testing and Evaluation")
    parser.add_argument('--config-file', type=str, default='./config/tinyTimeMixer.json', help='Path to config file', required=True)
    file_path_arg = parser.parse_args()
    config_file = file_path_arg.config_file
    # config_file = './config/tinyTimeMixer-16-3-3.json'
    with open(config_file, 'r') as f:
        args = json.load(f)

    model_name = config_file.split('/')[-1][:-5]
    # load pretrained model
    model = TinyTimeMixerForPrediction.from_pretrained(args["model_save_dir"])

    parms = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("trainable model's parameter count is:",parms)

    testing_arguments  = TrainingArguments(
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

    trainer = Trainer(
        model=model,
        args=testing_arguments,
        callbacks=[early_stopping_callback],
    )


    # load test dataset
    dataset_path = args["dataset_path"] + '/' + 'test'
    median_res = []
    for dataset in os.listdir(dataset_path):
        building_path = dataset_path + '/' + dataset
        inputs = []
        trues = []
        preds = []

        res = []

        os.makedirs("./results/{}/{}".format(model_name, dataset), exist_ok=True)
        result_path = "./results/{}/{}".format(model_name, dataset)

        for building in os.listdir(building_path):
            test_dataset, tsp = dataset_split(args, building_path, building)
            building_id = building.rsplit(".csv",1)[0]
            input_list, true_list, pred_list, cv_rmse, mae, nrmse = eval_model(args, trainer, test_dataset, tsp)

            inputs.extend(input_list)
            trues.extend(true_list)
            preds.extend(pred_list)

            res.append([cv_rmse, mae, nrmse, building_id])

        columns = ["cv_rmse", "mae", "nrmse", "building_id"]
        df = pd.DataFrame(res, columns=columns)
        df.to_csv("{}/{}.csv".format(result_path, 'result'), index=False)


        sample_plots(args, inputs, trues, preds, dataset, result_path)


        med_nrmse = df["nrmse"].median()
        med_cvrmse = df["cv_rmse"].median()
        med_mae = df["mae"].median()

        median_res.append([dataset, med_nrmse, med_cvrmse, med_mae])

    med_columns = ['dataset','nrmse', 'cv_rmse', 'mae']
    median_df = pd.DataFrame(median_res, columns=med_columns)
    median_df.to_csv("./results/{}/median_buildings_results.csv".format(model_name), index=False)










































# def test_dataset_split(args,building_path, building):

#     testDataset = []

#     df = pd.read_csv(building_path + '/' + building, parse_dates=[args["timestamp_column"]])
    
#     n_samples = df.shape[0]

#     train_sample = int(0.7*n_samples)
#     eval_sample = int(0.8*n_samples)

#     train_data = df.iloc[:train_sample]
#     test_data = df.iloc[eval_sample:]

#     tsp = TimeSeriesPreprocessor(
#         context_length=args["context_length"],
#         timestamp_column=args["timestamp_column"],
#         id_columns=args["id_columns"],
#         target_columns=args["forecast_columns"],
#         scaling=args["is_scaling"]
#     )
#     tsp.train(train_data)


#     test_dataset = ForecastDFDataset(
#         tsp.preprocess(test_data),
#         timestamp_column=args["timestamp_column"],
#         id_columns=args["id_columns"],
#         target_columns=args["forecast_columns"],
#         context_length=args["context_length"],
#         prediction_length=args["prediction_length"],
#         stride=args["patch_stride"]
#     )
    
#     testDataset.append(test_dataset)


#     combinedTestDataset = ConcatDataset(testDataset)
#     return combinedTestDataset, tsp
    


    







