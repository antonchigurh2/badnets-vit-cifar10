import argparse
import os
import pathlib
import re
import time
import datetime

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import build_poisoned_training_set, build_testset, build_ood_testset
from deeplearning import evaluate_badnets, optimizer_picker, train_one_epoch, evaluate_badnets_ood
from models import BadNet

import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
    # import torch
import pickle
parser = argparse.ArgumentParser(description='Reproduce the basic backdoor attack in "Badnets: Identifying vulnerabilities in the machine learning model supply chain".')
parser.add_argument('--dataset', default='CIFAR10', help='Which dataset to use (MNIST or CIFAR10, default: CIFAR10)')
parser.add_argument('--model', default='vit', help='resnet18, vit, simple_conv')
parser.add_argument('--nb_classes', default=10, type=int, help='number of the classification types')
parser.add_argument('--load_local', action='store_true', help='train model or directly load model (default true, if you add this param, then load trained local model to evaluate the performance)')
parser.add_argument('--loss', default='mse', help='Which loss function to use (mse or cross, default: mse)')
parser.add_argument('--optimizer', default='sgd', help='Which optimizer to use (sgd or adam, default: sgd)')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train backdoor model, default: 100')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size to split dataset, default: 64')
parser.add_argument('--num_workers', type=int, default=64, help='Batch size to split dataset, default: 64')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate of the model, default: 0.001')
parser.add_argument('--download', action='store_true', help='Do you want to download data ( default false, if you add this param, then download)')
parser.add_argument('--data_path', default='./data/', help='Place to load dataset (default: ./dataset/)')
parser.add_argument('--device', default='cpu', help='device to use for training / testing (cpu, or cuda:1, default: cpu)')
# poison settings
parser.add_argument('--poisoning_rate', type=float, default=0.1, help='poisoning portion (float, range from 0 to 1, default: 0.1)')
parser.add_argument('--trigger_label', type=int, default=2, help='The NO. of trigger label (int, range from 0 to 10, default: 0)')
parser.add_argument('--trigger_path', default="./triggers/6.png", help='Trigger Path (default: ./triggers/trigger_white.png)')
parser.add_argument('--trigger_size', type=int, default=10, help='Trigger Size (int, default: 5)')
parser.add_argument('--print_step', type=int, default=2, help='')
parser.add_argument('--clean_label', action='store_true', help='')
parser.add_argument('--class_distinct_trigger', action='store_true', help='')
parser.add_argument('--image_width', type=int, default=224, help='')
parser.add_argument('--image_height', type=int, default=224, help='')


args = parser.parse_args()


# Assuming dataset_train is your PyTorch dataset
# Replace 'output_directory' with the desired path where you want to save the dataset

def main():
    print("TESTING IF MAIN IS CHANGED OR NOT!!!")
    print("{}".format(args).replace(', ', ',\n'))

    if re.match('cuda:\d', args.device):
        cuda_num = args.device.split(':')[1]
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_num
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # if you're using MBP M1, you can also use "mps"

    # create related path
    pathlib.Path("./checkpoints/").mkdir(parents=True, exist_ok=True)
    pathlib.Path("./logs/").mkdir(parents=True, exist_ok=True)

    print("\n# load dataset: %s " % args.dataset)
    dataset_train, args.nb_classes = build_poisoned_training_set(is_train=True, args=args)
    dataset_val_clean, dataset_val_poisoned = build_testset(is_train=False, args=args)
    dataset_val_clean_ood, dataset_val_poisoned_ood = build_ood_testset(is_train=False, args=args)

    output_file = 'dataset_train.pth'
    torch.save(dataset_train, output_file)

    print(f'Dataset has been saved to {output_file}')
    data_loader_val_clean_ood    = DataLoader(dataset_val_clean_ood,     batch_size=args.batch_size, shuffle=True, num_workers=2)
    data_loader_val_poisoned_ood = DataLoader(dataset_val_poisoned_ood,  batch_size=args.batch_size,  shuffle=True, num_workers=2)
    output_file='dataset_val_clean.pth'
    torch.save(dataset_val_clean,output_file)
    print(f'Dataset-Val-Clean has been saved to {output_file}')
    output_file='dataset_val_poisoned.pth'
    torch.save(dataset_val_poisoned,output_file)
    print(f'Dataset-Val-poisoned has been saved to {output_file}')
if __name__ == "__main__":
    main()