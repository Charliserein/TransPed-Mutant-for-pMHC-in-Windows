# coding: utf-8

from model import *
from attention import peptide_peptide_attns_draw_save
from mutation import *
from utils import Logger, cut_peptide_to_specific_length

import math
from sklearn import metrics
from sklearn import preprocessing
import numpy as np
import pandas as pd
import re
import time
import datetime
import random
random.seed(1234)
from scipy import interp
import warnings
warnings.filterwarnings("ignore")

from collections import Counter
from collections import OrderedDict
from functools import reduce
from tqdm import tqdm, trange
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

import matplotlib as mpl
mpl.use('AGG')
import matplotlib.pyplot as plt
import seaborn as sns

import difflib

seed = 19961231
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import os
import argparse
import logging
import sys

def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

parser = argparse.ArgumentParser(usage = 'peptide-peptide binding prediction')
parser.add_argument('--peptide_file_1', type = str, help = 'the path of the first .fasta file contains peptides')
parser.add_argument('--peptide_file_2', type = str, help = 'the path of the second .fasta file contains peptides')
parser.add_argument('--threshold', type = float, default = 0.5, help = 'the threshold to define predicted binder, float from 0 - 1, the recommended value is 0.5')
parser.add_argument('--cut_peptide', type = bool, default = True, help = 'Whether to split peptides larger than cut_length?')
parser.add_argument('--cut_length', type = int, default = 9, help = 'if there is a peptide sequence length > 15, we will segment the peptide according to the length you choose, from 8 - 15')
parser.add_argument('--output_dir', type = str, help = 'The directory where the output results are stored.')
parser.add_argument('--output_attention', type = bool, default = True, help = 'Output the mutual influence of peptides on the binding?')
parser.add_argument('--output_heatmap', type = bool, default = True, help = 'Visualize the mutual influence of peptides on the binding?')
parser.add_argument('--output_mutation', type = bool, default = True, help = 'Whether to perform mutations with better affinity for each sample?')
args = parser.parse_args()
print(args)

errLogPath = args.output_dir + '/error.log'
if args.threshold <= 0 or args.threshold >= 1:
    log = Logger(errLogPath)
    log.logger.critical('The threshold is invalid, please check whether it ranges from 0-1.')
    sys.exit(0)
if not args.peptide_file_1:
    log = Logger(errLogPath)
    log.logger.critical('The first peptide file is empty.')
    sys.exit(0)
if not args.peptide_file_2:
    log = Logger(errLogPath)
    log.logger.critical('The second peptide file is empty.')
    sys.exit(0)
if not args.output_dir:
    log = Logger(errLogPath)
    log.logger.critical('Please fill the output file directory.')
    sys.exit(0)

cut_length = args.cut_length
if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)

# 读取文件

with open(args.peptide_file_1, 'r') as f:
    peptide_file_1 = f.readlines()

with open(args.peptide_file_2, 'r') as f:
    peptide_file_2 = f.readlines()

if len(peptide_file_1) != len(peptide_file_2):
    log = Logger(errLogPath)
    log.logger.critical('Please ensure the same number of peptides in both files.')
    sys.exit(0)

i = 0
ori_peptides_1, ori_peptides_2 = [], []
for pep1, pep2 in zip(peptide_file_1, peptide_file_2):
    if i % 2 == 0:
        pep1 = str.upper(pep1.replace('\n', '').replace('\t', ''))
        ori_peptides_1.append(pep1)
    if i % 2 == 1:
        pep2 = str.upper(pep2.replace('\n', '').replace('\t', ''))
        ori_peptides_2.append(pep2)
    i += 1

peptides_1, peptides_2 = [], []
for pep1, pep2 in zip(ori_peptides_1, ori_peptides_2):
    if not (pep1.isalpha() and pep2.isalpha()):
        continue
    if len(set(pep1).difference(set('ARNDCQEGHILKMFPSTWYV'))) != 0:
        continue
    if len(set(pep2).difference(set('ARNDCQEGHILKMFPSTWYV'))) != 0:
        continue
    length1 = len(pep1)
    length2 = len(pep2)
    if length1 < 15 and length2 < 15:
        if args.cut_peptide:
            if length1 > cut_length or length2 > cut_length:
                cut_peptides_1 = [pep1] + [pep1[i : i + cut_length] for i in range(length1 - cut_length + 1)]
                cut_peptides_2 = [pep2] + [pep2[i : i + cut_length] for i in range(length2 - cut_length + 1)]
                peptides_1.extend(cut_peptides_1)
                peptides_2.extend(cut_peptides_2)
            else:
                peptides_1.append(pep1)
                peptides_2.append(pep2)
        else:
            peptides_1.append(pep1)
            peptides_2.append(pep2)
    else:
        cut_peptides_1 = [pep1[i : i + cut_length] for i in range(length1 - cut_length + 1)]
        cut_peptides_2 = [pep2[i : i + cut_length] for i in range(length2 - cut_length + 1)]
        peptides_1.extend(cut_peptides_1)
        peptides_2.extend(cut_peptides_2)

predict_data = pd.DataFrame([peptides_1, peptides_2], index = ['peptide_1', 'peptide_2']).T
if predict_data.shape[0] == 0:
    log = Logger(errLogPath)
    log.logger.critical('No suitable data could be predicted. Please check your input data.')
    sys.exit(0)

predict_data, predict_pep_inputs_1, predict_pep_inputs_2, predict_loader = read_predict_data(predict_data, batch_size)

# 预测

use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")

model_file = 'peptide_peptide_model.pkl'

model_eval = Transformer().to(device)
model_eval.load_state_dict(torch.load(model_file, map_location='cpu'), strict = True)

model_eval.eval()
y_pred, y_prob, attns = eval_step(model_eval, predict_loader, args.threshold, use_cuda)

predict_data['y_pred'], predict_data['y_prob'] = y_pred, y_prob
predict_data = predict_data.round({'y_prob': 4})

predict_data.to_csv(args.output_dir + '/predict_results.csv', index = False)

# 作图

if args.output_attention or args.output_heatmap:
    if args.output_attention:
        attn_savepath = args.output_dir + '/attention/'
        if not os.path.exists(attn_savepath):
            os.makedirs(attn_savepath)
    else:
        attn_savepath = False
    if args.output_heatmap:
        fig_savepath = args.output_dir + '/figures/'
        if not os.path.exists(fig_savepath):
            os.makedirs(fig_savepath)
    else:
        fig_savepath = False

    for pep1, pep2 in zip(predict_data.peptide_1, predict_data.peptide_2):
        peptide_peptide_attns_draw_save(predict_data, attns, pep1, pep2, attn_savepath, fig_savepath)

# 突变

if args.output_mutation:
    mut_savepath = args.output_dir + '/mutation/'
    if not os.path.exists(mut_savepath):
        os.makedirs(mut_savepath)

    for idx in range(predict_data.shape[0]):
        peptide_1 = predict_data.iloc[idx].peptide_1
        peptide_2 = predict_data.iloc[idx].peptide_2

        if len(peptide_1) < 8 or len(peptide_1) > 14 or len(peptide_2) < 8 or len(peptide_2) > 14: continue

        mut_peptides_df = peptide_peptide_mutation(predict_data, attns, peptide_1 = peptide_1, peptide_2 = peptide_2)
        mut_data, _, _, mut_loader = read_predict_data(mut_peptides_df, batch_size)

        model_eval = Transformer().to(device)
        model_eval.load_state_dict(torch.load(model_file, map_location='cpu'), strict = True)

        model_eval.eval()
        y_pred, y_prob, attns = eval_step(model_eval, mut_loader, args.threshold, use_cuda)

        mut_data['y_pred'], mut_data['y_prob'] = y_pred, y_prob
        mut_data = mut_data.round({'y_prob': 4})

        if mut_savepath:
            sanitized_peptide_1 = sanitize_filename(peptide_1)
            sanitized_peptide_2 = sanitize_filename(peptide_2)
            file_path = os.path.join(mut_savepath, '{}_{}_mutation.csv'.format(sanitized_peptide_1, sanitized_peptide_2))
            mut_data.to_csv(file_path, index=False)
            print('********** {} | {} → # Mutation peptides = {}'.format(peptide_1, peptide_2, mut_data.shape[0] - 1))

        mut_peptides_IEDBfmt = ' '.join(mut_data.mutation_peptide)
        print('If you want to use IEDB tools to predict IC50, please use these format: \n {}'.format(mut_peptides_IEDBfmt))

# python peptide_peptide_predict.py --peptide_file_1 "peptides1.fasta" --peptide_file_2 "peptides2.fasta" --threshold 0.5 --cut_length 10 --cut_peptide True --output_dir './results/' --output_attention True --output_heatmap True --output_mutation True
