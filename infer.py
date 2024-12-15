from huggingface_hub import snapshot_download
import os
import torch
from transformers import AutoTokenizer, EncoderDecoderModel, AutoModel
from transformers import AutoTokenizer, EncoderDecoderModel, T5Tokenizer, T5ForConditionalGeneration
import sentencepiece as spm
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import json
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoTokenizer, AutoModel, PreTrainedTokenizerFast
from transformers import MarianMTModel, MarianTokenizer
from transformers import AutoTokenizer,AutoModel
from utils.download import download_model
from utils.Encoder_Decoder import get_predict_EncoderDecoder_6,get_predict_EncoderDecoder_7
from utils.T5_ver4 import get_predict_T5_ver4
from  utils.BiLSTM_2 import get_predict_BiLSTM_2
from utils.GRU import get_predict_GRU_with_attention_ver7
from utils.predict import predict

model_name = input("Enter model name: ")  # Use input() to get user input for model name
sentence = input("Enter sentence: ")  

model_path = download_model(model_name)

predictions = predict(model_path=model_path, sentence=sentence)

print(predictions)



