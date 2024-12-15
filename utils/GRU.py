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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
MAX_LENGTH = 50
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size).to(device)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True).to(device)
        self.hidden_transform = nn.Linear(hidden_size * 2, hidden_size).to(device)

    def forward(self, input):
        input = input.to(device)
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded)
        output, hidden = output.to(device), hidden.to(device)
        output = self.hidden_transform(output)

        return output, hidden

class CrossAttention(nn.Module):
    def __init__(self, hidden_size):
        super(CrossAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size).to(device)  # Linear layer for the query
        self.Ua = nn.Linear(hidden_size, hidden_size).to(device)    # Linear layer for the keys
        self.Va = nn.Linear(hidden_size, hidden_size).to(device)  # Linear layer for the values
        self.softmax = nn.Softmax(dim=-1).to(device)  

    def forward(self, query, keys):

        query_proj = self.Wa(query).to(device)  # Shape: (batch_size, query_len, hidden_size)
        key_proj = self.Ua(keys).to(device)      # Shape: (batch_size, key_len, hidden_size)
        value_proj = self.Va(keys).to(device)  # Shape: (batch_size, key_len, hidden_size)

        scores = torch.bmm(query_proj, key_proj.transpose(1, 2)).to(device)  # Shape: (batch_size, query_len, key_len)
        scores = scores / torch.sqrt(torch.tensor(key_proj.size(-1), dtype=torch.float32, device=device))  # Scale by sqrt(hidden_size)
        

        # Compute attention weights
        attention_weights = self.softmax(scores).to(device)  # Shape: (batch_size, query_len, key_len)

        # Compute context vectors as weighted sum of values
        context = torch.bmm(attention_weights, value_proj).to(device)  # Shape: (batch_size, query_len, hidden_size)

        return context, attention_weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size).to(device)
        self.attention = CrossAttention(hidden_size).to(device)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout_p), batch_first=True).to(device)
        self.out = nn.Linear(hidden_size, output_size).to(device)
        self.dropout = nn.Dropout(dropout_p).to(device)
        self.hidden_transform = nn.Linear(hidden_size * 2, hidden_size).to(device)
        self.hidden_input_transform = nn.Linear(hidden_size * 2, hidden_size).to(device)
        self.hidden_size = hidden_size

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        encoder_outputs = encoder_outputs.to(device)
        encoder_hidden = encoder_hidden.to(device)
        if target_tensor is not None:
            target_tensor = target_tensor.to(device)

        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token).to(device)
        decoder_hidden = self.transform_bidirectional_hidden(encoder_hidden)
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1).to(device)  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach().to(device)  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1).to(device)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1).to(device)
        attentions = torch.cat(attentions, dim=1).to(device)

        return decoder_outputs, decoder_hidden, attentions

    def transform_bidirectional_hidden(self, encoder_hidden):
        forward_states = encoder_hidden[0::2, :, :].to(device)  # Forward states: (batch, num_layers, hidden_size)
        backward_states = encoder_hidden[1::2, :, :].to(device)  # Backward states: (batch, num_layers, hidden_size)
        combined_hidden = torch.cat((forward_states, backward_states), dim=2).to(device)  # Shape: (batch, num_layers, hidden_size * 2)
        combined_hidden = self.hidden_transform(combined_hidden).to(device)
        return combined_hidden

    def forward_step(self, input, hidden, encoder_outputs):
        encoder_outputs = encoder_outputs.to(device)

        embedded = self.embedding(input).to(device)
        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)
        input_gru = self.hidden_input_transform(input_gru)
        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights
class Translator(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Translator, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, input_tensor, target_tensor=None):
        if target_tensor is not None:
            target_tensor = target_tensor.to(self.device)
        encoder_outputs, encoder_hidden = self.encoder(input_tensor)
        decoder_outputs, _, _ = self.decoder(encoder_outputs, encoder_hidden, target_tensor)
        return decoder_outputs

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()

def load_model_GRU(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()

def clean_decoded_sentence(sentence):
    special_tokens = ["<s>", "</s>", "<pad>", "<unk>"]
    for token in special_tokens:
        sentence = sentence.replace(token, "").strip() 
    return sentence

def get_predict_GRU_with_attention_ver7(model_path,test_data):
    import os
    directory_path = os.path.dirname(model_path)
    encoder_pth = directory_path + "/encoder.pth"
    decoder_pth = directory_path + "/decoder.pth"
    
    texts = []
    predictions = []
    references = []

    VOCAB_SIZE = 64000
    hidden_size = 256
    encoder = EncoderRNN(VOCAB_SIZE, hidden_size)
    decoder = AttnDecoderRNN(hidden_size, VOCAB_SIZE)
    load_model_GRU(encoder, encoder_pth)
    load_model_GRU(decoder, decoder_pth)
    translator = Translator(encoder,decoder,device)
    english_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vietnamese_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

    for item in test_data:
        source = item["en"]
        target = item["vi"]
        
        input_sentence = source
        english_tokens = english_tokenizer.encode(input_sentence)
        english_tensor = torch.tensor(english_tokens).unsqueeze(0).to(device)
        with torch.no_grad():
            output_tensor = translator(english_tensor)
        predicted_token_ids = torch.argmax(output_tensor, dim=-1).squeeze(0).tolist()
        vietnamese_sentence = vietnamese_tokenizer.decode(predicted_token_ids)
        vietnamese_sentence_cleaned = clean_decoded_sentence(vietnamese_sentence)
                    
        texts.append(source)
        predictions.append(vietnamese_sentence_cleaned)
        references.append(target)
    return texts, predictions, references