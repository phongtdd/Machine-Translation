# Machine-Translation
Mini-project for course of Intro to Deep Learning, with the aim of developing a Machine Translator to translates from English to Vietnamese. 

# Table of Contents

- [machine-translation-en-vi](#machine-translation-en-vi)
- [Table of Contents](#table-of-contents)
- [Dataset](#dataset)
- [How to use our code](#how-to-use-our-code)
  - [Installation](#installation)
  - [Training](#training)
  - [Download the models and User Interface](#download-the-models-and-user-interface)
  - [Infer file](#infer-file)
- [Trained models](#trained-models)
- [References](#references)

# Dataset
PhoMT is a high-quality and large-scale Vietnamese-English parallel dataset of 3.02M sentence pairs for machine-translation. 
The dataset construction process involves collecting parallel document pairs, preprocessing for cleaning and quality, aligning parallel 
sentences, and postprocessing to filter out duplicates and verify set quality

# How to use our code
## Installation
All of our experiments have been conducted in the Kaggle environment. We highly recommend using Kaggle to run the code for the best compatibility and to avoid potential setup issues.

If you prefer to run the code on your local machine, you can install the required packages by using the following command:
```bash
pip install -r requirements.txt
```

## Download the models and User Interface

# Trained models
The trained models are available on the Hugging Face model hub: [Hugging Face model hub](https://huggingface.co/Sag1012/machine-translation)

You can see so many models here. However, if you want to try the best model of our experiments, you can try the model in the following folder:
- MarianMT: MarianMT_ver4
- BertBartpho finetuning: EncoderDecoder_6, EncoderDecoder_7
- T5: T5_ver4
- GRU with attention: GRU_with_attention_ver7
- LSTM with attention: LSTM_Attention_2
- BiLSTM without attention: BiLSTM_2


