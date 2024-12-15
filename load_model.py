import sys
import os

import torch
from transformers import AutoTokenizer, EncoderDecoderModel, T5Tokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM, MarianMTModel, MarianTokenizer
import pickle
import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
from GRU_with_attention_ver4.load_GRU_model import translate_GRU

import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_BERT_BARTPho():
    encoder_model_name = "bert-base-uncased"  
    decoder_model_name = "vinai/bartpho-word"  
    model_path = "./EncoderDecoder_6"
    
    encoder = AutoTokenizer.from_pretrained(encoder_model_name)
    decoder = AutoTokenizer.from_pretrained(decoder_model_name)
    model = EncoderDecoderModel.from_pretrained(model_path).to(device)
    return encoder, decoder, model

def translate_BERT_BARTPho(input_text, encoder=None, decoder=None, model=None):
    if encoder is None or decoder is None or model is None:
        encoder, decoder, model = load_model_BERT_BARTPho()
    
    inputs = encoder(
        input_text,
        return_tensors="pt", 
        padding=True, 
        truncation=True
    ).to(device)

    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    outputs = model.generate(inputs["input_ids"], max_length=64, num_beams=4)
    
    return decoder.decode(outputs[0], skip_special_tokens=True)

def load_model_T5():
    model_folder = "./T5_ver3"
    decoder_path = model_folder + "/vi_tokenizer_32128.model"
    
    encoder = T5Tokenizer.from_pretrained("t5-small", skip_special_tokens=True)
    decoder = T5Tokenizer.from_pretrained(pretrained_model_name_or_path = decoder_path, skip_special_tokens=True)
    model = T5ForConditionalGeneration.from_pretrained(model_folder, max_length = 64).to(device)
    return encoder, decoder, model

def translate_T5(input_text, encoder=None, decoder=None, model=None):
    if encoder is None or decoder is None or model is None:
        encoder, decoder, model = load_model_T5()
    
    # Tiến hành dịch
    inputs = encoder(input_text, return_tensors="pt").to(device)
    outputs = model.generate(inputs['input_ids'])
    output_text = decoder.decode(outputs[0].tolist(), skip_special_tokens=True) 
    
    return output_text
    
def load_model_BiLSTM():
    model_folder = f"./BiLSTM_2"
    encoder_path = model_folder + "/english_tokenizer.pkl"
    decoder_path = model_folder + "/vietnamese_tokenizer.pkl"
    model_path = model_folder + "/my_model_1.keras"
    
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)
    with open(decoder_path, "rb") as f:
        decoder = pickle.load(f)
    
    model = load_model(model_path)
    return encoder, decoder, model
    
def translate_BiLSTM(input_text, encoder=None, decoder=None, model=None):                                                                                           
    if encoder is None or decoder is None or model is None:
        encoder, decoder, model = load_model_BiLSTM()

    # Extract components from the model
    encoder_input = model.input[0]  # Input tensor for the encoder
    encoder_output = model.get_layer("bidirectional").output[0]
    encoder_state_h = model.get_layer("state_h_concat").output
    encoder_state_c = model.get_layer("state_c_concat").output

    # Build encoder model
    encoder_model = Model(encoder_input, [encoder_output, encoder_state_h, encoder_state_c])

    # Extract decoder components
    decoder_embedding = model.get_layer("decoder_embedding")
    decoder_lstm = model.get_layer("decoder_lstm")
    decoder_dense = model.get_layer("decoder_dense")

    # Define decoder inference inputs
    units = 128  # LSTM units
    decoder_state_input_h = Input(shape=(units * 2,), name="decoder_state_input_h")
    decoder_state_input_c = Input(shape=(units * 2,), name="decoder_state_input_c")
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    # Reuse the embedding and LSTM layers
    decoder_input = Input(shape=(1,), name="decoder_input")  # Decoder input for one time step
    decoder_embedding_inf = decoder_embedding(decoder_input)
    decoder_lstm_inf = decoder_lstm(decoder_embedding_inf, initial_state=decoder_states_inputs)
    decoder_output_inf, state_h_inf, state_c_inf = decoder_lstm_inf

    decoder_states_inf = [state_h_inf, state_c_inf]

    # Dense layer for probabilities
    decoder_output_inf = decoder_dense(decoder_output_inf)

    # Build decoder inference model
    decoder_model = Model(
        [decoder_input] + decoder_states_inputs,  # Inputs
        [decoder_output_inf] + decoder_states_inf  # Outputs
    )

    # Helper functions
    def preprocess_sentence(sentence, tokenizer, max_length):
        """Preprocess and tokenize an input sentence."""
        sequence = tokenizer.texts_to_sequences([sentence])
        return pad_sequences(sequence, maxlen=max_length, padding='post')

    def decode_sequence(input_seq):
        """Generate a Vietnamese sentence from an English input sequence."""
        # Encode the input sequence to get initial states
        encoder_output, state_h, state_c = encoder_model.predict(input_seq)

        # Initialize the decoder input with the <start> token
        target_seq = np.zeros((1, 1))  # Shape: (batch_size, 1)
        target_seq[0, 0] = decoder.texts_to_sequences(["<SOS>"])[0][0]

        # Initialize states
        states = [state_h, state_c]

        # Generate the output sequence token by token
        decoded_sentence = []
        for _ in range(232):
            output_tokens, h, c = decoder_model.predict([target_seq] + states)

            # Sample the next token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_token = decoder.index_word.get(sampled_token_index, '<unk>')
            if sampled_token == '<eos>':
                break

            decoded_sentence.append(sampled_token)

            # Update the target sequence (input to the decoder)
            target_seq[0, 0] = sampled_token_index

            # Update states
            states = [h, c]

        return ' '.join(decoded_sentence)
    max_input_length = 193  # Adjust based on your tokenizer setup
    input_sequence = preprocess_sentence(input_text, encoder, max_input_length)

    # Generate translation
    translation = decode_sequence(input_sequence)

    return translation

# def load_model_GRU():
#     model_folder = f"./GRU_with_attention_ver3"
#     return encoder, decoder, model
    
# def translate_GRU(input_text, encoder=None, decoder                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               =None, model=None):
#     translation = translate_GRU(input_text)
#     return translation

def load_model_LSTM():
    encoder_model_name = "bert-base-uncased"  
    decoder_model_name = "vinai/phobert-base"  
    model_path = r'LSTM_Attention_2\best_model.keras'
    
    encoder = AutoTokenizer.from_pretrained(encoder_model_name)
    decoder = AutoTokenizer.from_pretrained(decoder_model_name)
    model = load_model(model_path)
    return encoder, decoder, model

def translate_LSTM(input_text, encoder=None, decoder=None, model=None):
    max_length = 50
    
    if encoder is None or decoder is None or model is None:
        encoder, decoder, model = load_model_LSTM()
    
    def greedy_decode(input_sequence, model, decoder, max_length=50):

        input_sequence = tf.constant([input_sequence], dtype=tf.int64)

        # Start with the target sequence containing only the start token
        start_token = decoder.cls_token_id
        end_token = decoder.sep_token_id

        target_sequence = [start_token]

        for _ in range(max_length):
            # Prepare input for the decoder
            decoder_input = tf.constant([target_sequence], dtype=tf.int64)

            # Predict next token probabilities
            predictions = model.predict([input_sequence, decoder_input], verbose=0)

            # Take the last time-step and find the highest probability token
            next_token = tf.argmax(predictions[:, -1, :], axis=-1).numpy()[0]

            # Append the predicted token to the target sequence
            target_sequence.append(next_token)

            # Stop if the end token is predicted
            if next_token == end_token:
                break

        # Decode the target sequence to text
        translated_sentence = decoder.decode(target_sequence[1:], skip_special_tokens=True)
        return translated_sentence
    
    input_tokens = encoder.encode(input_text, add_special_tokens=True)
    translated_text = greedy_decode(input_tokens, model, decoder)
    return translated_text

def load_model_MarianMT():
    tokenizer_model_name = "Helsinki-NLP/opus-mt-en-vi"  
    model_path = "./MarianMT_ver2"
    
    tokenizer = MarianTokenizer.from_pretrained(tokenizer_model_name)
    model = MarianMTModel.from_pretrained(model_path).to(device)
    return tokenizer, model

def translate_MarianMT(input_text, model=None, tokenizer=None):
    if model is None or tokenizer is None:
        tokenizer, model = load_model_MarianMT()

    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    outputs = model.generate(**inputs, max_length=64, num_beams=4)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return translated_text

if __name__ == "__main__":    
    input = """
    I go to school
    """

    translated_text = translate_LSTM(input)
    
    print(translated_text)
