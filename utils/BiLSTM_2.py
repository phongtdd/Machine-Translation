def get_predict_BiLSTM_2(model_path,test_data):
    import os
    directory_path = os.path.dirname(model_path)
    texts = []
    predictions = []
    references = []
    from tensorflow.keras.models import Model
    from tensorflow.keras.models import load_model
    model = load_model(model_path)
    encoder_input = model.input[0]  # Input tensor for the model
    encoder_output = model.get_layer("bidirectional").output[0]
    encoder_state_h = model.get_layer("state_h_concat").output
    encoder_state_c = model.get_layer("state_c_concat").output
    
    # Encoder inference model
    encoder_model = Model(encoder_input, [encoder_output, encoder_state_h, encoder_state_c])
    decoder_embedding = model.get_layer("decoder_embedding")
    decoder_lstm = model.get_layer("decoder_lstm")
    decoder_dense = model.get_layer("decoder_dense")
    from tensorflow.keras.layers import Input
    units = 128  # LSTM units
    # Decoder inference inputs
    decoder_state_input_h = Input(shape=(units * 2,), name="decoder_state_input_h")  # BiLSTM doubles the size
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
    
    # Decoder inference model
    decoder_model = Model(
        [decoder_input] + decoder_states_inputs,  # Inputs
        [decoder_output_inf] + decoder_states_inf)  # Outputs
    import numpy as np
    from tensorflow.keras.preprocessing.sequence import pad_sequences
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
        target_seq[0, 0] = vi_loaded_tokenizer.texts_to_sequences(["<SOS>"])[0][0]
    
        # Initialize states
        states = [state_h, state_c]
    
        # Generate the output sequence token by token
        decoded_sentence = []
        for _ in range(232):
            output_tokens, h, c = decoder_model.predict([target_seq] + states)
    
            # Sample the next token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_token = vi_loaded_tokenizer.index_word.get(sampled_token_index, '<unk>')
            if sampled_token == '<eos>':
                break
    
            decoded_sentence.append(sampled_token)
    
            # Update the target sequence (input to the decoder)
            target_seq[0, 0] = sampled_token_index
    
            # Update states
            states = [h, c]
    
        return ' '.join(decoded_sentence)
    import pickle
    with open(directory_path + '/english_tokenizer.pkl', 'rb') as file:
        eng_loaded_tokenizer = pickle.load(file)
    with open(directory_path +'/vietnamese_tokenizer.pkl', 'rb') as file:
        vi_loaded_tokenizer = pickle.load(file)
    

    for item in test_data:
        source = item["en"]
        target = item["vi"]
        
        input_sentence = source
        input_sequence = preprocess_sentence(input_sentence, eng_loaded_tokenizer, 193)
        
        prediction = decode_sequence(input_sequence)
            
        texts.append(source)
        predictions.append(prediction)
        references.append(target)
        
    
    return texts, predictions, references   
