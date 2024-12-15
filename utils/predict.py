
from Encoder_Decoder import get_predict_EncoderDecoder_6,get_predict_EncoderDecoder_7
from T5_ver4 import get_predict_T5_ver4
from BiLSTM_2 import get_predict_BiLSTM_2
from GRU import get_predict_GRU_with_attention_ver7
from MarianMT import get_predict_MarianMT_ver4
def predict(model_name, model_path, sentence):
    test_data = [sentence]
    if model_name == "EncoderDecoder_7" :
            texts, predictions, references = get_predict_EncoderDecoder_7(model_path, test_data)
    if model_name == "EncoderDecoder_6":
            texts, predictions, references = get_predict_EncoderDecoder_6(model_path, test_data)
    if model_name == "MarianMT_ver4":
            texts, predictions, references = get_predict_MarianMT_ver4(model_path, test_data) 
    if model_name == "GRU_with_attention_ver7":
            texts, predictions, references = get_predict_GRU_with_attention_ver7(model_path, test_data)
    if model_name == "T5_ver4":
            texts, predictions, references = get_predict_T5_ver4(model_path, test_data)
    if model_name == "BiLSTM_2":
        texts, predictions, references = get_predict_BiLSTM_2(model_path, test_data)

    return predictions