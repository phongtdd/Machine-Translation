from transformers import AutoTokenizer, EncoderDecoderModel, AutoModel
import torch
def load_model_Bert_BARTPho(model_path):
    encoder_model_name = "bert-base-uncased"  
    decoder_model_name = "vinai/bartpho-word"  
    
    encoder = AutoTokenizer.from_pretrained(encoder_model_name)
    decoder = AutoTokenizer.from_pretrained(decoder_model_name)
    model = EncoderDecoderModel.from_pretrained(model_path)
    return encoder, decoder, model

def get_predict_EncoderDecoder_6(model_path, test_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder, model = load_model_Bert_BARTPho(model_path)
    model = model.to(device)
    
    texts = []
    predictions = []
    references = []
    
    for item in test_data:
        source = item["en"]
        target = item["vi"]
        
        inputs = encoder(source, 
                        padding=True, 
                        truncation=True, 
                        max_length=len(target.split()),
                        return_tensors="pt")
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
            # Generate translation
        outputs = model.generate(inputs["input_ids"], max_length=64, num_beams=4)
        prediction = decoder.decode(outputs[0], skip_special_tokens=True)

        texts.append(source)
        predictions.append(prediction)
        references.append(target)
        
    
    return texts, predictions, references 
def get_predict_EncoderDecoder_7(model_path, test_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder, model = load_model_Bert_BARTPho(model_path)
    model = model.to(device)
    
    texts = []
    predictions = []
    references = []
    
    for item in test_data:
        source = item["en"]
        target = item["vi"]
        
        inputs = encoder(source, 
                        padding=True, 
                        truncation=True, 
                        max_length=len(target.split()),
                        return_tensors="pt")
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
            # Generate translation
        outputs = model.generate(inputs["input_ids"], max_length=64, num_beams=4)
        prediction = decoder.decode(outputs[0], skip_special_tokens=True)

        texts.append(source)
        predictions.append(prediction)
        references.append(target)
        
    
    return texts, predictions, references    