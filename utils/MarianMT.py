from transformers import MarianMTModel, MarianTokenizer
import torch
def load_model_MarianMT(model_path):

    tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-vi')
    model = MarianMTModel.from_pretrained(model_path)
    return tokenizer, model

def get_predict_MarianMT_ver4(model_path, test_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_model_MarianMT(model_path)
    model = model.to(device)
    
    texts = []
    predictions = []
    references = []
    
    for item in test_data:
        source = item["en"] 
        target = item["vi"] 
        
        # Tokenize input
        inputs = tokenizer(source, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        # Generate translation
        outputs = model.generate(**inputs, max_length=64, num_beams=4)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        texts.append(source)
        predictions.append(prediction)
        references.append(target)
    
    return texts, predictions, references