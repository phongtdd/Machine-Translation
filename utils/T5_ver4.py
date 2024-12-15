from transformers import AutoTokenizer, EncoderDecoderModel, T5Tokenizer, T5ForConditionalGeneration
import sentencepiece as spm
def load_model_T5(model_path):
    encoder = T5Tokenizer.from_pretrained("t5-small")
    decoder = spm.SentencePieceProcessor(model_file = '/kaggle/working/models--Sag1012--machine-translation/snapshots/164bfec8e7d09d77ab222a6055293e66934994ca/T5/vi_tokenizer_32128.model')
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    return encoder, decoder, model

def get_predict_T5_ver4(model_path, test_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder, model = load_model_T5(model_path)
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
        outputs = model.generate(inputs["input_ids"], max_length=64, num_beams=4)
        prediction = decoder.decode(outputs[0].tolist())
            
        texts.append(source)
        predictions.append(prediction)
        references.append(target)
        
    
    return texts, predictions, references    
