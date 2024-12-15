from huggingface_hub import snapshot_download
import os
from huggingface_hub import login

# Đặt token API từ Hugging Face
login(token="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")
def download_model(model_name: str, cache_dir: str = "./") -> str:
    """
    Downloads a model from Hugging Face Hub and returns the model path.

    Args:
    model_name (str): The name of the model on Hugging Face.
    cache_dir (str): The directory where the model will be saved. Default is "/kaggle/working".

    Returns:
    str: The path to the downloaded model.
    """
    # Define repo_id (replace 'model' with repo type if necessary)
    repo_id = model_name
    repo_type = "model"  # Can be adjusted if it's a dataset or other repo type

    # Download the model snapshot
    model_path = snapshot_download(
        repo_id="Sag1012/machine-translation",
        repo_type=repo_type,
        cache_dir=cache_dir,
        allow_patterns=[model_name+"/**"]  # Adjust pattern if needed
    )
    model_path = model_path + "/"+ model_name
    if model_name == "EncoderDecoder_7" or model_name == "EncoderDecoder_6":
        model_path = model_path
    if model_name == "MarianMT_ver4":
        model_path = model_path 
    if model_name == "GRU_with_attention_ver7":
        model_path = model_path + "/decoder.pth"
    if model_name == "T5_ver4":
        model_path = model_path + "/rng_state.pth"
    if model_name == "BiLSTM_2":
        model_path = model_path + "/my_model_1.keras"
    

    print(f"Model '{model_name}' downloaded to {model_path}")
    return model_path



