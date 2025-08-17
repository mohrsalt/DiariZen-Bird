import os 
import argparse

from diarizen.models.pruning.utils import convert_wavlm

def run():
    hf_dir = os.path.abspath("/home/users/ntu/bhargavi/scratch/wavlm_base_loaded")
    out_dir = os.path.abspath("/home/users/ntu/bhargavi/scratch/wavlmlocal")

    if not os.path.isdir(hf_dir):
        raise FileNotFoundError(f"HuggingFace directory does not exist: {hf_dir}")
    
    os.makedirs(out_dir, exist_ok=True)
    convert_wavlm(hf_dir, out_dir)

if __name__ == "__main__":
#    parser = argparse.ArgumentParser(
#        description="Convert HuggingFace WavLM model to custom format "
#                    "(e.g. from /pre-trained/HF/wavlm-base-plus)"
#    )
#    parser.add_argument(
#        "hf_dir", type=str, 
#        help="Path to the HuggingFace WavLM directory containing config.json and pytorch_model.bin."
#    )
#    parser.add_argument(
#        "out_dir", type=str, 
#        help="Path to output directory where the converted model will be saved."
#    )
    
#    args = parser.parse_args()
    run()