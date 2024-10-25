import os
import sys
sys.path.insert(0, os.path.abspath('/home/bld56/gsoc/nemo/NeMo-opensource/'))
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf
import torch
import json

# Function to load the model from a .nemo file
def load_model(nemo_file_path):
    model = nemo_asr.models.AV_EncDecCTCModelBPE.restore_from(nemo_file_path)
    model.eval()
    return model

# Function to perform inference on a single sample
def infer_single_sample(model, sample):
    # Prepare input data
    audio_file = sample['audio_filepath']
    video_file = sample['video_filepath']
    feature_file = sample['feature_file']
    duration = sample['duration']
    
    # Perform inference
    transcription = model.transcribe(
        audio=[audio_file],
        return_hypotheses = True,
        override_duration = duration,
    )
    
    return transcription[0]

# Function to run inference on a manifest file
def run_inference(manifest_file_path, nemo_file_path, output_file_path):
    # Load the model
    model = load_model(nemo_file_path)
    
    # Read the manifest file
    with open(manifest_file_path, 'r') as f:
        manifest_data = [json.loads(line.strip()) for line in f]
    
    # Run inference on each sample in the manifest
    results = []
    for sample in manifest_data:
        transcription = infer_single_sample(model, sample)
        result = {
            'audio_filepath': sample['audio_filepath'],
            'video_filepath': sample['video_filepath'],
            'feature_file': sample['feature_file'],
            'duration': sample['duration'],
            'transcription': transcription
        }
        results.append(result)
    
    # Save the results to the output file
    with open(output_file_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    print(f"Inference completed. Results saved to {output_file_path}")

# Main function
def main():
    manifest_file_path = '/tmp/bld56_dataset_v1/it2/annotations/manifest_eval.json'  # Path to your input manifest file
    nemo_file_path = '/tmp/bld56_dataset_v1/tmp/av_ndec_lman_ntok_0.5/2024-08-16_11-16-34/checkpoints/av_ndec_lman_ntok_0.5.nemo'  # Path to your trained .nemo file
    output_file_path = 'temp.json'  # Path to save the inference results
    
    run_inference(manifest_file_path, nemo_file_path, output_file_path)

if __name__ == "__main__":
    main()
