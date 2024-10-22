import os
import av
import bisect
import numpy as np
import torch
from torch.utils.data import Dataset
import json
from datasets import Dataset
import math
import time
from transformers import BitsAndBytesConfig, LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor

# ================================================================================================

MAX_LENGTH = 256
MODEL_ID = "llava-hf/LLaVA-NeXT-Video-7B-hf"

# Configuration
USE_BASE = False
DEVICE = 0

test_annotations = './annotations/updated_val_annotations.json'
test_directory = "./updated_val_videos"
MODEL_PATH = "./outputs/LLaVA-NeXT-Video-7B-hf_demo_full_QLORA_8bit_r64_alpha128"
MODEL_TAG = MODEL_PATH.split("/")[-1]

# ================================================================================================

def read_video_pyav(video_path, start, end):
    """Reads a video for given start-end timestamps interval and uniformly samples 8 frames of it"""
    container = av.open(video_path)
    video = container.streams.get(0)[0]

    av_timestamps = [
        int(packet.pts * video.time_base) for packet in container.demux(video) if packet.pts is not None
    ]

    av_timestamps.sort()
    start_id = bisect.bisect_left(av_timestamps, start)
    end_id = bisect.bisect_left(av_timestamps, end)

    if end_id - start_id < 10:
        end_id = min(len(av_timestamps) - 1, end_id + 10)
        start_id = max(0, start_id - 10)

    end_id = min(len(av_timestamps) - 1, end_id)
    start_id = max(0, start_id)

    num_frames_to_sample = min(2, end_id - start_id + 1)
    indices = np.linspace(start_id, end_id, num_frames_to_sample).astype(int)

    frames = []
    container.seek(0)

    for i, frame in enumerate(container.decode(video=0)):
        if i > end_id:
            break
        if i >= start_id and i in indices:
            frames.append(frame)

    assert len(frames) == 2, f"Got {len(frames)} frames but should be 2. Check the indices: {indices}; start_id: {start_id}, end_id: {end_id}. Len of video is {len(av_timestamps)} frames."

    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def collate_read_video(example, path):
    # Some datasets have a start-end interval, so we try to get it if exists. Otherwise just set a very large end timestamp
    clip = read_video_pyav(f'{path}/{example["video"]}', example.get("start", 0), example.get("end", 1e+10))
    example["clip"] = clip
    return example

class LlavaNextDataset(Dataset):
    """PyTorch Dataset for LlavaNextDataset. This class takes a HuggingFace Dataset as input."""
    
    def __init__(self, dataset, video_path):
        super().__init__()
        self.dataset = dataset
        self.video_path = video_path

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        clip = read_video_pyav(f'{self.video_path}/{sample["video"]}', sample.get("start", 0), sample.get("end", 1e+10))
        answer = sample['conversations'][1]['value']
        tmp_prompt = sample['conversations'][0]['value']

        prompt = f"USER: {tmp_prompt}\n ASSISTANT: Answer: {answer}"
        return prompt, clip, answer

# Load test annotations
with open(test_annotations, 'r') as file:
    test_data = json.load(file)

# Create dictionary for testing data
test_dataset_dict = {
    "video": [item['video'] for item in test_data],
    "conversations": [item['conversations'] for item in test_data],
}

# Convert dictionaries to HuggingFace datasets
test_dataset_tmp = Dataset.from_dict(test_dataset_dict)
test_dataset = LlavaNextDataset(test_dataset_tmp, test_directory)

print(MODEL_PATH)

start_time = time.time()

if USE_BASE:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    processor = LlavaNextVideoProcessor.from_pretrained(MODEL_ID)
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map=DEVICE
    )

    results = []

    # Open the file in append mode before the loop
    with open(f'results_base_LLaVA-NeXT-Video.json', 'a') as f:
        for test in test_data:
            true_value = test['conversations'][1]['value']

            # Generate the predicted response
            inputs = processor(
                text=test['conversations'][0]['value'],
                videos=read_video_pyav(f'{test_directory}/{test["video"]}', 0, 1e+10),
                padding=True,
                return_tensors="pt"
            ).to(model.device)

            generate_kwargs = {"max_new_tokens": 256, "do_sample": True, "top_p": 0.9}
            output = model.generate(**inputs, **generate_kwargs)
            generated_text = processor.batch_decode(output, skip_special_tokens=True)

            # Create result entry
            result_entry = {
                'id': test['id'],
                'video': test['video'],
                'true': true_value,
                'generated': generated_text[0]  # Add the cleaned text from the batch
            }

            # Write each entry as a separate JSON object
            f.write(json.dumps(result_entry) + '\n')

else:
    print("Using local model")

    # Load processor and model from local directory
    processor = LlavaNextVideoProcessor.from_pretrained(MODEL_PATH)

    # Define quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        quantization_config=quantization_config,
        device_map=DEVICE
    )

    results = []
    batch_size = 25  # Define your batch size here
    
    model.eval()

    # Check if file already exists
    file_path = f"results_{MODEL_TAG}.json"
    if not os.path.exists(file_path):
        # Create a new file with an opening array bracket if it doesn't exist
        with open(file_path, 'w') as f:
            f.write('[')

    with torch.no_grad():
        # Split test data into batches
        for i in range(math.ceil(len(test_data) / batch_size)):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(test_data))
            
            # Collect texts and videos for the current batch
            batch_texts = [test['conversations'][0]['value'] for test in test_data[start_idx:end_idx]]
            batch_videos = [read_video_pyav(f'{test_directory}/{test["video"]}', 0, 1e+10) for test in test_data[start_idx:end_idx]]
            
            # Process the entire batch at once
            inputs = processor(text=batch_texts, videos=batch_videos, return_tensors="pt", padding=True).to(model.device)
            generated_ids = model.generate(**inputs, max_new_tokens=MAX_LENGTH)
            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

            # Append new results directly to the file
            with open(file_path, 'a') as f:
                for idx, test in enumerate(test_data[start_idx:end_idx]):
                    true_value = test['conversations'][1]['value']
                    result_entry = {
                        'id': test['id'],
                        'video': test['video'],
                        'true': true_value,
                        'generated': generated_texts[idx]
                    }
                    
                    # Add a comma before each entry except the first one to keep JSON valid
                    if os.path.getsize(file_path) > 1:  # File is not empty
                        f.write(',\n')
                    
                    # Write the result entry
                    json.dump(result_entry, f)

        # Close the JSON array if it's the last batch
        if i == math.ceil(len(test_data) / batch_size) - 1:
            with open(file_path, 'a') as f:
                f.write('\n]')  # Close the JSON array

    print("--- %s seconds ---" % (time.time() - start_time))
