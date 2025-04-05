import av
import bisect
import numpy as np

from transformers import AutoProcessor
from transformers import BitsAndBytesConfig, VideoLlavaForConditionalGeneration
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import json
from datasets import Dataset

from transformers import AutoProcessor, BitsAndBytesConfig, VideoLlavaForConditionalGeneration

from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch import Trainer

import torch.nn.utils.prune as prune

import os
import argparse


MAX_LENGTH = 350
MODEL_ID = "LanguageBind/Video-LLaVA-7B-hf"
MODEL_NAME = MODEL_ID.split("/")[-1]

#======================================================================================================

# USE_LORA = False
# USE_QLORA = True   # <-- QLORA takes priority over LORA, change to False before changing LORA to true
# USE_8BIT = False  #Change to use 8bit configuration with qlora, otherwise, default is 4bit

# PRUNE = True #Change this to use pruning
# MAGNITUDE, ATTENTION, CHANNEL = True, False, False #Choose the type of pruning, leave only one True.

# prune_amount = 0.05 #pruning percentage (5% here)

DEVICE = int(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0])
print(DEVICE)

# #the MODEL_TYPE is used in naming the checkpoint/output model for easier reference
# # MODEL_TYPE = "sample" #for 10k sample dataset
# MODEL_TYPE = "full" #for the full 100k dataset

# batch_size = 3

# #lora parameters
# lora_r = 64
# lora_alpha = 128

parser = argparse.ArgumentParser(description="Configure video processing with optional LoRA, QLoRA, 8-bit quantization, and pruning.")

# LoRA, QLoRA, and 8-bit options
parser.add_argument("--use_lora", action="store_true", default=False, help="Enable LoRA")
parser.add_argument("--use_qlora", action="store_true", default=True, help="Enable QLoRA (takes priority over LoRA)")
parser.add_argument("--use_8bit", action="store_true", default=False, help="Use 8-bit configuration with QLoRA")

# Pruning options
parser.add_argument("--prune", action="store_true", default=True, help="Enable pruning")
parser.add_argument("--magnitude", action="store_true", default=False, help="Enable magnitude pruning")
parser.add_argument("--attention", action="store_true", default=False, help="Enable attention pruning")
parser.add_argument("--channel", action="store_true", default=False, help="Enable channel pruning")

# Pruning amount
parser.add_argument("--prune_amount", type=float, default=0.05, help="Set the pruning percentage (default is 0.05)")

# Model type and batch size
parser.add_argument("--model_type", type=str, default="full", help="Specify the model type (e.g., 'sample' or 'full')")
parser.add_argument("--batch_size", type=int, default=3, help="Batch size for processing")
parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for finetuning the model")

# LoRA parameters
parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank parameter")
parser.add_argument("--lora_alpha", type=int, default=128, help="LoRA alpha parameter")

args = parser.parse_args()

#===================================================================================================================================

if args.model_type == "sample":
    train_annotations = "./annotations/sample_annotations.json"
else:
    train_annotations = "./annotations/updated_train_annotations.json"
    
test_annotations =  './annotations/updated_val_annotations.json' #only needed for evaluating the model during the val loop (one sample)

train_directory = "/l/users/hosam.elgendy/updated_train_videos"
test_directory = "/l/users/hosam.elgendy/updated_val_videos"

##For easier evaluation, copy the MODEL_PATH that gets printed here and input it as the MODEL_PATH in the other code file

#======================================================================================================

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
    assert len(frames) == 2, f"Got {len(frames)} frames but should be 2. Check the indices: {indices};, start_id: {start_id}, end_id: {end_id}. Len of video is {len(av_timestamps)} frames."
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "right" # during training, one always uses padding on the right

from torch.utils.data import Dataset

class VideoLlavaDataset(Dataset):
    """
    PyTorch Dataset for VideoLlavaDataset. This class takes a HuggingFace Dataset as input.
    """

    def __init__(self, dataset, video_path):
        super().__init__()
        self.dataset = dataset
        self.video_path = video_path

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        # Lazy load video clip here
        clip = read_video_pyav(f'{self.video_path}/{sample["video"]}', sample.get("start", 0), sample.get("end", 1e+10))
        answer = sample['conversations'][1]['value']
        tmp_prompt = sample['conversations'][0]['value']

        prompt = f"USER: {tmp_prompt}" \
                 f"\n ASSISTANT: Answer: {answer}"

        return prompt, clip, answer
    
def train_collate_fn(examples):
    videos = []
    texts = []
    texts, videos, _ = list(zip(*examples))
    batch = processor(text=texts, videos=videos, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    labels = batch["input_ids"].clone()

    # We don't want to compute loss for pad tokens, lets mask with -100. Some methods also mask the prompt, calculating loss only on the answers/captions/etc
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values_videos = batch["pixel_values_videos"]
    labels = batch["labels"]

    return input_ids, attention_mask, pixel_values_videos, labels


def eval_collate_fn(examples):
    # We only feed the prompt to the model
    # Make sure to separate prompt from answers/captions/etc depending on your own task and dataset
    # Otherwise your model will peek into the ground truth
    videos = []
    texts = []
    true_answers = []
    texts, videos, true_answers = list(zip(*examples))
    texts = [text[:-2] for text in texts]  # Get text without answers, so the model has to generate the answers itself during eval
    batch = processor(text=texts, videos=videos, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values_videos = batch["pixel_values_videos"]

    return input_ids, attention_mask, pixel_values_videos, true_answers


with open(train_annotations, 'r') as file:
    train_data  = json.load(file)

with open(test_annotations, 'r') as file:
    test_data  = json.load(file)

# Create dictionary for training data
train_dataset_dict = {
    "video": [item['video'] for item in train_data],
    "conversations": [item['conversations'] for item in train_data],
}

# Create dictionary for testing data
test_dataset_dict = {
    "video": [item['video'] for item in test_data],
    "conversations": [item['conversations'] for item in test_data],
}


from datasets import Dataset

# Convert these dictionaries to HuggingFace datasets
train_dataset_tmp = Dataset.from_dict(train_dataset_dict)
test_dataset_tmp = Dataset.from_dict(test_dataset_dict)


train_dataset = VideoLlavaDataset(train_dataset_tmp, train_directory)
test_dataset = VideoLlavaDataset(test_dataset_tmp, test_directory)

## Load model
# Three options for training, from the lowest precision training to the highest precision training:
# QLoRA: model uses 4-bit quantization, which helps in reducing memory usage while maintaining performance.
# Standard LoRA:  model is loaded with standard LoRA adaptations.
# Full Fine-Tuning: no memory optimization are done. In that case Flash Attention is used to speed up training, if hardware supports it.

if args.use_qlora:
    # QLORA setup with quantization 4bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    if args.use_8bit:
    # #QLORA setup with quantization 8bit
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    model = VideoLlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        device_map="auto",
    )


elif args.use_lora:
    # LORA setup without quantization

    bnb_config = BitsAndBytesConfig(
    load_in_8bit=False,
    load_in_4bit=False,
    llm_int8_threshold=0.5,  # Lower threshold for increased precision
    llm_int8_skip_modules=None,  # None if skipping is not needed
    llm_int8_enable_fp32_cpu_offload=False,
    llm_int8_has_fp16_weight=True,  # Use FP16 weights for better precision
    bnb_4bit_compute_dtype=torch.float16  # Ensures highest precision in computations
    )

    model = VideoLlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        device_map="auto",
    )
else:
    # Full fine-tuning with Flash Attention
    model = VideoLlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        _attn_implementation="flash_attention_2",
        device_map="auto",
    )


def get_num_parameters(model: torch.nn.Module, count_nonzero_only=False) -> int:
    """
    calculate the total number of parameters of model
    :param count_nonzero_only: only count nonzero weights
    """
    num_counted_elements = 0
    for param in model.parameters():
        if count_nonzero_only:
            num_counted_elements += param.count_nonzero()
        else:
            num_counted_elements += param.numel()
    return num_counted_elements


def get_model_size(model: torch.nn.Module, data_width=32, count_nonzero_only=False) -> int:
    """
    calculate the model size in bits
    :param data_width: #bits per element
    :param count_nonzero_only: only count nonzero weights
    """
    return get_num_parameters(model, count_nonzero_only) * data_width

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB

dense_model_size = get_model_size(model)
print(f"BEFORE PRUNING dense model has size={dense_model_size/MiB:.2f} MiB")
      

if args.prune:

    if args.magnitude:
            # Apply magnitude-based pruning to selected layers
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):  # Apply pruning to all Linear layers, adjust as needed
                prune.l1_unstructured(module, name='weight', amount=args.prune_amount)  # Adjust the amount as per your requirements
                prune.remove(module, 'weight')  # Remove the pruning mask after pruning

    if args.attention:
         # Apply structured pruning to attention heads
        for name, module in model.named_modules():
            # Identify attention modules for head pruning
            if hasattr(module, 'num_heads') and hasattr(module, 'q_proj'):
                # Calculate number of heads to prune based on pruning ratio
                num_heads_to_prune = int(module.num_heads * args.prune_amount)
                if num_heads_to_prune > 0:
                    # print(f"Pruning {num_heads_to_prune} heads from {name}")

                    # Example pruning strategy: remove heads with smallest weights
                    # Reshape to separate heads
                    q_proj_weights = module.q_proj.weight.view(module.num_heads, -1).clone()  # Detach copy for modification
                    # Get indices of heads with the lowest L2-norm weight
                    # head_norms = q_proj_weights.norm(dim=1)
                    head_norms = q_proj_weights.float().norm(dim=1)

                    heads_to_prune = torch.topk(head_norms, num_heads_to_prune, largest=False).indices

                    # Zero out the weights of the selected heads
                    for head in heads_to_prune:
                        q_proj_weights[head] = 0

                    # Assign pruned weights back to the q_proj layer
                    module.q_proj.weight.data.copy_(q_proj_weights.view_as(module.q_proj.weight))
                    
                    # Optional: apply similar pruning to k_proj, v_proj if needed


    if args.channel:
        # Set pruning amounts
        head_pruning_ratio = args.prune_amount  # Prune 20% of attention heads
        channel_pruning_ratio = args.prune_amount  # Prune 20% of channels in convolutional layers

        # Apply attention head pruning
        for name, module in model.named_modules():
            if hasattr(module, 'num_heads') and hasattr(module, 'q_proj'):
                num_heads_to_prune = int(module.num_heads * head_pruning_ratio)
                if num_heads_to_prune > 0:
                    print(f"Pruning {num_heads_to_prune} heads from {name}")
                    q_proj_weights = module.q_proj.weight.view(module.num_heads, -1).clone()
                    # head_norms = q_proj_weights.norm(dim=1)
                    head_norms = q_proj_weights.float().norm(dim=1)
                    heads_to_prune = torch.topk(head_norms, num_heads_to_prune, largest=False).indices
                    for head in heads_to_prune:
                        q_proj_weights[head] = 0
                    module.q_proj.weight.data.copy_(q_proj_weights.view_as(module.q_proj.weight))

        # Apply channel pruning on convolutional layers
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                num_channels_to_prune = int(module.out_channels * channel_pruning_ratio)
                if num_channels_to_prune > 0:
                    print(f"Pruning {num_channels_to_prune} channels from {name}")
                    channel_norms = module.weight.view(module.out_channels, -1).norm(dim=1)
                    channels_to_prune = torch.topk(channel_norms, num_channels_to_prune, largest=False).indices
                    
                    # Set selected channel weights to zero (pruning)
                    for channel in channels_to_prune:
                        module.weight.data[channel] = 0
                        if module.bias is not None:
                            module.bias.data[channel] = 0  # Prune corresponding bias


dense_model_size = get_model_size(model)
print(f"AFTER PRUNING dense model has size={dense_model_size/MiB:.2f} MiB")

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_model']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


lora_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=0.1,
    target_modules=find_all_linear_names(model),
    init_lora_weights="gaussian",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

class VideoLlavaModelPLModule(L.LightningModule):
    def __init__(self, config, processor, model):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model
        self.batch_size = config.get("batch_size")

    def training_step(self, batch, batch_idx):

        input_ids, attention_mask, pixel_values_videos, labels = batch

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values_videos=pixel_values_videos,
            labels=labels
        )
        loss = outputs.loss

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):

        input_ids, attention_mask, pixel_values_videos, answers = batch

        # autoregressively generate token IDs
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values_videos=pixel_values_videos,
            max_new_tokens=MAX_LENGTH,
            do_sample=False,
        )
        # turn them back into text, chopping of the prompt
        predictions = self.processor.batch_decode(generated_ids[:, input_ids.size(1):], skip_special_tokens=True)

        correct = 0
        for pred, answer in zip(predictions, answers):
            correct += (pred.strip().lower() == answer.lower())
        self.log("val_accuracy", correct / len(answers))

        return correct

    def configure_optimizers(self):
        # you could also add a learning rate scheduler if you want
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.get("lr"))

        return optimizer

    def train_dataloader(self):
        return DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(test_dataset, collate_fn=eval_collate_fn, batch_size=self.batch_size, shuffle=False, num_workers=8)
    

config = {"max_epochs": args.epochs,
          "val_check_interval": 0.2, # how many times we want to validate during an epoch
          "check_val_every_n_epoch": 1,
          "gradient_clip_val": 1.0,
          "accumulate_grad_batches": 1,
          "lr": 1e-4,
          "batch_size": args.batch_size,
          "num_nodes": 1,
          "warmup_steps": 50,
}

model_module = VideoLlavaModelPLModule(config, processor, model)
# early_stop_callback = EarlyStopping(monitor="train_loss", patience=3, verbose=True, mode="min")


lora_type = "QLORA" if args.use_qlora else "LORA"
bit_type = "8bit" if args.use_8bit else "4bit"

prune_type = ""
if args.prune:
    if args.magnitude:
        prune_type = "prune_mag_"
    elif args.attention:
        prune_type = "prune_attn_"
    elif args.channel:
        prune_type = "prune_chnl_"
    else:
        prune_type = "prune_"

prune_perc = str(args.prune_amount*100) + "_" if args.prune else ""


MODEL_PATH = f"./outputs/{prune_type}{prune_perc}{MODEL_NAME}_{args.model_type}_{lora_type}_{bit_type}_r{args.lora_r}_alpha{args.lora_alpha}_{args.epochs}epochs/"


print(MODEL_PATH)



# Define checkpoint callback to save only the most recent 5 checkpoints (modified to save the last checkpoint to save space)
checkpoint_callback = ModelCheckpoint(
    save_top_k=1,  # Keeps only the last checkpoint
    monitor="train_loss",  # Monitor training loss for checkpointing
    mode="min",  # Minimize the train_loss
    save_last=True,  # Always save the latest checkpoint
    dirpath=MODEL_PATH,  # Path to save the checkpoints
    filename="videollava-{epoch:02d}-{train_loss:.2f}"  # Checkpoint file naming convention
)

trainer = Trainer(
    default_root_dir=MODEL_PATH,
    accelerator="gpu",
    devices=[0],
    max_epochs=config.get("max_epochs"),
    accumulate_grad_batches=config.get("accumulate_grad_batches"),
    check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
    gradient_clip_val=config.get("gradient_clip_val"),
    precision="16-mixed",
    limit_val_batches=1,
    num_sanity_val_steps=1,
    callbacks=[checkpoint_callback],  # Add checkpoint callback here
    log_every_n_steps=1
)


trainer.fit(model_module)


# # Save the processor and model locally
processor.save_pretrained(MODEL_PATH)
model.save_pretrained(MODEL_PATH)


