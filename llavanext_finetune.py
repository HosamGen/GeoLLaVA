import av
import bisect
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BitsAndBytesConfig, LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Callback
from datasets import Dataset
from lightning.pytorch import Trainer

#======================================================================================================
# Constants
MAX_LENGTH = 256
MODEL_ID = "llava-hf/LLaVA-NeXT-Video-7B-hf"
MODEL_NAME = MODEL_ID.split("/")[-1]

# Configuration Flags
USE_LORA = False
USE_QLORA = True  # QLORA takes priority over LORA
USE_8BIT = False  # Change to use 8bit configuration with QLoRA
PRUNE = False  # Change this to use pruning
prune_amount = 0.05  # Pruning percentage (5%)
DEVICE = 0

# Model Type for checkpointing
MODEL_TYPE = "sample"  # For 10k sample dataset
# MODEL_TYPE = "full"  # For the full 100k dataset

batch_size = 3  # Batch size for training

# LoRA Parameters
lora_r = 64
lora_alpha = 128

# Annotations and directories
if MODEL_TYPE == "sample":
    train_annotations = "./annotations/sample_annotations.json"
else:
    train_annotations = "./annotations/updated_train_annotations.json"
test_annotations = './annotations/updated_val_annotations.json'  # Needed for validation loop
train_directory = "./updated_train_videos"
test_directory = "./updated_val_videos"

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

def collate_read_video(example, path):
    # Some datasets have a start-end interval, so we try to get it if exists. Otherwise just set a very large end timestamp
    clip = read_video_pyav(f'{path}/{example["video"]}', example.get("start", 1), example.get("end", 1e+10))
    example["clip"] = clip
    return example

processor = LlavaNextVideoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "right" # during training, one always uses padding on the right

class LlavaNextDataset(Dataset):
    """
    PyTorch Dataset for LlavaNextDataset. This class takes a HuggingFace Dataset as input.
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
    # answer_choice = batch["answer"] 

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


# Convert these dictionaries to HuggingFace datasets
train_dataset_tmp = Dataset.from_dict(train_dataset_dict)
test_dataset_tmp = Dataset.from_dict(test_dataset_dict)


train_dataset = LlavaNextDataset(train_dataset_tmp, train_directory)
test_dataset = LlavaNextDataset(test_dataset_tmp, test_directory)

## Load model
# Three options for training, from the lowest precision training to the highest precision training:
# QLoRA: model uses 4-bit quantization, which helps in reducing memory usage while maintaining performance.
# Standard LoRA:  model is loaded with standard LoRA adaptations.
# Full Fine-Tuning: no memory optimization are done. In that case Flash Attention is used to speed up training, if hardware supports it.

if USE_QLORA:
    # QLORA setup with quantization 4bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    if USE_8BIT:
    # #QLORA setup with quantization 8bit
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        device_map={"": DEVICE},
    )
elif USE_LORA:
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

    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        device_map={"": DEVICE},
    )
else:
    # Full fine-tuning with Flash Attention
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        _attn_implementation="flash_attention_2",
        device_map={"": DEVICE},
    )


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
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=0.1,
    target_modules=find_all_linear_names(model),
    init_lora_weights="gaussian",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

class LlavaNextModelPLModule(L.LightningModule):
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

def fine_grained_prune(tensor: torch.Tensor, sparsity: float) -> torch.Tensor:
    """
    Magnitude-based pruning for a single tensor.

    :param tensor: torch.(cuda.)Tensor, weight of conv/fc layer
    :param sparsity: float, pruning sparsity
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    :return:
        torch.(cuda.)Tensor, mask for zeros
    """
    # Ensure sparsity is within bounds
    sparsity = min(max(0.0, sparsity), 1.0)

    # If full pruning is required
    if sparsity == 1.0:
        tensor.zero_()
        return torch.zeros_like(tensor, dtype=tensor.dtype)

    # If no pruning is required
    elif sparsity == 0.0:
        return torch.ones_like(tensor, dtype=tensor.dtype)

    # Total number of elements in the tensor
    num_elements = tensor.numel()

    # Step 1: calculate the number of zeros after pruning
    num_zeros = round(num_elements * sparsity)

    # Step 2: calculate the importance of weight
    importance = torch.abs(tensor)

    # Step 3: calculate the pruning threshold
    if num_zeros > 0:
        threshold, _ = torch.kthvalue(importance.view(-1), num_zeros)
    else:
        threshold = torch.min(importance)

    # Step 4: get binary mask (1 for nonzeros, 0 for zeros)
    mask = torch.gt(importance, threshold).to(tensor.dtype)

    # Step 5: apply mask to prune the tensor
    tensor.mul_(mask)

    return mask

class FineGrainedPruner:
    def __init__(self, model, global_sparsity: float):
        self.global_sparsity = global_sparsity
        self.masks = FineGrainedPruner.prune(model, global_sparsity)

    @torch.no_grad()
    def apply(self, model):
        for name, param in model.named_parameters():
            if ("lm_head" not in name): 
                if ("video_tower" not in name):
                    if name in self.masks:
                        param *= self.masks[name]

    @staticmethod
    @torch.no_grad()
    def prune(model, global_sparsity: float):
        # List of layers that should not be pruned
        layers_to_skip = [
            "model.video_tower.vision_model.embeddings",
            "model.language_model.lm_head",
            "model.language_model.model.embed_tokens"
        ]
        # Adding encoder layers dynamically to the list
        for i in range(24):
            layers_to_skip.append(f"model.video_tower.vision_model.encoder.layers[{i}].mlp")

        masks = dict()
        total_params = 0
        pruned_params = 0
        
        for name, param in model.named_parameters():
            # Check if the layer is in the list of layers to skip
            if any(layer in name for layer in layers_to_skip):
                # print(f"Skipping pruning for layer: {name}")
                continue  # Skip pruning this layer

            if param.dim() > 1:  # Only prune conv and fc weights
                # print(f"Pruning layer: {name} with sparsity: {global_sparsity}")
                mask = fine_grained_prune(param, global_sparsity)
                masks[name] = mask

                # Calculate the number of parameters for statistics
                total_params += param.numel()
                pruned_params += (mask == 0).sum().item()

        print(f"Total parameters: {total_params}")
        print(f"Pruned parameters: {pruned_params}")
        print(f"Actual sparsity: {pruned_params / total_params:.2%}")

        return masks
    
class PruningCallback(Callback):
    def __init__(self, pruner):
        super().__init__()
        self.pruner = pruner

    def on_train_epoch_end(self, trainer, pl_module):
        # Apply the pruning to the model at the end of each training epoch
        print(f"Applying pruning at the end of epoch {trainer.current_epoch}")
        self.pruner.apply(pl_module.model)
        
if PRUNE:

    global_sparsity = prune_amount  # Set your desired sparsity level here
    # Ensure 'model' is defined before initializing the pruner
    pruner = FineGrainedPruner(model, global_sparsity)

    pruning_callback = PruningCallback(pruner)


config = {"max_epochs": 1,
          "val_check_interval": 0.2, # how many times we want to validate during an epoch
          "check_val_every_n_epoch": 1,
          "gradient_clip_val": 1.0,
          "accumulate_grad_batches": 1,
          "lr": 1e-4,
          "batch_size": batch_size,
          "num_nodes": 1,
          "warmup_steps": 50,
}

model_module = LlavaNextModelPLModule(config, processor, model)
early_stop_callback = EarlyStopping(monitor="train_loss", patience=3, verbose=True, mode="min")

lora_type = "QLORA" if USE_QLORA else "LORA"
bit_type = "8bit" if USE_8BIT else "4bit"

MODEL_PATH = f"./outputs/{MODEL_NAME}_{MODEL_TYPE}_{lora_type}_{bit_type}_r{lora_r}_alpha{lora_alpha}/"

print(MODEL_PATH)

# Define checkpoint callback to save only the most recent 5 checkpoints
checkpoint_callback = ModelCheckpoint(
    save_top_k=5,  # Keeps only the best 5 checkpoints
    monitor="train_loss",  # Monitor training loss for checkpointing
    mode="min",  # Minimize the train_loss
    save_last=True,  # Always save the latest checkpoint
    dirpath=MODEL_PATH,  # Path to save the checkpoints
    filename="llavanext-{epoch:02d}-{train_loss:.2f}"  # Checkpoint file naming convention
)

trainer = Trainer(
    default_root_dir=MODEL_PATH,
    accelerator="gpu",
    devices=[DEVICE],
    max_epochs=config.get("max_epochs"),
    accumulate_grad_batches=config.get("accumulate_grad_batches"),
    check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
    gradient_clip_val=config.get("gradient_clip_val"),
    precision="16-mixed",
    limit_val_batches=1,
    num_sanity_val_steps=1,
    callbacks=[early_stop_callback, checkpoint_callback],  # Add checkpoint callback here
    log_every_n_steps=1
)


trainer.fit(model_module)

# # Save the processor and model locally
processor.save_pretrained(MODEL_PATH)
model.save_pretrained(MODEL_PATH)
