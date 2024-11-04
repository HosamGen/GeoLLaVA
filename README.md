# GeoLLaVA: Efficient Fine-Tuned Vision-Language Models for Temporal Change Detection in Remote Sensing 🌍

GeoLLaVA is designed to enhance vision-language models (VLMs) for detecting temporal changes in remote sensing data. By leveraging fine-tuning techniques like LoRA and QLoRA, it significantly improves model performance in tasks such as environmental monitoring and urban planning, especially in detecting geographical landscape evolution over time.

#### [Hosam Elgendy](https://scholar.google.com/citations?user=6RA4_m8AAAAJ&hl=en&oi=ao), [Ahmed Sharshar](https://scholar.google.com/citations?user=GC8A9k0AAAAJ&hl=en), [Ahmed Aboeitta](https://scholar.google.com/citations?user=sEZTgaYAAAAJ&hl=en&oi=ao), [Yasser Ashraf](https://scholar.google.com/citations?user=Q_r99BgAAAAJ&hl=en&oi=ao) and [Mohsen Guizani](https://scholar.google.com/citations?user=RigrYkcAAAAJ&hl=en&oi=ao)
#### Mohamed bin Zayed University of AI (MBZUAI)

---
<p align='center'>
<img src="assets/Overview.jpg" height="400">
</p>

---

## Contents
- [Setup](#setup)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

---

## Setup

1. Clone this repository:
    ```shell
    git clone https://github.com/HosamGen/GeoLLaVA.git
    cd GeoLLaVA
    ```

2. Install the necessary dependencies:
    ```shell
    conda create -n geollava python=3.10
    conda activate geollava
    pip install -r requirements.txt
    ```

---

## GeoLLaVA Custom Dataset

[OPTIONAL] Please refer to the [fMoW dataset](https://github.com/fMoW/dataset?tab=readme-ov-file) for the original remote sensing dataset. We provide cleaned annotations in the [Annotations]() section below.

> [!NOTE]
> The full 100K annotations are too large for direct download and can be accessed via [Drive](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/hosam_elgendy_mbzuai_ac_ae/Es2IRaXpBPRAk2gX6J5IDsgBBttITHCHbxpr4FIcRVWleg?e=pCKhFH).

The videos used in this project can also be found on [Drive](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/hosam_elgendy_mbzuai_ac_ae/Es2IRaXpBPRAk2gX6J5IDsgBBttITHCHbxpr4FIcRVWleg?e=pCKhFH) and unzipped using the following commands:

```shell
unzip updated_train_videos.zip
unzip updated_val_videos.zip
```

Your directory structure should look like this:
```
GeoLLaVA
├── annotations
|    ├── updated_train_annotations.json
|    ├── updated_val_annotations.json
├── updated_train_videos
|    ├── airport_hangar_0_4-airport_hangar_0_2.mp4
|    |   .....
├── updated_val_videos
|    ├── airport_hangar_0_4-airport_hangar_0_1.mp4
|    |   .....
├── llavanext_eval.py
├── llavanext_finetune.py
├── videollava_finetune.py
├── videollava_test.py
...
```

## Training

To fine-tune the model on the dataset, run the `videollava_finetune.py` or `llavanext_finetune.py` scripts, depending on your model configuration.

For Video-LLaVA:
```shell
python videollava_finetune.py
```

For LLaVA-NeXT:
```shell
python llavanext_finetune.py
```

Modify parameters such as:
```shell
MAX_LENGTH = 256
USE_LORA = False
USE_QLORA = True 
USE_8BIT = False 
PRUNE = False 
prune_amount = 0.05 
MODEL_TYPE = "sample" #for 10k sample dataset
# MODEL_TYPE = "full" #for the full 100k dataset
batch_size = 2

#lora parameters
lora_r = 64
lora_alpha = 128
```

## Evaluation

To evaluate the fine-tuned models on the test dataset, use the following commands:

For Video-LLaVA:
```shell
python videollava_test.py
```

For LLaVA-NeXT:
```shell
python llavanext_eval.py
```

> [!IMPORTANT]
> The MODEL_PATH must be changed during evaluation based on the model that was finetuned.

These commands will run the evaluation on the specified test dataset and generate performance metrics, including ROUGE, BLEU, and BERT scores. The results will help assess the model's performance in detecting temporal changes in remote sensing data.

## Results

We evaluated the performance of GeoLLaVA across various metrics, including ROUGE, BLEU, and BERT scores. The fine-tuned model demonstrated significant improvements in capturing and describing temporal changes in geographical landscapes.

To calculate the scores after evaluating the models, please check the steps in the [Results.ipynb](https://github.com/HosamGen/GeoLLaVA/blob/main/Results.ipynb) notebook.

## Video-LLaVA Results

| Model                | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU  | BERT  |
|----------------------|---------|---------|---------|-------|-------|
| **Base**             | 0.211   | 0.041   | 0.122   | 0.039 | 0.456 |
| **10K LoRA**         | 0.563   | 0.214   | 0.313   | 0.243 | 0.849 |
| **100K LoRA**        | **0.576**   | **0.226**   | **0.325**   | **0.250** | **0.863** |
| **10K QLoRA**        | 0.565   | 0.212   | 0.310   | 0.243 | 0.845 |
| **100K QLoRA**       | 0.571   | 0.220   | 0.316   | **0.250** | 0.854 |
| **10K Pruning 5%**   | 0.031   | 0.007   | 0.024   | 0.010 | 0.265 |
| **100K Pruning 5%**  | 0.125   | 0.034   | 0.110   | 0.043 | 0.359 |

## LLaVA-NeXT Results

| Model                | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU  | BERT  |
|----------------------|---------|---------|---------|-------|-------|
| **Base**             | 0.197   | 0.037   | 0.113   | 0.042 | 0.404 |
| **10K LoRA**         | 0.554   | 0.198   | 0.300   | 0.232 | 0.856 |
| **100K LoRA**        | **0.562**   | 0.199   | 0.300   | **0.239** | **0.864** |
| **10K QLoRA**        | 0.543   | 0.193   | 0.283   | 0.213 | 0.836 |
| **100K QLoRA**       | 0.561   | **0.202**   | **0.302**   | 0.229 | 0.858 |
| **10K Pruning 5%**   | 0.532   | 0.178   | 0.278   | 0.209 | 0.829 |
| **100K Pruning 5%**  | 0.541   | 0.183   | 0.284   | 0.210 | 0.840 |

**Final Model (100K LoRA)** | **0.556** | **0.202** | **0.290** | **0.227** | **0.850** |



These metrics illustrate how well the models performed in describing temporal changes in remote sensing data, with fine-tuning techniques like LoRA and QLoRA leading to notable improvements.

## Acknowledgement

+ [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA/) Video-LLaVA: Learning United Visual Representation by Alignment Before Projection. We have used Video-LLaVA as one of the models for finetuning.
+ [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) LLaVA-NeXT: Open Large Multimodal Models. The video model was used as the second model.
+ [fMoW RGB Dataset](https://github.com/fMoW/dataset) Original fMoW dataset repo.

## Citation
please cite using this BibTeX:
```bibtex
    @misc{elgendy2024geollava,
      title={GeoLLaVA: Efficient Fine-Tuned Vision-Language Models for Temporal Change Detection in Remote Sensing}, 
      author={Hosam Elgendy and Ahmed Sharshar and Ahmed Aboeitta and Yasser Ashraf and Mohsen Guizani},
      year={2024},
      eprint={2410.19552},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.19552}, 
}
```


