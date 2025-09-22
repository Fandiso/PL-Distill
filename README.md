# PL-Distill: 

# A Knowledge Distillation Framework for LALMs



## Abstract

The emergence of Large Audio-Language Models (LALMs) has advanced Speech Emotion Recognition (SER), but their size limits deployment in resource-constrained environments. While Knowledge Distillation is effective for LALM compression, existing methods remain underexplored in distilling the cross-modal projection module (Projector), and often struggle with alignment due to differences in feature dimensions. We propose **PL-Distill**, a KD framework that combines **Projector-Level Distillation** (PDist) to align audio embeddings and **Logits-Level Distillation** (LDist) to align output logits. PDist introduces **Attention-weighted Centered Kernel Alignment**, a novel approach we proposed to highlight important time steps and address dimension mismatches. Meanwhile, LDist minimizes the Kullback-Leibler divergence between teacher and student logits from audio and text modalities. On IEMOCAP, RAVDESS, and SAVEE, PL-Distill compresses an 8.4B-parameter teacher to a compact 1.1B-parameter student, consistently outperforming the teacher, state-of-the-art pretrained models, and other KD baselines across all metrics.

------



## Overview

<img width="2050" height="667" alt="image-20250919172137488" src="https://github.com/user-attachments/assets/1b586f9a-46ee-4afd-82dd-7f1fe2499345" />


​                 **Fig. 1**. An overview of our PL-Distill framework, which includes Projector-level Distillation (PDist) and Logits-level Distillation (LDist).

------



## Evaluation

<img width="1942" height="574" alt="image-20250919172657384" src="https://github.com/user-attachments/assets/6a566657-9e91-426d-9eef-dbe2ba90e949" />


**Table 1**. Comparison of main performance metrics for various models on the IEMOCAP, RAVDESS, and SAVEE datasets. Results for pretrained Models are cited from the Emobox benchmark. The baseline for comparison refers to the Forward KL method. SOTA results among Pretrained Models are highlighted in **bold**, and the overall best results across all models are highlighted in **red**.

------



## Prepare the environment

```bash
pip install git+https://github.com/huggingface/transformers
pip install peft
pip install argparse
pip install librosa
pip install urllib
pip install sklearn
```

------



## Student model initialization weights

- We utilize the Whisper large v3  model as the student's audio encoder. And we load the weights from the Qwen2-audio.
- We use the smaller Qwen2-0.5B as the student's LLM , and initialized from Qwen2-0.5B-Instruct to leverage its language capabilities.



## Quickstart

1.Download the pre-trained weights from the Qwen2-audio's audio encoder and the Qwen2-0.5B-Instruct, then merge them into the "*model.safetensors",* and place them in the Qwen2-Audio-0.5B folder.

2.Download the Datasets from [IEMOCAP- Release](https://sail.usc.edu/iemocap/iemocap_release.htm), [RAVDESS Emotional speech audio](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio), [Surrey Audio-Visual Expressed Emotion (SAVEE)](https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee)  and place them in the path:(/PL-Distill/Emobox/data/Downloads)

3.run the code to process data

```
python process_iemocap.py
```

4.Finally, run the following code: 

```
python evaluate.py \
  --prediction_path /Path to save your prediction results and ground truth \
  --scores_path /Path to save your review score \
  --lora_path /Path to Checkpoint(e.g., ./checkpoint_test/checkpoint_iemocap) \
  --data_path /Path to your test data(e.g., PL_Distill/Emobox/data/iemocap/iemocap_test_fold_1.jsonl)
```

