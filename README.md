# Bridging Domain Shifts and Detecting Anomalies in Urban Driving

This repository explores closed-set semantic/panoptic segmentation and open-world out-of-distribution (OoD) anomaly segmentation for autonomous driving. It benchmarks a fast, pixel-based convolutional baseline (**ERFNet**) against a high-capacity, mask-based Vision Transformer (**EoMT** with DINOv2) to address accuracy–latency trade-offs, dataset domain shifts (MS COCO to Cityscapes), and anomaly detection.

---

##  Project Overview

* **Domain Adaptation & Progressive Fine-Tuning**: Adapts an EoMT model pre-trained on generic MS COCO (133 classes) to urban driving scenes in Cityscapes (19 classes). A 3-stage progressive fine-tuning strategy (head-only $\rightarrow$ top ViT blocks $\rightarrow$ full model) boosts closed-set semantic performance from **55.1% to 78.8% mIoU**.


* **Cross-Dataset Taxonomy Mapping**: Maps 25 shared classes between COCO and Cityscapes while routing unmodeled classes to an ignore index.


* **Mask Collapse Anomaly Detection**: Unlike pixel-wise CNNs that force unknown anomalies into known classes due to softmax constraints, mask transformers isolate mask extraction from classification. Unseen objects fail to trigger masks (mask collapse), translating directly into peak uncertainty for post-hoc anomaly scorers.


* **Uncertainty & Calibration Scorers**: Evaluates Maximum Softmax Probability (MSP), MaxLogit, Predictive/Max Entropy, and Rejected by All (RbA), optimized via Temperature Scaling ($T=2.0$) across five road anomaly benchmarks (SMIYC RA-21, SMIYC RO-21, Fishyscapes Lost&Found, Fishyscapes Static, Road Anomaly).



---

##  Repository Structure & File Breakdown

### Evaluation & Baseline (`/eval`)

* **`eval/evalAnomaly.py`**: Runs anomaly segmentation benchmarks (AUPRC, FPR95) for pixel-level models (ERFNet) using post-hoc scorers.


* **`eval/evalAnomalyEOMT.py`**: Executes sliding-window out-of-distribution anomaly evaluation on EoMT checkpoints across all benchmark datasets.


* **`eval/eval_iou.py`**: Measures closed-set mean Intersection-over-Union (mIoU) on semantic validation splits.


* **`eval/iouEval.py`**: Computes the confusion matrix, IoU metrics, and per-class performance tracking.


* **`eval/dataset.py`**: Data loader and pipeline for Cityscapes and OoD anomaly datasets.


* **`eval/transform.py`**: Image transformation, resizing, and normalization routines for evaluation.


* **`eval/erfnet.py`** & **`eval/erfnet_nobn.py`**: PyTorch implementations of ERFNet with residual factorized 1D convolutions (with and without Batch Normalization).


* **`eval/eval_forwardTime.py`**: Benchmarks model runtime, forward pass latency, and FPS.


* **`eval/eval_cityscapes_color.py`** & **`eval/eval_cityscapes_server.py`**: Utilities to export colorized visual predictions and format outputs for the official Cityscapes benchmark.


* **`eval/results.txt`**: Raw benchmark metrics and validation evaluation records.



---

### Model, Training & Configurations (`/eomt`)

* **`eomt/main.py`**: Main CLI entrypoint to launch training, validation, and fine-tuning experiments.


* **`eomt/inference.ipynb`**: Interactive notebook for model checkpoint inference, prediction plotting, and anomaly map visualization.


* **`eomt/models/eomt.py`**: Implementation of the Encoder-only Mask Transformer architecture (query classification and mask generation).


* **`eomt/models/vit.py`**: Vision Transformer (ViT) backbone loaded with DINOv2 self-supervised weights.


* **`eomt/models/scale_block.py`**: Multi-scale spatial feature projection and upsampling blocks.


* **`eomt/callbacks/freeze.py`**: Lightning callback executing the progressive unfreezing schedule across training epochs.


* **`eomt/configs/dinov2/`**: YAML experiment configurations:
* `coco/panoptic/eomt_finetuning_step1.yaml`: Stage 1 frozen-backbone head adaptation.


* `coco/panoptic/eomt_finetuning_step2.yaml`: Stage 2 unfreezing of the top 2 ViT blocks.


* `coco/panoptic/eomt_finetuning_step3.yaml`: Stage 3 full end-to-end network optimization.


* `cityscapes/semantic/eomt_base_640.yaml`: Native Cityscapes semantic segmentation baseline.




* **`eomt/datasets/`**:
* `cityscapes_semantic.py`, `coco_panoptic.py`, `coco_instance.py`, `ade20k_semantic.py`, `ade20k_panoptic.py`: Dataset parsers for semantic, instance, and panoptic formats.


* `lightning_data_module.py`: PyTorch Lightning DataModule standardizing dataloaders and workers.


* `transforms.py`: Training data augmentations (random crops, scaling, flips).




* **`eomt/evaluation/mapping.py`**: Maps 133 COCO taxonomy IDs to the 19 Cityscapes evaluation classes (with ignore index 255).


* **`eomt/evaluation/utils_eomt.py`**: Mathematical definitions for anomaly scorers (MSP, MaxLogit, Max Entropy, clamped RbA) and Temperature Scaling.


* **`eomt/evaluation/run_eval.py`**: Evaluator orchestrating dataset-wide validation runs.


* **`eomt/training/`**:
* `lightning_module.py`: PyTorch Lightning system coordinating forward passes, optimizers, and checkpointing.


* `mask_classification_loss.py`: Bipartite Hungarian matching, cross-entropy, and dice losses.


* `mask_classification_semantic.py` / `_instance.py` / `_panoptic.py`: Target-specific query and mask decoders.


* `two_stage_warmup_poly_schedule.py`: Custom learning rate scheduler with warm-up and polynomial decay.





---

### Weights & Documentation

* **`trained_models/erfnet_encoder_pretrained.pth.tar`**: Pretrained encoder weights for the baseline ERFNet architecture.


* **`eomt/docs/`**: Web assets, architecture diagrams, and documentation pages.
