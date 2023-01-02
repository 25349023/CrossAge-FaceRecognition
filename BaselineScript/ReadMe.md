## How to Run

- Train:
  ```
  python train.py --epochs 60 --lr 2e-3 --optimizer Adam --schedule-lr --schedule-epoch 10 --crop 128 112 --flip 0.5 --jitter --jitter-param 0.125 0.125 0.3 0
  ```
- Evaluate:
  ```
  python Evaluate.py --ckpt path/to/pth
  ```

## Log

updated at 08, Dec, 2022

---

## Goal

Evaluation scripts for **Cross-Age Face Recognition**, including data processing, evaluation matric, and so on.

---

## Requirements

* torch, torchvision

---

## More datails about the scripts

### Evaluation Matrix

* Compute Simlarity

### Model Factory

* Face feature extractor
    * use insightface and backbone is mobilefacenet
    * https://github.com/TreB1eN/InsightFace_Pytorch