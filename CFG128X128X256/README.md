🧠 TridentBrain V2 — Hybrid GNN + CNN + FFN Deep Learning Framework

CFG128X128X256 Build
Developed by Saunak Das Chaudhuri (BIT Mesra, 2025)

------------------------------------------------------------
Overview

TridentBrain is a unified deep learning framework that merges
Convolutional Neural Networks (CNNs), Feed-Forward Networks (FFNs),
and Graph Neural Networks (GNNs) — forming a tri-modal architecture
capable of performing:

- Classification
- Regression
- Object Detection
- Instance Segmentation

It introduces a Gaussian Process Head for probabilistic reasoning
and an Attention Fusion Layer that enables cross-modality learning.

------------------------------------------------------------
Core Architecture

[ CNN Encoder ]        [ FFN Encoder ]
       │                      │
       └──► [ Attention Fusion Patcher ] ◄──┘
                     │
                [ GNN Brain ]
                     │
            [ Gaussian / Task Head ]
                     │
                ───► Output

------------------------------------------------------------
Features

- Hybrid CNN + FFN + GNN architecture
- Multi-task: classification, regression, detection, segmentation
- Gaussian Process head for uncertainty
- Multi-head attention fusion
- Modular CI/CD-ready pipeline
- Dataset auto-downloading (MNIST, COCO-mini)

------------------------------------------------------------
Installation

git clone https://gitlab.com/<your-username>/CFG128X128X256.git
cd CFG128X128X256
pip install -e .

------------------------------------------------------------
Usage

python -m tridentbrain.train

Artifacts:
classification_model.pt
segmentation_model.pt
detection_model.pt

------------------------------------------------------------
Author

Saunak Das Chaudhuri
B.Tech, Electronics & Communication Engineering
Birla Institute of Technology, Mesra (BIT Mesra)

------------------------------------------------------------
License

MIT License
© 2025 Saunak Das Chaudhuri