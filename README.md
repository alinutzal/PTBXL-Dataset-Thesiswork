# PTB-XL ECG Classification Models

This repository contains multiple deep learning models developed and tested on the [PTB-XL ECG dataset](https://physionet.org/content/ptb-xl/1.0.1/). The models use convolutional neural networks (CNNs), transformer architectures, and basic neural networks to detect and classify ECG abnormalities.

## Notebook Descriptions

| Notebook                  | Description |
|---------------------------|-------------|
| `PTBXL_CNN.ipynb`         | A convolutional neural network (CNN) implementation trained on 500 Hz ECG waveforms to classify multiple diagnostic labels. |
| `PTBXL_CNN_WandB.ipynb`   | An enhanced CNN model that integrates Weights & Biases (WandB) logging for experiment tracking and performance visualization. |
| `PTBXL NN.ipynb`          | A simple, fully-connected neural network (NN) for baseline performance comparison on the PTB-XL dataset. |
| `PTBXL Transformer.ipynb` | A transformer-based model that applies self-attention to ECG time-series, supporting multi-label classification. |
| `TransformerOSC.ipynb`    | A full pipeline transformer model that runs in the OSC environment. This notebook includes dataset downloading, preprocessing, model training, and evaluation — all in one place. |

## Models & Features

- **Multi-label classification** using SCP diagnostic codes
- **Transformer architecture** with adaptive pooling
- **CNN architecture** with 1D convolutions
- **NN baseline** for benchmarking
- **Training & validation split**
- **ROC curves, AUC scores**, and **classification reports**
- **Support for OSC (Ohio Supercomputer Center)** environment setup

## Getting Started

1. Clone this repository:
   ```bash
   git clone https://github.com/NicholasKanos/ptbxl-ecg-models.git
   cd ptbxl-ecg-models
   ```

2. Create or activate a Python environment (e.g., via `conda`):
   ```bash
   conda create -n ptbxl-env python=3.10
   conda activate ptbxl-env
   pip install -r requirements.txt
   ```

3. Run any notebook in JupyterLab or Jupyter Notebook.

## Requirements

Install packages via:
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install torch torchvision wfdb pandas numpy scikit-learn matplotlib tqdm
```

## Dataset

- PTB-XL: A large publicly available ECG dataset (10-second 12-lead recordings)
- Automatically downloaded and extracted in `TransformerOSC.ipynb`

## License

This project is licensed under the MIT License.

---

## Author

Nicholas Kanos  
Master of Computing and Information Systems, Youngstown State University  
Research focus: ECG signal classification, Transformer models, ML for healthcare
