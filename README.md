# SMSNet

The corresponding paper for this project is **SMSNet: An Efficient Network for Small Object Detection in UAV Aerial Images**.

This repository contains the full project code for SMSNet, implemented on top of the Ultralytics framework.

## Installation

```bash
conda create -n smsnet python=3.11
conda activate smsnet
pip install -e .
```

Install PyTorch according to your CUDA environment before training or evaluation.

## Train

```bash
yolo detect train model=ultralytics/cfg/models/smsnet/smsnet.yaml data=your_dataset.yaml imgsz=640
```

## Val

```bash
yolo detect val model=your_weight.pt data=your_dataset.yaml imgsz=640
```

## Notes

- The SMSNet model configuration is located at `ultralytics/cfg/models/smsnet/smsnet.yaml`.
- The core SMSNet modules are implemented in `ultralytics/nn/modules/smsnet.py`.

## License

This project follows the license terms provided in the included `LICENSE` file.
