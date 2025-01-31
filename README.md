# Overview

This repository contains the code for the face classification and verification task.

## Setup

```bash
conda create -n face-cls-ver python=3.12.4
conda activate face-cls-ver
pip install -r requirements.txt
```

## Usage

```bash
python main.py --config config.yaml
```

## Config

The config file is used to configure the training and evaluation process.

## Data format

The data should be present in the following format:

```
root/
    cls_data/
        train/
            images/
            labels.txt
        val/
            images/
            labels.txt
        test/
            images/
            labels.txt
    ver_data/
        val_pairs.txt
        test_pairs.txt
```

In labels.txt, each line contains the image path and the corresponding label.
In val_pairs.txt and test_pairs.txt, each line contains two image paths and a label (1 (match) or 0 (non-match)).
The config file should be set to the correct paths for the data. For example, if the data is present in the `root` directory, the config file should be set as follows:

```yaml
cls_data_dir: root/cls_data
ver_data_dir: root/ver_data
val_pairs_file: root/val_pairs.txt
test_pairs_file: root/test_pairs.txt
```
