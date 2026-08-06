# Mamba-FSCIL: Dynamic Adaptation with Selective State Space Model for Few-Shot Class-Incremental Learning

This is the official repository for "Mamba-FSCIL: Dynamic Adaptation with Selective State Space Model for Few-Shot Class-Incremental Learning," accepted by **IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2026**.

> **Mamba-FSCIL: Dynamic Adaptation with Selective State Space Model for Few-Shot Class-Incremental Learning**<br>
> [Xiaojie Li](https://xiaojieli0903.github.io/)^1,2, [Yibo Yang](https://yibo.yang93.github.io/)^4, [Jianlong Wu](https://wujianlong.hit.github.io)^1,3, [Yue Yu](https://yuyue.github.io/)^2, [Ming-Hsuan Yang](https://faculty.ucmerced.edu/mhyang/)^5, [Liqiang Nie](https://nieliqiang.github.io)^1,3, [Min Zhang](https://zhangmin2021.hit.github.io)^1,3<br>
> ^1Harbin Institute of Technology (Shenzhen), ^2Pengcheng Laboratory, ^3Shenzhen Loop Area Institute, ^4King Abdullah University of Science and Technology (KAUST), ^5University of California, Merced

![Mamba-FSCIL Framework](figs/framework.png)

## 🔨 Installation

Follow these steps to set up your environment:

- **Create and activate a new Conda environment**:
   ```shell
   conda create --name mambafscil python=3.10 -y
   conda activate mambafscil
   ```
- **Install CUDA and cuDNN**:
   Follow the [official CUDA installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

- **Install PyTorch and torchvision**:
   - Using pip:
     ```shell
     pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
     ```
   - Using conda:
     ```shell
     conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
     ```

- **Install MMCV, OpenCV, and other dependencies**:
   ```bash
   pip install -U openmim
   mim install mmcv-full==1.7.0
   pip install opencv-python matplotlib einops rope timm==0.6.12 scikit-learn==1.1.3 yapf==0.40.1
   git clone https://github.com/state-spaces/mamba.git; cd mamba; git checkout v1.2.0.post1; pip install .
   ```

- **Clone the repository and set up the directory**:
   ```shell
   git clone https://github.com/xiaojieli0903/Mamba-FSCIL.git
   cd Mamba-FSCIL; mkdir ./data
   ```

## ➡️ Data Preparation

- **Download datasets** from [this link](https://huggingface.co/datasets/HarborYuan/Few-Shot-Class-Incremental-Learning/blob/main/fscil.zip) provided by NC-FSCIL.

- **Organize the datasets** in the `./data` folder:
  ```shell
  --data
    ----cifar/
    ----CUB_200_2011/
    ----miniimagenet/
  ```

## 🚀 Training

Execute the provided scripts to start training.

### CIFAR

```commandline
sh train_cifar.sh
```

| Session  | 0     | 1     | 2     | 3     | 4     | 5     | 6     | 7     | 8     |
|----------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Mamba-FSCIL | 82.95±0.13 | 77.87±0.16 | 74.12±0.37 | 69.67±0.30 | 66.75±0.17 | 63.62±0.13 | 61.44±0.13 | 59.80±0.33 | 57.52±0.23 |

[[Base Log]](logs/cifar_base.log) [[Incremental Log]](logs/cifar_inc.log)

### Mini Imagenet

```commandline
sh train_miniimagenet.sh
```

| Session  | 0     | 1     | 2     | 3     | 4     | 5     | 6     | 7     | 8     |
|----------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Mamba-FSCIL | 84.54±0.15 | 79.23±0.14 | 74.68±0.90 | 71.42±0.60 | 68.85±0.31 | 65.89±0.29 | 62.77±0.10 | 61.01±0.32 | 59.29±0.15 |

[[Base Log]](logs/mini_base.log) [[Incremental Log]](logs/mini_inc.log)

### CUB

```commandline
sh train_cub.sh
```

| Session  | 0      | 1     | 2     | 3      | 4      | 5      | 6      | 7      | 8      | 9     | 10   |
|----------|--------|-------|-------|--------|--------|--------|--------|--------|--------|-------|-------|
| Mamba-FSCIL | 80.92±0.04 | 76.28±0.07 | 73.03±0.08 | 70.12±0.03 | 67.77±0.12 | 65.72±0.06 | 65.29±0.13 | 64.05±0.07 | 62.35±0.04 | 62.14±0.06 | 61.52±0.11 |

[[Base Log]](logs/cub_base.log) [[Incremental Log]](logs/cub_inc.log)

## ✏️ Citation

If you find our work useful in your research, please consider citing:

```bibtex
@article{li2026mamba,
  title={Mamba-FSCIL: Dynamic Adaptation with Selective State Space Model for Few-Shot Class-Incremental Learning},
  author={Li, Xiaojie and Yang, Yibo and Wu, Jianlong and Yu, Yue and Yang, Ming-Hsuan and Nie, Liqiang and Zhang, Min},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2026}
}
```

## 👍 Acknowledgments

This codebase builds on [FSCIL](https://github.com/NeuralCollapseApplications/FSCIL). Thank you to all the contributors.
