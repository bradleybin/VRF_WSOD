## A Visual Representation-guided Framework with Global Affinity for Weakly Supervised Salient Object Detection

This code was implemented with Python 3.6, PyTorch 1.8.0, and CUDA 11.1 on an an NVIDIA GeForce GTX 3090.

## Usage
### 1. Download datasets
Download the DUTS and other datasets and unzip them into VRF_WSOD/data folder. 
Scribble labels can be downloaed from [Scribble Saliency](https://github.com/JingZhang617/Scribble_Saliency).

### 2. General visual representation generation
Clone the Tokencut project
```bash
git clone https://github.com/YangtaoWANG95/TokenCut
```
Enter the directory
```bash
cd Tokencut
```
Install dino_deitsmall8_pretrain from https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth.
Then, copy ***main_token_similarity_save.py*** into this directory.

General visual representation generation
```bash
python main_token_similarity_save.py --dataset 'DUTS' --mode 'train'
```

The general visual representations of the test dataset is also generated in this way, such as the DUTS-TE dataset.
```bash
python main_token_similarity_save.py --dataset 'DUTS-TE' --mode 'test'
```

### 3. Train
```bash
cd VRF
```
Download the pretrianed weight for backbone [Res50](https://drive.google.com/file/d/1arzcXccUPW1QpvBrAaaBv1CapviBQAJL/view).


```bash
#Train
python train_token_similarity_40_40_384_mul_dino.py
```

### 4. Test
Pretrained models can be downloaded from [Google Drive](https://drive.google.com/file/d/1nj-ryvvsW1zlc0kFqlVsA2j-lp9wIiBQ/view?usp=share_link).
!!!The general visual representations of the test dataset shoud be generated before starting testing. (refer to Part 2)

```bash
#Prediction
python test_dino.py --model_path path/to/pretrained_model/folder/
```
 
We provide the pre-computed saliency maps [Google Drive](https://drive.google.com/file/d/1B0Y--vy4nhVlXltfn6RSxmn8lI6dcGoD/view?usp=share_link).


Thanks to [Tokencut](https://github.com/YangtaoWANG95/TokenCut) and [SCWSSOD](https://github.com/siyueyu/SCWSSOD).

