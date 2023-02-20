# stockfish-diffusion
Easier version of fish-diffusion

[English Document](https://github.com/fishaudio/fish-diffusion/blob/main/README.en.md)

[English Colab](https://colab.research.google.com/drive/1MxbO25q2IpJ_ia0CjpMKRKABWUQhruhh)
```
conda create -n neosovit python=3.10
conda activate neosovit
```
### Step 1: Install requirements
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install pytorch-lightning

pip install -U openmim
mim install mmengine

pip install librosa
pip install loguru
pip install wandb
pip install pyloudnorm
pip install transformers
pip install torchcrepe
pip install praat-parselmouth
pip install pyworld
pip install ffmpeg
pip install pypinyin
```
Training
```
conda config --add channels conda-forge
conda install montreal-forced-aligner
pip install textgrid
pip install pykakasi
```

### Step 2: Download pretrained model
https://github.com/fishaudio/fish-diffusion/releases

(You are looking for the **ckpt** and **finetune.py**)

### Step 3: Inference
```
python .\inference.py --config "stockfish-diffusion\svc_cn_hubert_soft_finetune.py" --model "cn-hubert-soft-600-singers-pretrained-v1.ckpt" --input "stockfish-diffusion\test.wav" --output "out.wav"
```

### Step 4: Training
#### Step 4.1: Prepare dataset
- Create folders: `dataset\train` and `dataset\valid`
- Add audio files into above folders (All training data into `train`, 2 or 3 files into `valid`)
- Run
```
python .\extract_features.py --config "stockfish-diffusion\model\svc_cn_hubert_soft_finetune_crepe.py" --path "\dataset\train" --clean
```

#### Step 4.2: Train (Change batch_size to increase or decrease GPU memory usage)
```
python .\train.py --config "stockfish-diffusion\model\svc_cn_hubert_soft_finetune.py" --pretrained "stockfish-diffusion\cn-hubert-soft-600-singers-pretrained-v1.ckpt" --batch_size 12
````
