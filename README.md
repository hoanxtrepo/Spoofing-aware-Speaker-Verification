# Spoofing-aware Speaker Verification 

<h1> 1. Setup </h1>

```bash
conda create --name venv python=3.8.10
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install -r requirements.txt
```

<h1> 2. Train & Test </h1>

**2.3. Training**

- [x] [MUSAN: A Music, Speech, and Noise Corpus](https://www.openslr.org/17/) (David et al., 2015)

- [x] [Room Impulse Response and Noise Database](https://www.openslr.org/28/) (Brecht et al., 2020)

NOTE: Replace the relative data path 

```console
## Train single subsystem: ASV or CM only
[**] python train.py  --save_path ...  --model {asv, cm} 

## Jointly-optimised training 
[**] python finetune.py --save_path exps/exp/ --asv_checkpoint ... --cm_checkpoint ...
```

**2.3. Inferring**

```console

[**] python {train.py | finetune.py} --check_point ... --eval_path ... --eval
```

