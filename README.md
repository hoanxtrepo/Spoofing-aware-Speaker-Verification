# Speaker Verification - PyTorch Implementation

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

NOTE: tải dữ liệu từ 2 link ở trên và thay thế vào argument tại `musan_path` and `rir_path`

```console
## train asv hoặc cm model
[**] python train.py \
    --save_path ... \
    --model {asv, cm} 

## train cho mô hình joint train
[**] python finetune.py \
    --save_path exps/joint_trained_exp/ \
    --asv_checkpoint ...\
    --cm_checkpoint ...
```

**2.3. Testing**

```console

[**] python {train.py | finetune.py} --initial_model ... --eval_list ... --eval
```

