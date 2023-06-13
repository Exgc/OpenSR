# LRS2 and LRS2-COMMON Data Preprocessing

This folder contains scripts for data preparation for LRS2 and LRS2-COMMON datasets, as well as audio noise preparation (for noisy environment simulation).

## Installation
To preprocess, you need some additional packages:
```
pip install -r requirements.txt
```

## LRS2 Preprocessing

Assume the data directory is `${lrs2}`, which contains three folders (`pretrain,trainval,test`). Follow the steps below:

### 1. Data preparation
```sh
python lrs2_prepare.py --lrs2 ${lrs2} --ffmpeg /path/to/ffmpeg --rank ${rank} --nshard ${nshard} --step ${step}
```
This will generate a list of file-ids (`${lrs2}/file.list`) and corresponding text labels (`${lrs2}/label.list`). Specifically, it includes 4 steps, where `${step}` ranges from `1,2,3,4`. Step 1, split long utterances in lrs2 `pretraining` into shorter utterances, generate their time boundaries and labels. Step 2, trim videos and audios according to the new time boundary. Step 3, extracting audio for trainval and test split. Step 4, generate a list of file ids and corresponding text transcriptions.  `${nshard}` and `${rank}` are only used in step 2 and 3. This would shard all videos into `${nshard}` and processes `${rank}`-th shard, where rank is an integer in `[0,nshard-1]`. 


### 2. Detect facial landmark and crop mouth ROIs:
```sh
python detect_landmark.py --root ${lrs2} --landmark ${lrs2}/landmark --manifest ${lrs2}/file.list \
 --cnn_detector /path/to/dlib_cnn_detector --face_detector /path/to/dlib_landmark_predictor --ffmpeg /path/to/ffmpeg \
 --rank ${rank} --nshard ${nshard}
```
```sh
python align_mouth.py --video-direc ${lrs2} --landmark ${landmark_dir} --filename-path ${lrs2}/file.list \
 --save-direc ${lrs2}/video --mean-face /path/to/mean_face --ffmpeg /path/to/ffmpeg \
 --rank ${rank} --nshard ${nshard}
```

This generates mouth ROIs in `${lrs2}/video`. It shards all videos in `${lrs2}/file.list` into `${nshard}` and generate mouth ROI for `${rank}`-th shard , where rank is an integer in `[0,nshard-1]`. The face detection and landmark prediction are done using [dlib](https://github.com/davisking/dlib). The links to download `cnn_detector`, `face_detector`, `mean_face` can be found in the help message

### 3. Count number of frames per clip
```sh
python count_frames.py --root ${lrs2} --manifest ${lrs2}/file.list --nshard ${nshard} --rank ${rank}
```
This counts number of audio/video frames for `${rank}`-th shard and saves them in `${lrs2}/nframes.audio.${rank}` and `${lrs2}/nframes.video.${rank}` respectively. Merge shards by running:

```
for rank in $(seq 0 $((nshard - 1)));do cat ${lrs2}/nframes.audio.${rank}; done > ${lrs2}/nframes.audio
for rank in $(seq 0 $((nshard - 1)));do cat ${lrs2}/nframes.video.${rank}; done > ${lrs2}/nframes.video
```

### 4. Partitioning LRS2 according to TF（term-frequency）
```sh
python lrs2_partioning.py --lrs2 ${lrs2}
```

This generate the LRS2-COMMON file lists `${lrs2}/train_{10, 20, 50, 100}.txt`.
### 5. Set up data directory
```sh
python lrs2_manifest.py --lrs2 ${lrs2} --manifest ${lrs2}/file.list \
 --valid-ids /path/to/valid --vocab-size ${vocab_size}
```

This sets up data directory of trainval-only (~30h training data) and pretrain+trainval (~433h training data). It will first make a tokenizer based on sentencepiece model and set up target directory containing `${train|valid|test}.{tsv|wrd}`. `*.tsv` are manifest files and `*.wrd` are text labels.  `/path/to/valid` contains held-out clip ids used as validation set.

## Audio Noise Preparation (Optional)
If you want to test your model under noisy setting, you should prepare audio noise data. First download and decompress the [MUSAN](https://www.openslr.org/17/) corpus. Assume the data directory is `${musan}`, which contains the following folders `{music,speech,noise}`.

### 1. MUSAN data preparation
```sh
python musan_prepare.py --musan ${musan} --nshard ${nshard}  --slurm_partition ${slurm_partition}
```
This will: (1) split raw audios into 10-second clips, (2) generate babble noise from MUSAN speech audio, (3) count number of frames per clip. The whole data will be sharded into `${nshard}` parts and each job processes one part. It runs on Slurm and has dependency on [submitit](https://github.com/facebookincubator/submitit)


### 2. lrs2 audio noise preparation
```sh
python lrs2_noise.py --lrs2 ${lrs2}
```
It will generate lrs2 babble and speech noise including their manifest files, which are stored in `${lrs2}/noise/{babble,speech}`. `${lrs2}` is the lrs2 data directory. Make sure you already finished setting up lrs2 before running the command.

The following command generates babble noise from lrs2 training set.
```sh
python mix_babble.py --lrs2 ${lrs2}
```


### 3. Set up noise directory
```sh
python noise_manifest.py --lrs2 ${lrs2}  --musan ${musan}
```
It will make manifest (tsv) files for MUSAN babble, music and noise in `${musan}/tsv/{babble,music,noise}`, as well as a combined manifest in `${musan}/tsv/all` including MUSAN babble, music, noise and lrs2 speech. 
