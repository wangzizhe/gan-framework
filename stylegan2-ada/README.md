# StyleGAN2-ADA-PyTorch Pipeline

## Introduction

Pipeline for training on custom dataset and generating new images using StyleGAN2-ADA with PyTorch for unsupervised learning (without labels).

Based on the official GitHub-Repo of StyleGAN2-ADA (PyTorch Version), for more details of StyleGAN2-ADA please refer to the official GitHub-Repo:

[https://github.com/NVlabs/stylegan2-ada-pytorch](https:////github.com/NVlabs/stylegan2-ada-pytorch)

5 scripts in this repository are created by me for my [GAN framework](../README.md):

- [environment.yml](./environment.yml) 
  - for environment installation
- [optimize.py](./optimize.py) 
  - for automated hyperparameter tuning
- For data preparation
  - [make_csv.py](./make_csv.py)
    - for generating a .csv file of fake data
  - [data_preparation_classifier.py](./data_preparation_classifier.py) 
    - for data preparation of real and generated data for Classifier (replace real data with fake data)
  - [data_preparation_classifier_augmentation.py](./data_preparation_classifier_augmentation.py) 
    - for data preparation of real and generated data for Classifier (using fake data as data augmentation)

**Please note: This pipeline is to show you how to use StyleGAN2-ADA if you don't want to use the framework and only use this GAN technology**

The pipeline includes 4 steps:

* Environment Configuration

* Data Preparation

* Train The Network

* Generate Images using Trained Network

## Environment Configuration

**Requirements**

- Linux and Windows are supported, but we recommend Linux for performance and compatibility reasons.
- 1–8 high-end NVIDIA GPUs with at least 12 GB of memory.
- 64-bit Python 3.7 or 3.8 and PyTorch 1.7.1. See https://pytorch.org/ for PyTorch install instructions.
- CUDA toolkit 11.0 or later. Use at least version 11.1 if running on RTX 3090.
- Python libraries: `pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3`. We use the Anaconda3 2020.11 distribution which installs most of these by default.
- Docker users: use the provided Dockerfile to build an image with the required library dependencies.

**Check GPU Information**

```python
nvidia-smi
```

**Check PyTorch Version**

```python
import torch
torch.__verison__
```

**Update PyTorch Version If Needed**

```python
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

**Install Python libraries**

```python
pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3
```

## Data Preparation

- It is recommended that the datasets are stored as uncompressed ZIP archives containing uncompressed PNG files.
- Alternatively, the folder which contains images can also be used directly as a dataset, without running it through `dataset_tool.py` first, but doing so may lead to suboptimal performance.
- It is recommended to use squared image, the image resolution can be as high as 1024 x 1024 pixel. For images larger than 1024 x 1024 pixel it is not tested.

**Parameter**

- `source` the folder which stores your images
- `dest` directory of output file
- `transform` **only for images which are not squared:** crop the image to squared image
- `width` width of the cropped image
- `height` height of the cropped image

**Example**

```python
python dataset_tool.py --source=../Dataset/bone_marrow_slides/wsi --dest=../Dataset/bone_marrow_slides/wsi1024x1024.zip --transform=center-crop --width=1024 --height=1024
```

In this example your dataset is store in `../Dataset/bone_marrow_slides/wsi`, you want to save the output ZIP archives as `wsi1024x1024.zip` in the folder  `../Dataset/bone_marrow_slides`, and your images are not squared image (for example rectangular images), you want to crop the images to 1024 x 1024 pixel.

## Train The Network

**Parameter**

```python
# General options (not included in desc).
gpus     = 4, # Number of GPUs: <int>, default = 1 gpu
snap     = 50, # Snapshot interval: <int>, default = 50 ticks
metrics  = None, # List of metric names: [], ['fid50k_full'] (default), ...
seed     = None, # Random seed: <int>, default = 0
 
# Dataset.
data     = 'DATADIR', # Training dataset (required): <path>
cond     = None, # Train conditional model based on dataset labels: <bool>, default = False
subset   = None, # Train with only N images: <int>, default = all
mirror   = None, # Augment dataset with x-flips: <bool>, default = False
 
# Base config.
cfg      = None, # Base config: 'auto' (default), 'stylegan2', 'paper256', 'paper512', 'paper1024', 'cifar'
gamma    = None, # Override R1 gamma: <float>
kimg     = None, # Override training duration: <int>
batch    = None, # Override batch size: <int>
 
# Discriminator augmentation.
aug      = None, # Augmentation mode: 'ada' (default), 'noaug', 'fixed'
p        = None, # Specify p for 'fixed' (required): <float>
target   = None, # Override ADA target for 'ada': <float>, default = depends on aug
augpipe  = None, # Augmentation pipeline: 'blit', 'geom', 'color', 'filter', 'noise', 'cutout', 'bg', 'bgc' (default), ..., 'bgcfnc'
 
# Transfer learning.
resume   = ffhq1024, # Load previous network: 'noresume' (default), 'ffhq256', 'ffhq512', 'ffhq1024', 'celebahq256', 'lsundog256', <file>, <url>
freezed  = None, # Freeze-D: <int>, default = 0 discriminator layers
```

The training configuration can be further customized with additional command line options:

- `aug=noaug` disables ADA.
- `cond=1` enables class-conditional training (requires a dataset with labels).
- `mirror=1` amplifies the dataset with x-flips. Often beneficial, even with ADA.
- `resume=ffhq1024` performs transfer learning from FFHQ trained at 1024x1024.
- `resume=~/training-runs/<NAME>/network-snapshot-<INT>.pkl` resumes a previous training run.
- `gamma=10` overrides R1 gamma. We recommend trying a couple of different values for each new dataset.
- `aug=ada --target=0.7` adjusts ADA target value (default: 0.6).
- `augpipe=blit` enables pixel blitting but disables all other augmentations.
- `augpipe=bgcfnc` enables all available augmentations (blit, geom, color, filter, noise, cutout).

**Example**

In its most basic form, training new networks boils down to:

```python
python train.py --snap=50 --data=../data/wsi_scale_removed_non_npm1/wsi_scale_removed_non_npm1_1024x1024.zip --outdir=./outdir --resume='ffhq1024' --gpus=4
```

In this example, the training performs transfer learning from FFHQ trained at 1024x1024, the results are saved to a newly created directory ~`/training-runs/<ID>-mydataset-auto1`, controlled by `--outdir`. The training exports network pickles (`network-snapshot-<INT>.pkl`) and example images (`fakes<INT>.png`) at regular intervals (`controlled by --snap`). For each pickle, it also evaluates FID (`controlled by --metrics`) and logs the resulting scores in `metric-fid50k_full.jsonl` (as well as TFEvents if TensorBoard is installed).

The name of the output directory reflects the training configuration. For example, `00000-mydataset-auto1` indicates that the *base configuration* was `auto1`, meaning that the hyperparameters were selected automatically for training on one GPU. The base configuration is controlled by `--cfg`:

You can also continue your training process to set the `resume` as one of your `.pkl` file.

**During training**

After starting training process, you'll see the following information includes

* Information of your dataset and your setting
* Network architecture of Generator and Discriminator
* Training progress

```
Loading training set...

Num images:  1536
Image shape: [3, 1024, 1024]
Label shape: [0]

Constructing networks...
Setting up PyTorch plugin "bias_act_plugin"... Done.
Setting up PyTorch plugin "upfirdn2d_plugin"... Done.

Generator              Parameters  Buffers  Output shape         Datatype
---                    ---         ---      ---                  ---     
mapping.fc0            262656      -        [4, 512]             float32 
mapping.fc1            262656      -        [4, 512]             float32 
mapping                -           512      [4, 18, 512]         float32 
synthesis.b4.conv1     2622465     32       [4, 512, 4, 4]       float32 
synthesis.b4.torgb     264195      -        [4, 3, 4, 4]         float32 
synthesis.b4:0         8192        16       [4, 512, 4, 4]       float32 
synthesis.b4:1         -           -        [4, 512, 4, 4]       float32 
synthesis.b8.conv0     2622465     80       [4, 512, 8, 8]       float32 
synthesis.b8.conv1     2622465     80       [4, 512, 8, 8]       float32 
synthesis.b8.torgb     264195      -        [4, 3, 8, 8]         float32 
synthesis.b8:0         -           16       [4, 512, 8, 8]       float32 
synthesis.b8:1         -           -        [4, 512, 8, 8]       float32 
synthesis.b16.conv0    2622465     272      [4, 512, 16, 16]     float32 
synthesis.b16.conv1    2622465     272      [4, 512, 16, 16]     float32 
synthesis.b16.torgb    264195      -        [4, 3, 16, 16]       float32 
synthesis.b16:0        -           16       [4, 512, 16, 16]     float32 
synthesis.b16:1        -           -        [4, 512, 16, 16]     float32 
synthesis.b32.conv0    2622465     1040     [4, 512, 32, 32]     float32 
synthesis.b32.conv1    2622465     1040     [4, 512, 32, 32]     float32 
synthesis.b32.torgb    264195      -        [4, 3, 32, 32]       float32 
synthesis.b32:0        -           16       [4, 512, 32, 32]     float32 
synthesis.b32:1        -           -        [4, 512, 32, 32]     float32 
synthesis.b64.conv0    2622465     4112     [4, 512, 64, 64]     float32 
synthesis.b64.conv1    2622465     4112     [4, 512, 64, 64]     float32 
synthesis.b64.torgb    264195      -        [4, 3, 64, 64]       float32 
synthesis.b64:0        -           16       [4, 512, 64, 64]     float32 
synthesis.b64:1        -           -        [4, 512, 64, 64]     float32 
synthesis.b128.conv0   1442561     16400    [4, 256, 128, 128]   float16 
synthesis.b128.conv1   721409      16400    [4, 256, 128, 128]   float16 
synthesis.b128.torgb   132099      -        [4, 3, 128, 128]     float16 
synthesis.b128:0       -           16       [4, 256, 128, 128]   float16 
synthesis.b128:1       -           -        [4, 256, 128, 128]   float32 
synthesis.b256.conv0   426369      65552    [4, 128, 256, 256]   float16 
synthesis.b256.conv1   213249      65552    [4, 128, 256, 256]   float16 
synthesis.b256.torgb   66051       -        [4, 3, 256, 256]     float16 
synthesis.b256:0       -           16       [4, 128, 256, 256]   float16 
synthesis.b256:1       -           -        [4, 128, 256, 256]   float32 
synthesis.b512.conv0   139457      262160   [4, 64, 512, 512]    float16 
synthesis.b512.conv1   69761       262160   [4, 64, 512, 512]    float16 
synthesis.b512.torgb   33027       -        [4, 3, 512, 512]     float16 
synthesis.b512:0       -           16       [4, 64, 512, 512]    float16 
synthesis.b512:1       -           -        [4, 64, 512, 512]    float32 
synthesis.b1024.conv0  51297       1048592  [4, 32, 1024, 1024]  float16 
synthesis.b1024.conv1  25665       1048592  [4, 32, 1024, 1024]  float16 
synthesis.b1024.torgb  16515       -        [4, 3, 1024, 1024]   float16 
synthesis.b1024:0      -           16       [4, 32, 1024, 1024]  float16 
synthesis.b1024:1      -           -        [4, 32, 1024, 1024]  float32 
---                    ---         ---      ---                  ---     
Total                  28794124    2797104  -                    -       


Discriminator  Parameters  Buffers  Output shape         Datatype
---            ---         ---      ---                  ---     
b1024.fromrgb  128         16       [4, 32, 1024, 1024]  float16 
b1024.skip     2048        16       [4, 64, 512, 512]    float16 
b1024.conv0    9248        16       [4, 32, 1024, 1024]  float16 
b1024.conv1    18496       16       [4, 64, 512, 512]    float16 
b1024          -           16       [4, 64, 512, 512]    float16 
b512.skip      8192        16       [4, 128, 256, 256]   float16 
b512.conv0     36928       16       [4, 64, 512, 512]    float16 
b512.conv1     73856       16       [4, 128, 256, 256]   float16 
b512           -           16       [4, 128, 256, 256]   float16 
b256.skip      32768       16       [4, 256, 128, 128]   float16 
b256.conv0     147584      16       [4, 128, 256, 256]   float16 
b256.conv1     295168      16       [4, 256, 128, 128]   float16 
b256           -           16       [4, 256, 128, 128]   float16 
b128.skip      131072      16       [4, 512, 64, 64]     float16 
b128.conv0     590080      16       [4, 256, 128, 128]   float16 
b128.conv1     1180160     16       [4, 512, 64, 64]     float16 
b128           -           16       [4, 512, 64, 64]     float16 
b64.skip       262144      16       [4, 512, 32, 32]     float32 
b64.conv0      2359808     16       [4, 512, 64, 64]     float32 
b64.conv1      2359808     16       [4, 512, 32, 32]     float32 
b64            -           16       [4, 512, 32, 32]     float32 
b32.skip       262144      16       [4, 512, 16, 16]     float32 
b32.conv0      2359808     16       [4, 512, 32, 32]     float32 
b32.conv1      2359808     16       [4, 512, 16, 16]     float32 
b32            -           16       [4, 512, 16, 16]     float32 
b16.skip       262144      16       [4, 512, 8, 8]       float32 
b16.conv0      2359808     16       [4, 512, 16, 16]     float32 
b16.conv1      2359808     16       [4, 512, 8, 8]       float32 
b16            -           16       [4, 512, 8, 8]       float32 
b8.skip        262144      16       [4, 512, 4, 4]       float32 
b8.conv0       2359808     16       [4, 512, 8, 8]       float32 
b8.conv1       2359808     16       [4, 512, 4, 4]       float32 
b8             -           16       [4, 512, 4, 4]       float32 
b4.mbstd       -           -        [4, 513, 4, 4]       float32 
b4.conv        2364416     16       [4, 512, 4, 4]       float32 
b4.fc          4194816     -        [4, 512]             float32 
b4.out         513         -        [4, 1]               float32 
---            ---         ---      ---                  ---     
Total          29012513    544      -                    -       

Setting up augmentation...
Distributing across 1 GPUs...
Setting up training phases...
Exporting sample images...
Initializing logs...
Training for 25000 kimg...

tick 0     kimg 0.0      time 1m 26s       sec/tick 7.3     sec/kimg 1828.68 maintenance 78.3   cpumem 5.42   gpumem 11.32  augment 0.000
Evaluating metrics...
{"results": {"fid50k_full": 447.91339439137664}, "metric": "fid50k_full", "total_time": 1087.0433948040009, "total_time_str": "18m 07s", "num_gpus": 1, "snapshot_pkl": "network-snapshot-000000.pkl", "timestamp": 1624959292.3564472}
tick 1     kimg 4.0      time 29m 25s      sec/tick 579.4   sec/kimg 144.85  maintenance 1099.8 cpumem 4.52   gpumem 7.49   augment 0.006
tick 2     kimg 8.0      time 39m 05s      sec/tick 580.4   sec/kimg 145.10  maintenance 0.1    cpumem 4.37   gpumem 7.49   augment 0.013
tick 3     kimg 12.0     time 48m 47s      sec/tick 581.5   sec/kimg 145.38  maintenance 0.1    cpumem 4.37   gpumem 7.52   augment 0.020
tick 4     kimg 16.0     time 58m 29s      sec/tick 581.7   sec/kimg 145.43  maintenance 0.1    cpumem 4.37   gpumem 7.52   augment 0.025
tick 5     kimg 20.0     time 1h 08m 11s   sec/tick 581.7   sec/kimg 145.43  maintenance 0.1    cpumem 4.36   gpumem 7.72   augment 0.031
```

Based on your settings, you will get `.pkl` files after some ticks, this is controlled by the parameter `snap`. For example, if you set your `snap` as 50, then you will get your first `.pkl` file after 50 ticks. The `.pkl` file stores the weights of the trained network.

During training, there will be one big image which includes many fake images generated after every `snap` as your setting. If you are satisfied with the results you can stop the training at any time.

**Expected training time**

According to the author's tests, the total training time depends heavily on resolution, number of GPUs, dataset, desired quality, and hyperparameters. The following table lists expected wallclock times to reach different points in the training, measured in thousands of real images shown to the discriminator ("kimg"):

| Resolution | GPUs | 1000 kimg | 25000 kimg |  sec/kimg   | GPU mem | CPU mem |
| :--------: | :--: | :-------: | :--------: | :---------: | :-----: | :-----: |
|  128x128   |  1   |  4h 05m   |   4d 06h   |  12.8–13.7  | 7.2 GB  | 3.9 GB  |
|  128x128   |  2   |  2h 06m   |   2d 04h   |   6.5–6.8   | 7.4 GB  | 7.9 GB  |
|  128x128   |  4   |  1h 20m   |   1d 09h   |   4.1–4.6   | 4.2 GB  | 16.3 GB |
|  128x128   |  8   |  1h 13m   |   1d 06h   |   3.9–4.9   | 2.6 GB  | 31.9 GB |
|  256x256   |  1   |  6h 36m   |   6d 21h   |  21.6–24.2  | 5.0 GB  | 4.5 GB  |
|  256x256   |  2   |  3h 27m   |   3d 14h   |  11.2–11.8  | 5.2 GB  | 9.0 GB  |
|  256x256   |  4   |  1h 45m   |   1d 20h   |   5.6–5.9   | 5.2 GB  | 17.8 GB |
|  256x256   |  8   |  1h 24m   |   1d 11h   |   4.4–5.5   | 3.2 GB  | 34.7 GB |
|  512x512   |  1   |  21h 03m  |  21d 22h   |  72.5–74.9  | 7.6 GB  | 5.0 GB  |
|  512x512   |  2   |  10h 59m  |  11d 10h   |  37.7–40.0  | 7.8 GB  | 9.8 GB  |
|  512x512   |  4   |  5h 29m   |   5d 17h   |  18.7–19.1  | 7.9 GB  | 17.7 GB |
|  512x512   |  8   |  2h 48m   |   2d 22h   |   9.5–9.7   | 7.8 GB  | 38.2 GB |
| 1024x1024  |  1   |  1d 20h   |  46d 03h   | 154.3–161.6 | 8.1 GB  | 5.3 GB  |
| 1024x1024  |  2   |  23h 09m  |  24d 02h   |  80.6–86.2  | 8.6 GB  | 11.9 GB |
| 1024x1024  |  4   |  11h 36m  |  12d 02h   |  40.1–40.8  | 8.4 GB  | 21.9 GB |
| 1024x1024  |  8   |  5h 54m   |   6d 03h   |  20.2–20.6  | 8.3 GB  | 44.7 GB |

The above measurements were done using NVIDIA Tesla V100 GPUs with default settings (`--cfg=auto --aug=ada --metrics=fid50k_full`). "sec/kimg" shows the expected range of variation in raw training performance, as reported in `log.txt`. "GPU mem" and "CPU mem" show the highest observed memory consumption, excluding the peak at the beginning caused by `torch.backends.cudnn.benchmark`.

In typical cases, 25000 kimg or more is needed to reach convergence, but the results are already quite reasonable around 5000 kimg. 1000 kimg is often enough for transfer learning, which tends to converge significantly faster. The following figure shows example convergence curves for different datasets as a function of wallclock time, using the same settings as above:

## Generate Images using Trained Network

Now you have trained your network for many epochs, you can use the `.pkl` files with `generate.py` to generate new images

**Parameter**

- `outdir` output directory
- `trunc` truncation, lower truncation = higher fidelity but lower diversity, higher truncation = lower fidelity but higher diversity
- `seeds` list of random seeds
- `network` network pickle filename

**Example**

```python
python generate.py --outdir=./outdir/images --trunc=0.7 --seeds=600-605 --network='./outdir/00001-bone_marrow_slides-auto1-resumecustom/network-snapshot-000400.pkl'
```

In this example, you want to generate images to the folder `./outdir/images`, you want to get higher fidelity but still keep its diversity, so you set the `trunc` as 0.7, you selected your random `seeds` as 600, 601, 602, 603, 604, 605 (seeds can also be typed in through this way), you used the trained network`network-snapshot-000400.pkl` which stored in `./outdir/00001-bone_marrow_slides-auto1-resumecustom`



Now it is the time to make others confused!
