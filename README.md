# MOCHA: Real-Time Motion Characterization via Context Matching

Official implementation of 'MOCHA: Real-Time Motion Characterization via Context Matching'.

Project Page: https://dk-jang.github.io/MOCHA_SIGASIA2023/

The code provides:
- [x] Data preparation
- [x] Pretrained model and Demo
- [ ] Trainging code for MOCHA

## Installation and Setup
First, clone the repo. Then, we recommend creating a clean conda environment, installing all dependencies, and finally activating the environment, as follows:
```bash
git clone https://github.com/DK-Jang/MOCHA_private.git
cd MOCHA_private
conda env create -f environment.yml
conda activate MOCHA
```

## Datasets and pre-trained networks
- To run the demo, please download the bvh, and pre-trained parameters both.

<b>[Recommend]</b> To download the datasets and the pretrained-networks, run the following commands:

```bash
bash download.sh datasets
bash download.sh pretrained-network
```

If you want to generate train datasets from the scratch, run the following commands:

```bash
python ./preprocess/generate_database_bin.py    # generate mirrored bvh files
```

## How to run the demo
After downloading the pre-trained parameterss, you can run the demo. \
We use the post-processed results (`Ours_*.bvh`) for all the demo.

To generate motion characterization results, run following commands:
```bash
python test_fullframework.py
```
Generated motions(bvh format) will be placed under `./results`. <br>
`Src_*.bvh`: ground-truth source motion, <br>
`Ours_*.bvh`: characterized output motion, <br>


## Acknowledgements
This repository contains pieces of code from the following repositories: \
[Learned Motion Matching](https://github.com/orangeduck/Motion-Matching). \
[Motion Puzzle: Arbitrary Motion Style Transfer by Body Part](https://github.com/DK-Jang/motion_puzzle).