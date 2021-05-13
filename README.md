# Vocabulary-critic sequence training for image captioning
This repository includes the implementation for [Vocabulary-Wide Credit Assignment for Training Image Captioning Models](https://ieeexplore.ieee.org/abstract/document/9329055). 
<br>
## Requirements 
    *python 3.6
    *pytorch 1.2
    *cider
    *coco-caption
## Training 
### Prepare Data
See details in data/README.md
### Start Training
```
bash train.sh
```
See opts.py for the options. (You can download the pretrained models from [here](https://drive.google.com/file/d/19rZ23UrEayb-ccreAoDTQ_Pksjap0h9O/view?usp=sharing).)
### Evaluation
```
bash test.sh
```
## Reference
If you find this repo helpful, please consider citing:
@article{liu2021vocabulary,
  title={Vocabulary-Wide Credit Assignment for Training Image Captioning Models},
  author={Liu, Han and Zhang, Shifeng and Lin, Ke and Wen, Jing and Li, Jianmin and Hu, Xiaolin},
  journal={IEEE Transactions on Image Processing},
  volume={30},
  pages={2450--2460},
  year={2021},
  publisher={IEEE}
}
