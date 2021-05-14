# Data Preparation
You can get more details from the [link](https://github.com/ruotianluo/self-critical.pytorch). 
## Prepare coco captions
Download preprocessed coco captions from [link](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) from Karpathy's homepage. Extract dataset_coco.json from the zip file and copy it in to data/.
This file provides preprocessed captions and also standard train-val-test splits. The do:
```
python scripts/prepro_labels.py --input_json data/dataset_coco.json --output_json data/cocotalk.json --output_h5 data/cocotalk
```
Then preprocess the dataset and get the cache for calculating cider score:
```
python scripts/prepro_ngrams.py --input_json data/dataset_coco.json --dict_json data/cocotalk.json --output_pkl data/coco-train --split train
```
Also, you can download the files processed be myself from this [link](https://drive.google.com/file/d/1pB4y6lIaprgTfwN59jQbZ7od9F0pR16H/view?usp=sharing). 
## Prepare Bottom-Up features
Download pre-extracted feature from (https://github.com/peteanderson80/bottom-up-attention). You can either download adaptive one or fixed one. We use the ''10 to 100 features per image (adaptive)''
```
python script/make_bu_data.py --output_dir data/cocobu
```
This will create data/cocobu_fc, data/cocobu_att and data/cocobu_box.
