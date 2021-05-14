# Data Preparation
You can get more details from the [link](https://github.com/ruotianluo/self-critical.pytorch). 
## Prepare coco captions
Download preprocessed coco captions from [link](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) from Karpathy's homepage. Extract dataset_coco.json from the zip file and copy it in to data/.
This file provides preprocessed captions and also standard train-val-test splits. The do:
```
python scripts/prepro_labels.py --input_json data/dataset_coco.json --output_json data/cocotalk.json --output_h5 data/cocotalk
```
Also, you can download these files from the [link](https://drive.google.com/file/d/1nHBBjOkATSPYiFvenIEX_hEYhGOiwVRO/view?usp=sharing) 
