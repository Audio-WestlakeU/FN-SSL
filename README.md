# FN-SSL
A python implementation of "FN-SSL: Full-Band and Narrow-Band Fusion for Sound Source Localization" (paper: https://arxiv.org/pdf/2305.19610.pdf), INTERSPEECH, 2023. (A simple version, more detailed introduction will be updated soon)

+ **Contributions** 
  - Full-Band and Narrow-Band Fusion for moving sound source localization, Following image is the Comparison results on the LOCATA dataset.
+ **Extension to microphone array with the number of microphones larger than two**: 
  - for DP-IPD regression:
  - for DOA classification:

## Datasets
+ **Source signals**: from <a href="http://www.openslr.org/12/" target="_blank">LibriSpeech database</a> 
+ **Real-world multi-channel microphone signals**: from <a href="https://www.locata.lms.tf.fau.de/datasets/" target="_blank">LOCATA database</a> 
  
## Quick start
+ **Preparation** 

Generate multi-channel data, You can set **data_num (in Simu.py)** to control the size of the dataset.
```
python Simu.py --train/--test/--dev
```
+ **Training**
```
python Train.py --train --gpu-id [*] --bz * * * 
```

+ **Evaluation** 
  
For simulated data evaluation
```
python Predict.py --test --datasetMode simulate --bz * * *
```
  
  For LOCATA dataset evaluation
 
```
python Predict.py --test --datasetMode locata
```
+ **Pytorch Lightning version**

We have re implemented FN-SSL using the Pytorch-lightning framework, which has a improvement in training speed compared to the torch.

+ **Pretrained models**


## Citation
If you find our work useful in your research, please consider citing:
```
@InProceedings{wang2023fnssl,
    author = "Yabo Wang and Bing Yang and Xiaofei Li",
    title = "FN-SSL: Full-Band and Narrow-Band Fusion for Sound Source Localization",
    booktitle = "Proceedings of INTERSPEECH",
    year = "2023",
    pages = ""}
```

## Reference code 
- <a href="https://github.com/BingYang-20/SRP-DNN" target="_blank">SRP-DNN</a> 
- <a href="https://github.com/DavidDiazGuerra/Cross3D" target="_blank">Cross3D</a> 

## Licence
MIT


