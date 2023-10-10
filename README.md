# FN-SSL
A python implementation of <a href="https://arxiv.org/pdf/2305.19610.pdf" target="_blank"> "FN-SSL: Full-Band and Narrow-Band Fusion for Sound Source Localization" </a>, INTERSPEECH, 2023. 

+ **Contributions** 
  - Full-Band and Narrow-Band Fusion for moving sound source localization, the full-band layers focus on learning the inter-frequency dependencies of spatial/localization cues. The narrow-band layers focus on exploiting these narrowband inter-channel information. Meanwhile, FNSSL has good generalization ability on real datasets.

+ **Extension to microphone array with the number of microphones larger than two**: 
  - for DP-IPD regression:  You can refer to our <a href="https://github.com/BingYang-20/SRP-DNN" target="_blank">SRP-DNN</a>  which is a 12-mic array.
  - for DOA classification: Modifying the dimensions of LSTM input data can extension to microphone array with more microphones.

## Datasets
+ **Source signals**: from <a href="http://www.openslr.org/12/" target="_blank">LibriSpeech database</a> 
+ **Real-world multi-channel microphone signals**: from <a href="https://www.locata.lms.tf.fau.de/datasets/" target="_blank">LOCATA database</a> 
  
## Quick start
+ **Preparation** 

    * Download the required dataset and organize the data according to the data_org in the data folder.

    * Generate multi-channel data, You can set **data_num (in Simu.py)** to control the size of the dataset. --train, -- test, --dev are used to control the generation of train dataset, test dataset, and validation dataset, respectively. The source data path of them are specified by **dirs ['sousig_train ']** in Opt.py.
    ```
    python Simu.py --train/--test/--dev
    ```
    * For DP-IPD regression, set **is_doa = False** (Model.FN_SSL), and use mse loss function, for DOA classification, set **is_doa = True** (Model.FN_SSL), and use ce loss function,  meanwhile, the predgt2doa needs to be replaced synchronously. The initial Learning rate of doa classification is set to 5e-4.
    ```
    net = at_model.FN_SSL(is_doa=True/False)
    ```
+ **Training**
    * For train step, --gpu-id is used to specify the gpu, ---bz corresponds to the batch size of train process, validation process, and test process, respectively.
    ```
    python Train.py --train --gpu-id [*] --bz * * * 
    ```
+ **Evaluation** 

    * In the inference stage, you can set **checkpoints_dir** (Predict. py) to select weights, we provide simulation dataset inference and locata dataset inference.
    * For simulated data evaluation
    ```
    python Predict.py --test --datasetMode simulate --bz * * *
    ```
    * For LOCATA dataset evaluation
    ```
    python Predict.py --test --datasetMode locata
    ```
+ **Pytorch Lightning version**

    * We have re implemented FN-SSL using the Pytorch-lightning framework, which has a improvement in training speed compared to the torch.
    * For Train,

    ```
    python main.py fit --data.batch_size=[*,*] --trainer.devices=*,*
    ```
    * For test,

    ```
    python main.py test  --ckpt_path logs/MyModel/version_x/checkpoints/**.ckpt --trainer.devices=*,*
    ```

+ **Pretrained models**

    * Using the FN_lightning model to load the lightning checkpoint in torch framework.

| Framework | Task | Checkpoint |
| --- | --- | --- |
| Lightning | DP-IPD regression | https://pan.baidu.com/s/1zRKpiqbSuo80Xu5ZRoS1gQ?pwd=6w51 |
| Lightning | DOA classification | https://pan.baidu.com/s/1U1Wl5ZBZBItc2Vku7AyqNA?pwd=ceqm |

more checkpoints will be update soon.

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


