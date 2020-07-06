# Mask Separated feature learning for person Re-Identification

A simple but effective structure for person re-identification.


## Architecture

<image src="image/architecture.png" width="900">

## Requirements
We use the 'amp' package which supports automatic mixed precision. You can downloads the package in [here!](https://github.com/NVIDIA/apex)

After install amp, 
```
$ conda create -n msreid python==3.7 -y
$ conda install pytorch torchvision cudatoolkit -c pytorch
$ pip install -r requirements.txt
```


## Results (Rank-1/mAP)

|    Dataset    | Rank-1/mAP  |
| :-----------: | :---------: |
|  Market1501   | 95.8%/88.8% |
| DukeMTMC-reID | 90.2%/80.6% |
|   CUHK03-L    | 84.6%/81.9% |
|   CUHK03-D    | 82.7%/79.5% |

