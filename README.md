# Deep Learning Models

The models are implemented inside "models" folder. The hyperparameters and experiment setup is under the configs folder.
Experiments:

## The Simple deep average network (DAN)
```shell
python train.py --config configs/dan.yml
```

## Transfer learning with DAN
### Source domain:  DAN running RoBERTa labeled dataset (near 20000 records)
We use the pretrained roBERTa model to label the dataset with nearly 20000 tweets on our topic, then we train DAN on this large
dataset,  to let our DAN model learn the generic sentiment analysis task.
```shell
python train.py --config configs/pretrain.yml
```

```shell
python train.py --config configs/transfer.yml
```

## Our Novelty: roBERTa-DAN

```shell
python train.py --config configs/transfer.yml
```

