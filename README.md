# Sleeper Agent: Scalable Hidden Trigger Backdoors for Neural Networks Trained from Scratch

This code is the official PyTroch implementation of the [Sleeper Agent](https://arxiv.org/abs/2106.08970). Our implementation is based on [Industrial Scale Data Poisoning via Gradient Matching](https://github.com/JonasGeiping/poisoning-gradient-matching).

![Overview](schematic.png)


## Dependencies

- PyTorch => 1.6.*
- torchvision > 0.5.*
- higher [best to directly clone https://github.com/facebookresearch/higher and use ```pip install .```]
- python-lmdb [only if datasets are supposed to be written to an LMDB]




## USAGE

The wrapper for the Sleeper Agent can be found in sleeper_agent.py. To craft poisons on CIFAR-10 and ResNet18 with the default values, you can use the following command:

```shell
python sleeper_agent.py
```

There are a bunch of optional arguments in the ```forest/options.py```. Here are some of them:

- ```--patch_size```, ```--eps```, and ```--budget``` : determine the power of backdoor attack.
- ```--dataset``` : which dataset to poison.
- ```--net``` : which model to attack on.
- ```--retrain_scenario``` : enable the retraining during poison crafting.
- ```--poison_selection_strategy``` : enables the data selection (choose ```max_gradient```)
- ```--ensemble``` : number of models used to craft poisons.
- ```--sources``` : Number of sources to be triggered in inference time.

### ImageNet 

To craft poisons on ImageNet and ResNet18 you can use the following sample command:

```shell
python sleeper_agent.py --patch_size 30 --budget 0.0005 --pbatch 128 --epochs 80 --sources 50 --dataset ImageNet --pretrained_model --data_path /your/path/to/ImageNet --source_gradient_batch 300
```



## Citations


Please cite the Sleeper Agent paper:


```
@article{souri2021sleeper,
  title={Sleeper Agent: Scalable Hidden Trigger Backdoors for Neural Networks Trained from Scratch},
  author={Souri, Hossein and Goldblum, Micah and Fowl, Liam and Chellappa, Rama and Goldstein, Tom},
  journal={arXiv preprint arXiv:2106.08970},
  year={2021}
}
```
