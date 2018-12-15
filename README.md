Tensorflow / keras re-implementation of the [nnPU Learning](https://arxiv.org/abs/1703.00593) paper. Including data loader for all 4 datasets mentioned in the paper.

This repo is originally a course project. This is NOT a official repo, so there might be mistakes / difference from the paper.

# Requirements
* tensorflow
* docopt
* Keras
* numpy
* gcc (if you need to use the epsilon dataset)

# Usage
```
Usage:
  train [--dataset=<dataset>] [--loss=<loss>] [--batch_size=<batch_size>]
        [--lr=<learning_rate>] [--pretrain=<pretrain>]

Options:
  --dataset=<dataset>        MNIST|epsilon|20News|CIFAR-10 [default: MNIST]
  --loss=<loss>              PN|uPU|nnPU [default: nnPU]
  --batch_size=<batch_size>  batch size [default: 30500]
  --lr=<learning_rate>       learning rate [default: 0.001]
  --pretrain=<pretrain>      pretrain|finetune|no [default: no]
  -h --help                  Show this screen.
```