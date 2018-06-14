# Pytorch-Squeeznet
Implementation of Squeezenet in pytorch using CIFAR-10 Dataset , Accuracy of model=66% 

Pytorch implementation of Squeezenet model as described in https://arxiv.org/abs/1602.07360 on cifar-10 Data.

The definition of Squeezenet model is present model_squeeznet.py. The training procedure resides in the file main.py

Command to train the Squeezenet model on CIFAR 10 data is:
```python
python main.py --batch_size 32 --epoch 10
```

I am currently using SGD for training : learning rate and weight decay are currently updated using a 55 epoch learning rule, this usually gives good performance, but if you want to use something of your own, you can specify it by passing learning_rate and weight_decay parameter like so

```
python main.py --batch_size 32 --epoch 10 --learning_rate 1e-3 --epoch_55
```
