# Intro CNN Part 2

## Padding

- Useful when valuable pixels are in edges
- Zero padding when objects are in the center 
- Common kernels: 3x3, 7x7

Spatial Dropout - Dropout2D

D2d: will drop full features maps (channels)

```python
m = torch.nn.Dropout2d(p=0.5)
input = torch.randn(1,3,5,5)
output= m(input)
```

Input:

1 = trianing example (images)

3= channels

5= hight

5= width

BatchNorm 2d
$$

$$

```python
class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.cn1 = nn.Conv2d= (3, 192, kernel_size=5, stride=1, paddinf=2, bias= False)
    self.bn1 = nn.BatchNorm2d(192)
    
    pass
```

3 = input channels

192= output channels
$$
\gamma: weight
$$

$$
\beta : Bias
$$



## CNN in GPU

```python
if torch.cuda.is_available():
  torch.backends.cudnn.deterministic = True
```

 

Ensures reporducible results bu always using the same convolution algorithm in CuDNN

## Common Architectures 

### VGG -16

**Visual Geometric Group**

16 Layers

- 3x3 convolutions
- stride=1
- same padding
- 2x2 max pooling: reduce size of feature maps but increase channels
- stride = 2
- MLP

Common setup: Start with very large (h,w) then you make smaller but increasing channels

### ResNet 

**Residual Networks**

Skip connections: skip layers 
$$
a(l+2) = \sigma(z(l+2)+a(l))
$$


### Fully convolutional netoworks

**Network in network** 

The first scenario where a convolutional layer is the same as a fully connected layer is where:

**Kernel_size = input size:** This computation is the same as the dot product in a FCL

Second scenario

When re-arrange the input in number of channels 

**4 input or 2x2 image = 1x1 image with 4 channels**

## Inception

key ideas:

- 1x1 convolutions: a way to reduce the number of channels, ej:

```python
Conv2d = (in_channels=64, out_channels=32)
```

- Global average pooling: 
- Use of auxiliary losses that are added to the total loss
  - Separete losses in different layers. (Checkpoints in the nn) (help losses)
- New: Inception module

...  Inception missing parts !



## Transfer Learning 

Key ideas :

- Feature extraction layers may be generally useful
- Use a pre-trained model

What is fin tune on target task ??

Fin-tune means re-train

- Freeze the weights: Only train last layer (or last few layers)

- Fine-tunning, train a pre-trained network on your smaller dataset

  ​	re-train on small datasets



Last layer always replace !



Batch_size = training examples

have the same size images as the input

