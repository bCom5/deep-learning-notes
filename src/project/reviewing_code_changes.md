# Reviewing Code Changes

## Jesus commited doc on quantization methods and additional references

change `quantization/QuantizationExamples.ipynb`

### Range-Based Linear Quantization

![](https://i.imgur.com/aPlYiVo.png)

* Asymmetric
* min max both float input and quantized type.
* Zero point introduced (bias)?
    * `k`: k-bit precision, `x_f`: float-point, `x_q`: quantized value, `zp`: zero point bias
    * (what is the bias), starting point of the value?
    * like bias in floating point

![](https://i.imgur.com/uzRaYzS.png)

![](https://i.imgur.com/U6sFRxV.png)

![](https://i.imgur.com/dhmhom3.png)

* Symmetric
* our range is positive and negative max abs value

![](https://i.imgur.com/aXadOdb.png)

* is this sign magnitute?

### DoReFa

![](https://i.imgur.com/hEQuzdK.png)

* `quantize_k()` with a real value then makes it k bit discrete, 2^k options.
* round((2^k - 1)x_f)
* requires quantization aware training

### PACT Parameterized Clipping Activation for Quantized Neural Networks
* clipping values are learned

### WRPN: Wide Reduced-Precision Networks
* activations clipped weights clipped

### PyTorch Quantized Tensors
* round(x/s+b)
* x: floating point, s: num bit precision


`quantization/q_funcs.py`
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def uniform_quantize(k):
  class qfn(torch.autograd.Function): 

    @staticmethod
    def forward(ctx, input):
      if k == 32:
        out = input # @BL full precision
      elif k == 1:
        out = torch.sign(input) # @BL single bit
      else:
        n = float(2 ** k - 1)
        out = torch.round(input * n) / n # @Brian 
      return out

      # @BL 5.2 quantized to 2 bits
      # 2 ** 2 - 1 = 3
      # 3 * 5.2 = 15.6
      # round(15.6) = 16
      # 16 / 3 = 5.33

    @staticmethod
    def backward(ctx, grad_output):
      grad_input = grad_output.clone() # @BL just use same gradient?
      return grad_input

  return qfn().apply # @BL it's a functino but what why apply?


class weight_quantize_fn(nn.Module):
  def __init__(self, w_bit):
    super(weight_quantize_fn, self).__init__()
    assert w_bit <= 8 or w_bit == 32 # @Brian has to be less than 8 bits
    self.w_bit = w_bit
    self.uniform_q = uniform_quantize(k=w_bit) # @Brian uses uniform quantize

  def forward(self, x):
    if self.w_bit == 32:
      weight_q = x
    elif self.w_bit == 1:
      E = torch.mean(torch.abs(x)).detach() # @BL detach()?
      weight_q = self.uniform_q(x / E) * E
      #print(weight_q)
    else:
      weight = torch.tanh(x)
      max_w = torch.max(torch.abs(weight)).detach()
      weight = weight / 2 / max_w + 0.5
      weight_q = max_w * (2 * self.uniform_q(weight) - 1)
      # @BL DoReFa

    return weight_q


class activation_quantize_fn(nn.Module):
  def __init__(self, a_bit):
    super(activation_quantize_fn, self).__init__()
    assert a_bit <= 8 or a_bit == 32
    self.a_bit = a_bit
    self.uniform_q = uniform_quantize(k=a_bit)

  def forward(self, x):
    if self.a_bit == 32:
      activation_q = x
    else:
      activation_q = self.uniform_q(torch.clamp(x, 0, 1))
      # print(np.unique(activation_q.detach().numpy()))
    return activation_q


def conv2d_Q_fn(w_bit):
  class Conv2d_Q(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
      super(Conv2d_Q, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias)
      self.w_bit = w_bit
      self.quantize_fn = weight_quantize_fn(w_bit=w_bit)

    def forward(self, input, order=None):
      weight_q = self.quantize_fn(self.weight) # @BL use quantized weights
      print(weight_q)
      # print(np.unique(weight_q.detach().numpy()))
      return F.conv2d(input, weight_q, self.bias, self.stride,
                      self.padding, self.dilation, self.groups)

  return Conv2d_Q


def linear_Q_fn(w_bit):
  class Linear_Q(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
      super(Linear_Q, self).__init__(in_features, out_features, bias)
      self.w_bit = w_bit
      self.quantize_fn = weight_quantize_fn(w_bit=w_bit)

    def forward(self, input):
      weight_q = self.quantize_fn(self.weight)
      # print(np.unique(weight_q.detach().numpy()))
      return F.linear(input, weight_q, self.bias) # @BL use quantized weights

  return Linear_Q

```

`ResNetQ_Example.ipynb`
```python
# Importing model
qparams = [4, 4, 'dorefa'] # [abit, qbit, quant_method] --- only dorefa has been implemented
# @Brian activation bit, quantize bit
# resnet models are 8, 14, 16, 24

# activation and weight quantization are computed slightly different.
quant_resnet = get_quant_model('resnet8', qparams) # @Brian get the quantized model with this config

# ----

# Note: first and last layer are not quantized 
# looking at layer ouptuts using hooks
outputs=[]
quant_resnet = get_quant_model('resnet8', qparams)


def hook(module, input, output):
    outputs.append(output)

# Registering quantized act and 
for n, m in quant_resnet.named_modules(): # @BL name and module
    if n == 'conv1': # first layer output
        m.register_forward_hook(hook) # @BL registers hook
    elif 'qfn' in n:
        m.register_forward_hook(hook)
    elif 'fc' in n:
        m.register_forward_hook(hook)
import torch
# reset model or it will continue appending outputs 
image = torch.unsqueeze(torch.rand(size=(3, 32, 32)), dim=0)
out = quant_resnet(image)
```

## Aakash Changes

`utils/setup_manager.py`
```python
parser.add_argument('--teacher-wbits', default = 8 ) # @BL teacher weight has 8 bits
parser.add_argument('--teacher-abits', default = 8 ) # @BL teacher activation has 8 bits
parser.add_argument('--teacher-quantization', default = 'dorefa' ) # @BL quantization method, This looks like qparams
parser.add_argument('--student-wbits', default = 4 )
parser.add_argument('--student-abits', default = 4 )
parser.add_argument('--student-quantization', default = 'dorefa' )
# @BL we set up the config from teacher to student and train!
```

```utils/train_manager.py
from resnet_quant import get_quant_model, is_resnet

teacher_model = get_quant_model(arg, use_cuda=args.cuda)
teacher_model = get_quant_model(args.teacher, [args.teacher_wbits, args.teacher_abits, args.teacher_quantization], dataset, use_cuda=args.cuda)

student_model = get_quant_model(args.student, [args.student_wbits, args.student_abits, args.student_quantization], dataset, use_cuda=args.cuda)
```

## Self Notes 20-04-20
* We can now do training!
* TODO:
    * Train individually
        * Train 32 bit
        * Train 16 bit
        * Train 8 bit
        * Train 4 bit
        * Train 2 bit
        * Train 1 bit
    * Train TA style
        * Train 16 bit student using 32 bit teacher
        * Train 8 bit student using 16 bit student
        * Train 4 bit student using 8 bit teacher
        * Train 2 bit student using 4 bit teacher
        * Train 1 bit student using 2 bit teacher

`resnet_quant.py`
```python
"""
resnet for cifar in pytorch

Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
"""
import torch
import torch.nn as nn
import math
from quantization.q_funcs import * # @BL use quantize function
from utils.dataset_loader import get_dataset
from resnet import get_model

class Conv2d_Q(nn.Conv2d):

    def __init__(self, wbit, in_planes, out_planes, kernel_size=3, stride=1,
                 padding=1, q_method='dorefa', bias=False):

        super(Conv2d_Q, self).__init__(in_planes, out_planes, kernel_size, stride,
                                       padding, bias=False)
        self.qfn = weight_quantize_fn(w_bit=wbit)


    def forward(self, x):
        weights_q = self.qfn(self.weight) # @BL quantize weights
        self.weights_q = weights_q
        self.weights_fp = self.weight # @BL doesn't actually use it anywhere
        return nn.functional.conv2d(x, weights_q, self.bias, self.stride,
                                    self.padding)

# @BL our models use BasicBlocks
class PreActBasicBlock_convQ(nn.Module):
    expansion = 1

    def __init__(self, q_method, wbit, abit, in_planes, out_planes, stride, downsample=None):
        super(PreActBasicBlock_convQ, self).__init__()
        self.act_qfn = activation_quantize_fn(a_bit=abit)
        # TODO: Move class from this file
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = Conv2d_Q(wbit, in_planes, out_planes, stride=stride, kernel_size=3, padding=1, bias=False, q_method=q_method)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = Conv2d_Q(wbit, out_planes, out_planes, stride=1, kernel_size=3, padding=1, q_method=q_method)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = x

        # activation qunatization applied here
        out = self.bn1(x)
        out = self.act_qfn(self.relu(out))

        # TODO: check how residual is accounted for. DoReFa seems to leave residual full precision??
        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)
        out = self.bn2(out)
        out = self.act_qfn(self.relu(out))
        out = self.conv2(out)

        out += residual

        return out


# @BL Bottleneck is an alternative we do not use
class PreActBottleneck_Q(nn.Module):
    expansion = 4

    def __init__(self, q_method, wbit, abit, inplanes, planes, stride=1, downsample=None):
        super(PreActBottleneck_Q, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = Conv2d_Q(wbit, inplanes, planes, kernel_size=1, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2d_Q(wbit, planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = Conv2d_Q(wbit, planes, planes * 4, kernel_size=1, stride=stride, padding=1, bias=False)
        self.downsample = downsample
        self.stride = stride

        self.act_qfn = activation_quantize_fn(a_bit=abit)

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.act_qfn(self.relu(out))

        # TODO: check if quant is needed here
        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)
        out = self.bn2(out)
        out = self.act_qfn(self.relu(out))

        out = self.conv2(out)
        out = self.bn3(out)
        out = self.act_qfn(self.relu(out))

        out = self.conv3(out)
        # TODO: Again, check if residual needs and qunatization
        out += residual

        return out




class PreAct_ResNet_Cifar_Q(nn.Module):

    def __init__(self, block, layers, wbit, abit, num_classes=10, q_method='dorefa'):
        super(PreAct_ResNet_Cifar_Q, self).__init__()
        self.inplanes = 16
        # first conv is not quantized
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, layers[0], wbit, abit, q_method=q_method) # @BL _make_layer using block
        self.layer2 = self._make_layer(block, 32, layers[1], wbit, abit, stride=2, q_method=q_method)
        self.layer3 = self._make_layer(block, 64, layers[2], wbit, abit, stride=2, q_method=q_method)
        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        # TODO: There is no linear quantization done here... check paper to see what they discuss
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, wbit, abit, stride=1, q_method='dorefa'):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)
            )

        layers = []
        layers.append(block(q_method, wbit, abit, self.inplanes, planes, stride, downsample)) # @BL append blockwith downsample?
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(q_method, wbit, abit, self.inplanes, planes))
        return nn.Sequential(*layers) # @BL take the list and make it into a sequental neural network

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet8_cifar(wbit, abit, q_method=None, **kwargs):
    model = PreAct_ResNet_Cifar_Q(PreActBasicBlock_convQ, [1, 1, 1], wbit, abit, q_method=q_method, **kwargs)
    return model


def resnet14_cifar(wbit, abit, q_method=None, **kwargs):
    model = PreAct_ResNet_Cifar_Q(PreActBasicBlock_convQ, [2, 2, 2], wbit, abit, q_method=q_method, **kwargs)
    return model


def resnet20_cifar(wbit, abit, q_method=None, **kwargs):
    model = PreAct_ResNet_Cifar_Q(wbit, abit, q_method, PreActBasicBlock_convQ, [3, 3, 3], wbit, abit, q_method=q_method, **kwargs)
    return model


def resnet26_cifar(wbit, abit, q_method=None, **kwargs):
    model = PreAct_ResNet_Cifar_Q(PreActBasicBlock_convQ, [4, 4, 4], wbit, abit, q_method=q_method, **kwargs)
    return model


resnet_models = {
    '8': resnet8_cifar,
    '14': resnet14_cifar,
    '20': resnet20_cifar,
    '26': resnet26_cifar,
}


def is_resnet(name):
    """
    Simply checks if name represents a resnet, by convention, all resnet names start with 'resnet'
    :param name:
    :return:
    """
    name = name.lower()
    return name.startswith('resnet')


def get_quant_model(name, qparams, dataset="cifar100", use_cuda=False):
    """
    Create a student for training, given student name and dataset
    :param name: name of the student. e.g., resnet110, resnet32, plane2, plane10, ...
    :param dataset: the dataset which is used to determine last layer's output size. Options are cifar10 and cifar100.
    :return: a pytorch student for neural network
    """
    num_classes = 100 if dataset == 'cifar100' else 10
    wbits, abits, q_method = qparams
    model = None
    if is_resnet(name):
        resnet_size = name[6:]
        resnet_model = resnet_models.get(resnet_size)
        model = resnet_model(wbits, abits, q_method, num_classes=num_classes)
    else:
        raise Exception('not resnet!')

    # copy to cuda if activated
    if use_cuda:
        model = model.cuda()

    return model
```