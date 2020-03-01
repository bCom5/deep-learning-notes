# Homework 2 LSTM
[Link](https://bcourses.berkeley.edu/courses/1487769/pages/assignment-2-description)

## Setup

```sh
conda install "tensorflow<2.0"          # cpu version
conda install "tensorflow-gpu<2.0"      # gpu version
# I think I will use PyTorch though

conda create -n cs182-assignment2  
conda activate cs182-assignment2
# to deactivate:  conda deactivate

pip3 install -r requirements.txt # @Brian changed to pip3
```

### GPUs
GPUs are not required for this assignment, but will help to speed up training and processing time for questions 3-4.

### Download Data
```sh
cd deeplearning/datasets
./get_assignment2_data.sh
```

Now you can use Jupyter Notebook `jupyter serve`!

## Q1: Image Captioning with Vanilla RNNs (30 points)

![](https://i.imgur.com/dysQhPY.png)

* RNN Equation

b included before tanh:
`sum_together = dot_x + dot_h + b`

## Q2: Image Captioning with LSTMs (30 points)

## Q3: Network Visualization: Saliency maps, Class Visualization, and Fooling Images (20 points)

## Q4: Style Transfer (20 points)

## Debugging
List Conda Enviornments:
```
conda env list
```

Trying without conda env.


```
pip install scipy==1.1.0
```

## Tips and Notes

### Numpy
```
np.zeros_like

A.shape # (3, 4)
B.shape # (4, 4)
np.dot(A, B).shape # (3, 4)
```

### Calculus

### Chain Rule:
$$ \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial x} $$

Same for \\( \frac{\partial L}{\partial b} \\) and \\( \frac{\partial L}{\partial W} \\).

We have \\( \frac{\partial L}{\partial y} \\) and we can solve for \\( \frac{\partial y}{\partial y} \\) locally. This may need values cached from the forward pass.