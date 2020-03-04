# Homework 2 RNN and LSTM
* [Link](https://bcourses.berkeley.edu/courses/1487769/pages/assignment-2-description)
* [RNN and LSTM lecture slides](https://briantliao.com/store/cs282-lectures-slides/lec09.pdf)
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
* \\( \odot \\) is the elementwise product of vectors.
* [Understanding LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

![](https://i.imgur.com/5LFX2M2.png)

* RNN

![](https://i.imgur.com/OHAy8uM.png)

* LSTM


LSTM backwards

## Q3: Network Visualization: Saliency maps, Class Visualization, and Fooling Images (20 points)

### Saliency

![](https://i.imgur.com/agqsFrM.png)

* Which pixel has the most effect on the input

![](https://i.imgur.com/a1XejED.png)

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
```python
np.zeros_like( another numpy array )

A.shape # (3, 4)
B.shape # (4, 4)
np.dot(A, B).shape # (3, 4)

np.zeros((dim1, dim2, dim3))

 x_pad[n,:,y_padded:y_padded+HH,x_padded:x_padded+WW]

 x[0,0,0] # indexes a single value
 x[0:5, 1:4, 6:8] # slices a tensor

x[-1] == x[len(x) - 1] 

# let x in 10 elements, index 0 to 9
for i in range(x - 1, -1, -1):
    print(i)
# 9, 8, 7 ... 2, 1, 0

```

### Calculus

### Chain Rule:
$$ \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial x} $$

Same for \\( \frac{\partial L}{\partial b} \\) and \\( \frac{\partial L}{\partial W} \\).

We have \\( \frac{\partial L}{\partial y} \\) and we can solve for \\( \frac{\partial y}{\partial y} \\) locally. This may need values cached from the forward pass.