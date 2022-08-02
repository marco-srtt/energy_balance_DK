import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

number = tf.Variable(3, tf.int16)
string = tf.Variable('string', tf.string)

# RANK/DEGREE OF A TENSOR
# total dimensions of a tensor

rank1_tensor = tf.Variable(['hello', 'there'], tf.string)
rank2_tensor = tf.Variable([['hello', 'there'], ['hello', 'there']], tf.string)

print(tf.rank(rank2_tensor))

# SHAPE
# how many items in each dimension

print(tf.shape(rank2_tensor))

# RESHAPE
# same amount of elements but in different shapes

tensor1 = tf.ones([1, 2, 3])
print(tensor1)

# LINEAR REGRESSION
x = np.array([1, 2, 3])
y = np.array([5, 7, 18])
plt.plot(x, y)
a, b, c = np.polyfit(x, y, 2)
plt.plot(x, a * (x *2) + b * x + c)