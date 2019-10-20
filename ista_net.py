### IMPORT THE REQUIRED LIBRARIES

# To read the dataset in .mat format
import scipy.io as sio

# For matrix operations
import numpy as np

# Keras functions to create and compile the model
from keras.layers import Input, Conv2D, Lambda, Reshape, Multiply, Add, Subtract
from keras.activations import relu
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K


### READING THE DATA
phi_read = sio.loadmat('phi_0_25_1089.mat')
train = sio.loadmat('Training_Data_Img91.mat')

### PREPROCESSING

# Reading training input and labels
train_inp = train['inputs']
train_labels = train['labels']

# Preparing the constant matrices
phi = np.transpose(phi_read['phi'])
ptp = np.dot(phi, np.transpose(phi))    # phi^T x phi
temp1 = np.transpose(train_labels)      
temp2 = np.dot(np.transpose(phi), temp1)
temp3 = np.dot(np.dot(temp1, np.transpose(temp2)), np.linalg.inv(np.dot(temp2, np.transpose(temp2))))
phi_inv = np.transpose(temp3)           # phi^-1

# Instead of multiplying each batch by phi and then supplying it to the model as input, 
# we multiply the entire training set by phi in the preprocessing stage itself
x_inp = np.dot(train_labels, phi)       

### INITIALIZING CONSTANTS
n_input = 272
tau = 0.1
lambda_step = 0.1
soft_thr = 0.1
conv_size = 32
filter_size = 3

### PREPARING THE MODEL (An image of the model map has been attached)

# Defining the input and output
inp = Input((n_input,))
inp_labels = Input((1089, ))

# Defining the input for the first ISTA block
x0 = Lambda(lambda x: K.dot(x, K.constant(phi_inv)))(inp)
phi_tb = Lambda(lambda x: K.dot(x, K.constant(np.transpose(phi))))(inp)

# ISTA block #1
conv1_x1 = Lambda(lambda x: x - lambda_step * K.dot(x, K.constant(ptp)) + lambda_step * phi_tb, name='conv1_x1')(x0)
conv1_x2 = Reshape((33, 33, 1), name='conv1_x2')(conv1_x1)
conv1_x3 = Conv2D(conv_size, [filter_size, filter_size], padding='SAME', use_bias=False, name='conv1_x3')(conv1_x2)
conv1_sl1 = Conv2D(conv_size, [filter_size, filter_size], padding='SAME', use_bias=False, activation='relu', name='conv1_sl1')
conv1_x4 = conv1_sl1(conv1_x3)
conv1_sl2 = Conv2D(conv_size, [filter_size, filter_size], padding='SAME', use_bias=False, name='conv1_sl2')
conv1_x44 = conv1_sl2(conv1_x4)
conv1_x5 = Multiply(name='conv1_x5')([Lambda(lambda x: K.sign(x))(conv1_x44), Lambda(lambda x: relu(x - soft_thr))(Lambda(lambda x: K.abs(x))(conv1_x44))])
conv1_sl3 = Conv2D(conv_size, [filter_size, filter_size], padding='SAME', use_bias=False, activation='relu', name='conv1_sl3')
conv1_x6 = conv1_sl3(conv1_x5)
conv1_sl4 = Conv2D(conv_size, [filter_size, filter_size], padding='SAME', use_bias=False, name='conv1_sl4')
conv1_x66 = conv1_sl4(conv1_x6)
conv1_x7 = Conv2D(1, [filter_size, filter_size], padding='SAME', use_bias=False, name='conv1_x7a')(conv1_x66)
conv1_x7 = Add(name='conv1_x7b')([conv1_x7, conv1_x2])
conv1_x8 = Reshape((1089,), name='conv1_x8')(conv1_x7)
conv1_x3_sym = conv1_sl1(conv1_x3)
conv1_x4_sym = conv1_sl2(conv1_x3_sym)
conv1_x6_sym = conv1_sl3(conv1_x4_sym)
conv1_x7_sym = conv1_sl4(conv1_x6_sym)
conv1_x11 = Subtract(name='conv1_x11')([conv1_x7_sym, conv1_x3])
conv1 = conv1_x8
conv1_sym = conv1_x11

# ISTA block #2
conv2_x1 = Lambda(lambda x: x - lambda_step * K.dot(x, K.constant(ptp)) + lambda_step * phi_tb, name='conv2_x1')(conv1)
conv2_x2 = Reshape((33, 33, 1), name='conv2_x2')(conv2_x1)
conv2_x3 = Conv2D(conv_size, [filter_size, filter_size], padding='SAME', use_bias=False, name='conv2_x3')(conv2_x2)
conv2_sl1 = Conv2D(conv_size, [filter_size, filter_size], padding='SAME', use_bias=False, activation='relu', name='conv2_sl1')
conv2_x4 = conv2_sl1(conv2_x3)
conv2_sl2 = Conv2D(conv_size, [filter_size, filter_size], padding='SAME', use_bias=False, name='conv2_sl2')
conv2_x44 = conv2_sl2(conv2_x4)
conv2_x5 = Multiply(name='conv2_x5')([Lambda(lambda x: K.sign(x))(conv2_x44), Lambda(lambda x: relu(x - soft_thr))(Lambda(lambda x: K.abs(x))(conv2_x44))])
conv2_sl3 = Conv2D(conv_size, [filter_size, filter_size], padding='SAME', use_bias=False, activation='relu', name='conv2_sl3')
conv2_x6 = conv2_sl3(conv2_x5)
conv2_sl4 = Conv2D(conv_size, [filter_size, filter_size], padding='SAME', use_bias=False, name='conv2_sl4')
conv2_x66 = conv2_sl4(conv2_x6)
conv2_x7 = Conv2D(1, [filter_size, filter_size], padding='SAME', use_bias=False, name='conv2_x7a')(conv2_x66)
conv2_x7 = Add(name='conv2_x7b')([conv2_x7, conv2_x2])
conv2_x8 = Reshape((1089,), name='conv2_x8')(conv2_x7)
conv2_x3_sym = conv2_sl1(conv2_x3)
conv2_x4_sym = conv2_sl2(conv2_x3_sym)
conv2_x6_sym = conv2_sl3(conv2_x4_sym)
conv2_x7_sym = conv2_sl4(conv2_x6_sym)
conv2_x11 = Subtract(name='conv2_x11')([conv2_x7_sym, conv2_x3])
conv2 = conv2_x8
conv2_sym = conv2_x11

# ISTA block #3
conv3_x1 = Lambda(lambda x: x - lambda_step * K.dot(x, K.constant(ptp)) + lambda_step * phi_tb, name='conv3_x1')(conv2)
conv3_x2 = Reshape((33, 33, 1), name='conv3_x2')(conv3_x1)
conv3_x3 = Conv2D(conv_size, [filter_size, filter_size], padding='SAME', use_bias=False, name='conv3_x3')(conv3_x2)
conv3_sl1 = Conv2D(conv_size, [filter_size, filter_size], padding='SAME', use_bias=False, activation='relu', name='conv3_sl1')
conv3_x4 = conv3_sl1(conv3_x3)
conv3_sl2 = Conv2D(conv_size, [filter_size, filter_size], padding='SAME', use_bias=False, name='conv3_sl2')
conv3_x44 = conv3_sl2(conv3_x4)
conv3_x5 = Multiply(name='conv3_x5')([Lambda(lambda x: K.sign(x))(conv3_x44), Lambda(lambda x: relu(x - soft_thr))(Lambda(lambda x: K.abs(x))(conv3_x44))])
conv3_sl3 = Conv2D(conv_size, [filter_size, filter_size], padding='SAME', use_bias=False, activation='relu', name='conv3_sl3')
conv3_x6 = conv3_sl3(conv3_x5)
conv3_sl4 = Conv2D(conv_size, [filter_size, filter_size], padding='SAME', use_bias=False, name='conv3_sl4')
conv3_x66 = conv3_sl4(conv3_x6)
conv3_x7 = Conv2D(1, [filter_size, filter_size], padding='SAME', use_bias=False, name='conv3_x7a')(conv3_x66)
conv3_x7 = Add(name='conv3_x7b')([conv3_x7, conv3_x2])
conv3_x8 = Reshape((1089,), name='conv3_x8')(conv3_x7)
conv3_x3_sym = conv3_sl1(conv3_x3)
conv3_x4_sym = conv3_sl2(conv3_x3_sym)
conv3_x6_sym = conv3_sl3(conv3_x4_sym)
conv3_x7_sym = conv3_sl4(conv3_x6_sym)
conv3_x11 = Subtract(name='conv3_x11')([conv3_x7_sym, conv3_x3])
conv3 = conv3_x8
conv3_sym = conv3_x11

# ISTA block #4
conv4_x1 = Lambda(lambda x: x - lambda_step * K.dot(x, K.constant(ptp)) + lambda_step * phi_tb, name='conv4_x1')(conv3)
conv4_x2 = Reshape((33, 33, 1), name='conv4_x2')(conv4_x1)
conv4_x3 = Conv2D(conv_size, [filter_size, filter_size], padding='SAME', use_bias=False, name='conv4_x3')(conv4_x2)
conv4_sl1 = Conv2D(conv_size, [filter_size, filter_size], padding='SAME', use_bias=False, activation='relu', name='conv4_sl1')
conv4_x4 = conv4_sl1(conv4_x3)
conv4_sl2 = Conv2D(conv_size, [filter_size, filter_size], padding='SAME', use_bias=False, name='conv4_sl2')
conv4_x44 = conv4_sl2(conv4_x4)
conv4_x5 = Multiply(name='conv4_x5')([Lambda(lambda x: K.sign(x))(conv4_x44), Lambda(lambda x: relu(x - soft_thr))(Lambda(lambda x: K.abs(x))(conv4_x44))])
conv4_sl3 = Conv2D(conv_size, [filter_size, filter_size], padding='SAME', use_bias=False, activation='relu', name='conv4_sl3')
conv4_x6 = conv4_sl3(conv4_x5)
conv4_sl4 = Conv2D(conv_size, [filter_size, filter_size], padding='SAME', use_bias=False, name='conv4_sl4')
conv4_x66 = conv4_sl4(conv4_x6)
conv4_x7 = Conv2D(1, [filter_size, filter_size], padding='SAME', use_bias=False, name='conv4_x7a')(conv4_x66)
conv4_x7 = Add(name='conv4_x7b')([conv4_x7, conv4_x2])
conv4_x8 = Reshape((1089,), name='conv4_x8')(conv4_x7)
conv4_x3_sym = conv4_sl1(conv4_x3)
conv4_x4_sym = conv4_sl2(conv4_x3_sym)
conv4_x6_sym = conv4_sl3(conv4_x4_sym)
conv4_x7_sym = conv4_sl4(conv4_x6_sym)
conv4_x11 = Subtract(name='conv4_x11')([conv4_x7_sym, conv4_x3])
conv4 = conv4_x8
conv4_sym = conv4_x11

# ISTA block #5
conv5_x1 = Lambda(lambda x: x - lambda_step * K.dot(x, K.constant(ptp)) + lambda_step * phi_tb, name='conv5_x1')(conv4)
conv5_x2 = Reshape((33, 33, 1), name='conv5_x2')(conv5_x1)
conv5_x3 = Conv2D(conv_size, [filter_size, filter_size], padding='SAME', use_bias=False, name='conv5_x3')(conv5_x2)
conv5_sl1 = Conv2D(conv_size, [filter_size, filter_size], padding='SAME', use_bias=False, activation='relu', name='conv5_sl1')
conv5_x4 = conv5_sl1(conv5_x3)
conv5_sl2 = Conv2D(conv_size, [filter_size, filter_size], padding='SAME', use_bias=False, name='conv5_sl2')
conv5_x44 = conv5_sl2(conv5_x4)
conv5_x5 = Multiply(name='conv5_x5')([Lambda(lambda x: K.sign(x))(conv5_x44), Lambda(lambda x: relu(x - soft_thr))(Lambda(lambda x: K.abs(x))(conv5_x44))])
conv5_sl3 = Conv2D(conv_size, [filter_size, filter_size], padding='SAME', use_bias=False, activation='relu', name='conv5_sl3')
conv5_x6 = conv5_sl3(conv5_x5)
conv5_sl4 = Conv2D(conv_size, [filter_size, filter_size], padding='SAME', use_bias=False, name='conv5_sl4')
conv5_x66 = conv5_sl4(conv5_x6)
conv5_x7 = Conv2D(1, [filter_size, filter_size], padding='SAME', use_bias=False, name='conv5_x7a')(conv5_x66)
conv5_x7 = Add(name='conv5_x7b')([conv5_x7, conv5_x2])
conv5_x8 = Reshape((1089,), name='conv5_x8')(conv5_x7)
conv5_x3_sym = conv5_sl1(conv5_x3)
conv5_x4_sym = conv5_sl2(conv5_x3_sym)
conv5_x6_sym = conv5_sl3(conv5_x4_sym)
conv5_x7_sym = conv5_sl4(conv5_x6_sym)
conv5_x11 = Subtract(name='conv5_x11')([conv5_x7_sym, conv5_x3])
conv5 = conv5_x8
conv5_sym = conv5_x11


# Defining the custom loss metric
def custom_loss(y_true, y_pred):

  # Referred to in the paper as cost
  cost1 = K.mean(K.square(y_pred[1] - y_pred[0]))

  # Referred to in the paper as cost_sym
  cost2 = K.mean(K.square(y_pred[2])) + K.mean(K.square(y_pred[3])) + K.mean(K.square(y_pred[4])) + K.mean(K.square(y_pred[5])) + K.mean(K.square(y_pred[6]))
  
  # Referred to in the paper as cost_all
  cost = cost1 + 0.01*cost2
  return cost


### COMPILING THE MODEL

# Defining the inputs and outputs
model = Model(inputs=[inp, inp_labels], outputs=[conv5, conv1_sym, conv2_sym, conv3_sym, conv4_sym, conv5_sym])

# Display a model summary
model.summary()

# Define costs
cost1 = K.mean(K.square(conv5 - inp_labels))
cost2 = K.mean(K.square(conv1_sym)) + K.mean(K.square(conv2_sym)) + K.mean(K.square(conv3_sym)) + K.mean(K.square(conv4_sym)) + K.mean(K.square(conv5_sym))
cost = cost1 + 0.01*cost2

# Add custom loss
model.add_loss(K.mean(K.square(conv5 - inp_labels)) + 0.01 * K.mean(K.square(conv1_sym)) + K.mean(K.square(conv2_sym)) + K.mean(K.square(conv3_sym)) + K.mean(K.square(conv4_sym)) + K.mean(K.square(conv5_sym)))

# Compile the model
model.compile(optimizer=Adam(lr=0.0001), metrics=[cost, cost1, cost2])

# Define custom metrics to display
model.metrics_tensors.append(K.mean(K.square(conv5 - inp_labels)) + 0.01*K.mean(K.square(conv1_sym)) + K.mean(K.square(conv2_sym)) + K.mean(K.square(conv3_sym)) + K.mean(K.square(conv4_sym)) + K.mean(K.square(conv5_sym)))
model.metrics_names.append("cost")
model.metrics_tensors.append(K.mean(K.square(conv5 - inp_labels)))
model.metrics_names.append("cost1")
model.metrics_tensors.append(K.mean(K.square(conv1_sym)) + K.mean(K.square(conv2_sym)) + K.mean(K.square(conv3_sym)) + K.mean(K.square(conv4_sym)) + K.mean(K.square(conv5_sym)))
model.metrics_names.append("cost2")

### TRAINING THE MODEL

model.fit([x_inp, train_labels],
          epochs = 300,
          batch_size = 64)
