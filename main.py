import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from keras import backend as K


def load_data(file_name, header_column):
    input_file = pd.read_csv(file_name, skiprows=header_column - 1, error_bad_lines=False)
    exoplanet_headers = input_file.columns.values
    exoplanet_data = input_file.values

    return exoplanet_headers, exoplanet_data


def train_test_split(headers, data, test_size, random):
    split = int(len(data) * test_size)

    if random:
        np.random.shuffle(data)

    training_data = data[0: split]
    test_data = data[split: len(data)]

    x_train = np.asarray(training_data).astype('float32')
    x_test = np.asarray(test_data).astype('float32')

    return x_train, x_test


headers, data = load_data("nasa_filtered.csv", 1)

training, testing, = train_test_split(headers, data, 0.05, False)

# randomly remove some data
def random_remove(df, rate):
  df = pd.DataFrame(df)
  rate = rate+df.isnull().sum().sum()/(df.shape[0]*df.shape[1])
  dataset = df.reset_index()
  melt_one = pd.melt(dataset, id_vars = ['index'])
  sampled = melt_one.sample(frac = rate, random_state=1).reset_index(drop = True)
  dataset = sampled.pivot(index = 'index', columns = 'variable', values= 'value')
  dataset = np.asarray(dataset).astype('float32')
  return dataset

# impute mean value into dataset
def fill_mean(dataset):
  dataset = pd.DataFrame(dataset)
  for column in list(dataset.columns[dataset.isnull().sum() > 0]):
    mean_val = dataset[column].mean()
    dataset[column].fillna(mean_val, inplace=True)
  dataset = np.asarray(dataset).astype('float32')
  return dataset

# model with expectation-maximization algorithm
class EMAmodel(Model):
  def __init__(self, x, nodes, times):
    super(EMAmodel, self).__init__()
    self.times = times
    self.length = x.shape[1]
    self.layer = []
    for i in range(self.length):   # create branches
      self.layer.append(tf.keras.Sequential(
        layers=[Dense(nodes, activation='tanh', input_shape=(self.length,)),Dense(1)], name=None))

  def call(self, x):
    for i in range(self.times):   # iterations to calculate the mean value of x^n and x^(n+1)
      tmp = []
      for j in range(self.length):      # create multi-output network
        tmp.append(self.layer[j](x))
      output = tf.concat(tmp,1)
      x = (x+output)/2
    return x

def evaluation(predictions, y):                   # evaluate r2
  sse = np.sum((y - predictions) ** 2)
  sst = np.sum((y - np.mean(y)) ** 2)
  return 1 - sse / sst

def train(x_train, y_train):               # training step
  with tf.GradientTape() as tape:
    predictions = model(x_train)
    loss = loss_object(y_train, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)                  # calculate gradients
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))        # optimize parameters
  r2 = evaluation(predictions, y_train)
  return loss, r2

def N_folder_split_data(data, N):
  data = np.array(data)
  num = len(data)
  div = int(np.floor(num / N))
  res = num % N
  sep_poi = N - res
  training = []
  validation = []
  for factor in range(1, N + 1):
    if factor <= sep_poi:
      i_left = (factor - 1) * div
      i_right = factor * div
      data_validate = data[i_left:i_right]
    elif factor > sep_poi:
      i_left = sep_poi * div + (factor - sep_poi - 1) * (div + 1)
      i_right = sep_poi * div + (factor - sep_poi) * (div + 1)
      data_validate = data[i_left:i_right]
    data_test_left = data[0:i_left]
    data_test_right = data[i_right:]
    data_rest = np.concatenate((data_test_left, data_test_right), axis=0)
    validation.append(data_validate)
    training.append(data_rest)
  return training, validation

def test(x_test, y_test):  # testing step
    predictions = model(x_test)
    loss = loss_object(y_test, predictions)
    r2 = evaluation(predictions, y_test)
    template = 'test set: Loss {}, Evaluation {}'
    print(template.format(loss, r2))

def validate(x_validation, y_validation):  # validation step
    predictions = model(x_validation)
    loss = loss_object(y_validation, predictions)
    r2 = evaluation(predictions, y_validation)
    template = 'validate set: Loss {}, Evaluation {}'
    print(template.format(loss, r2))
    loss = K.eval(loss)
    return loss, r2

N = 5
nodes = 5
times = 5
epochs = 5
y_test = fill_mean(testing)
x_test = random_remove(testing, 0.5)
fill_mean(x_test)
loss_object = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam(learning_rate=0.1)
training_data, validation_data = N_folder_split_data(training, N)
loss_result = 0
r2_result = 0

for i in range(N):
    training = training_data[i]
    y_train = fill_mean(training)
    validation = validation_data[i]
    y_validation = fill_mean(validation)
    x_validation = random_remove(validation, 0.5)
    fill_mean(x_validation)
    model = EMAmodel(training, nodes, times)           # instantiate model
    for epoch in range(epochs):                        # training epochs
        x_train = random_remove(training, 0.5)
        fill_mean(x_train)
        loss, r2 = train(x_train, y_train)
        if (epoch % 1 == 0):
            template = 'Epoch {}, Loss {}, Evaluation {}'
            print(template.format(epoch, loss, r2))
        test(x_test, y_test)  # test

    loss, r2 = validate(x_validation, y_validation)
    loss_result = loss+loss_result
    r2_result = r2+r2_result

print(N, 'fold validation result is:', loss_result/N, r2_result/N)