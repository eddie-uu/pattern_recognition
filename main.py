import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense


def load_data(file_name, header_column):
    input_file = pd.read_csv(file_name, skiprows=header_column - 1, error_bad_lines=False)
    exoplanet_headers = input_file.columns.values
    exoplanet_data = input_file.values

    return exoplanet_headers, exoplanet_data


def train_test_split(headers, data, test_size, random):
    split = int(len(data) * test_size)

    # for checking the model with something more simplistic
    # orbital_period_days_index = np.where(headers == "Orbital Period Days")[0][0]
    # mass_index = np.where(headers == "Mass")[0][0]

    if random:
        np.random.shuffle(data)

    test_data = data[0: split]
    training_data = data[split: len(data)]

    # x_train = training_data[:, orbital_period_days_index:mass_index + 1]
    # x_test = test_data[:, orbital_period_days_index:mass_index + 1]

    x_train = np.asarray(test_data).astype('float32')
    x_test = np.asarray(training_data).astype('float32')

    return x_train, x_test  # np.nan_to_num(x_train), np.nan_to_num(x_test)


headers, data = load_data("database_all_numerical_filtered.csv", 104)

training, testing, = train_test_split(headers, data, 0.05, False)

print(testing)
'''
model = models.Sequential()

model.add(layers.Input(shape=training.shape[1:]))
# If you get an error related to incorrect output size, change units to the same size as the shape of your second fit
model.add(layers.Dense(units=67, activation="relu"))

model.compile(optimizer='adam',
               loss='categorical_crossentropy',
               metrics=['accuracy'])

model.fit(training, training)  # Second one determines output size of the model
'''
nodes=5  #nodes each output related to 
length=training.shape[1:] 
inputs = Input(shape=length)

# define independent output
class output_layer(Model):
  def __init__(self, nodes):
    super(output_layer, self).__init__()
    self.layer1 = Dense(nodes, activation='tanh')
    self.layer2 = Dense(1)
  def call(self, x):
    x = self.layer1(x)
    return self.layer2(x)

# creat output branches
output = list()
for i in range(length):
  temp = output_layer(nodes)(inputs)
  output.append(temp)

model = Model(inputs=inputs, outputs=output)
model.compile(optimizer='adam', loss=keras.losses.mean_squared_error) # temorary optimizer
predict = model(train)

# when reach the convergence begin parameter updating
def convergence(train, training):
    predict = model(train)
    while True:
        new_train = (train + predict) / 2 
        if (train - new_train) < 0.1: # not sure about this condition
            model.fit(train, training) # update parameters
            break
        else:
            train = new_train
            predict = model(train)

# set the epoch of model, use number or a certain condition. before reach this epoch continue running the convergence function
def model_epoch(train, training, somecondition):
    while True:
        if somecondition: # not sure about this condition
            break
        else:
            convergence(train, training)

model_epoch(training, training, somecondition)

print(len(training))
print(len(testing))
print(model.predict(testing))
exit()

# Below is an example setup, we can tweak this

    # PREPROCESSING
def filtering(full_dataset):    # Filter out all the rows that are missing data (and thus can't be used for training)
    ...
    return filtered_dataset

def divide_dataset(complete_dataset):   #takes a (complete) dataset and splits it into two parts (one for training, one for validation)
    ...
    return training_set, validation_set

def make_incomplete_set(complete_dataset):     # Remove some datapoints from a (complete) dataset to make a training dataset
    ... 
    return incomplete_dataset 

def preprocessing(dataset):
    complete_dataset = filtering(dataset)
    complete_training_set, complete_validation_set = divide_dataset(complete_dataset)
    incomplete_training_set = make_incomplete_set(complete_training_set)
    incomplete_validation_set = make_incomplete_set(complete_validation_set)
    return complete_training_set, incomplete_training_set, complete_validation_set, incomplete_validation_set

    # MODEL
def fill_in_blanks(incomplete_data): # Fill in some average values for missing data
    ...
    return estimated_data

def is_complete(data):   #checks if there are values missing in the data, returning False if values are missing or True if the data is complete
    ...
    return True/False

def not_converged(data1, data2):    # Checks if the difference between the two data vectors is smaller than some threshold (keep in mind to normalize values!)
    ...
    return True/False

def apply_model(input_data):
    if is_complete(input_data):
        return input_data
    else:
        ...
        old_data = fill_in_blanks(input_data)
        new_data = old_data
        while not_converged(old_data, new_data):
            old_data = new_data
            new_data = 0.5*(new_data + tf.ourmodel.applymodel(new_data))  # We'll obviously need to use different functions for this, this is more of an example of what it should do
    return new_data 
    # TRAINING
# honestly, I'm not quite sure what the tf methods for training a model that is called inside a function is




