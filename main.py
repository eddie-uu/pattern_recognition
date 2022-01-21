import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense


def load_data(file_name, header_column):
    input_file = pd.read_csv(file_name, skiprows=header_column - 1, error_bad_lines=False)
    exoplanet_headers = input_file.columns.values
    exoplanet_data = input_file.values

    return exoplanet_headers, exoplanet_data


def train_test_split(headers, data, test_size, random):
    split = int(len(data) * test_size)

    if random:
        np.random.shuffle(data)

    test_validation_data = data[0: split]
    training_data = data[split: len(data)]

    validation_split = int(len(test_validation_data) / 2)

    test_data = test_validation_data[0: validation_split]
    validation_data = test_validation_data[validation_split: len(test_validation_data)]

    x_train = np.asarray(training_data).astype('float32')
    x_test = np.asarray(test_data).astype('float32')
    x_validation = np.asarray(validation_data).astype('float32')

    return x_train, x_test, x_validation


headers, data = load_data("databases/nasa_filtered.csv", 104)
training, testing, validation = train_test_split(headers, data, 0.10, False)


# randomly remove some data
def random_remove(df, rate):
    dataset = df.reset_index()
    melt_one = pd.melt(dataset, id_vars = ['index'])
    sampled = melt_one.sample(frac = rate).reset_index(drop = True)
    dataset = sampled.pivot(index = 'index', columns = 'variable', values= 'value')

    return dataset


# impute mean value into dataset
def fill_mean(dataset):
    for column in list(dataset.columns[dataset.isnull().sum() > 0]):
        mean_val = dataset[column].mean()
        dataset[column].fillna(mean_val, inplace=True)

    return dataset


# preprocess of dataset
y_train = pd.DataFrame(training)
y_test = pd.DataFrame(testing)

x_train = random_remove(y_train, 0.5)
x_test = random_remove(y_test, 0.5)

fill_mean(y_train)
fill_mean(x_train)
fill_mean(y_test)
fill_mean(x_test)
# to test if all values are imputed: print (y_train.isnull().sum().sum())

x_train = np.asarray(x_train).astype('float32')
y_train = np.asarray(y_train).astype('float32')
x_test = np.asarray(x_test).astype('float32')
y_test = np.asarray(y_test).astype('float32')

# parameters of model
nodes = 5
times = 5


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
            output = tf.concat(tmp, 1)
            x = (x+output)/2
        return x


model = EMAmodel(x_train, nodes, times)  # instantiate model
loss_object = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam(learning_rate=0.1)


def evaluation(predictions, y):  # evaluate r2
    sse = np.sum((y - predictions) ** 2)
    sst = np.sum((y - np.mean(y)) ** 2)
    return 1 - sse / sst


def train(x_train, y_train):  # training step
    with tf.GradientTape() as tape:
        predictions = model(x_train)
        loss = loss_object(y_train, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)            # calculate gradients
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))  # optimize parameters
    r2 = evaluation(predictions, y_train)
    return loss, r2


epochs = 20000
for epoch in range(epochs):  # training epochs
    loss, r2 = train(x_train, y_train)
    if (epoch % 100) == 0:
        template = 'Epoch {}, Loss {}, Evaluation {}'
        print(template.format(epoch, loss, r2))


def test(x_test, y_test):  # testing step
    predictions = model(x_test)
    loss = loss_object(y_test, predictions)
    r2 = evaluation(predictions, y_test)
    template = 'Loss {}, Evaluation {}'
    print(template.format(loss, r2))


test(x_test, y_test)          # test

# predict only: predictons = model(dataset)
