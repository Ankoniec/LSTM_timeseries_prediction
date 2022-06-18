<<<<<<< HEAD
from tkinter import HIDDEN
from xml.dom import HierarchyRequestErr
import numpy as np
import matplotlib.pyplot as plt
=======
import numpy as np
>>>>>>> 3a3385426f5314dfc7f66420212db4c71a8bb3b1
import torch
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2

from dataset_manager import MeteoDataset, create_sequences
from plots_manager import PlotData, plot_taylor_diagram
from train_helper import train_model
from model import LSTM

<<<<<<< HEAD
mae_val = []
std_dev = []
r_score = []
labels = []
=======
>>>>>>> 3a3385426f5314dfc7f66420212db4c71a8bb3b1


# READ DATA
print("Reading data...")

dataset = MeteoDataset("C:\\Users\\AnnaMaria\\Desktop\\Fizyka Systemu Ziemi\\program\\data_pm10.csv")
data = dataset.df.values
sc = MinMaxScaler() 
training_data = sc.fit_transform(data)

<<<<<<< HEAD
SEQ_LENGTH = 21

x, y = create_sequences(training_data, SEQ_LENGTH)
ratio = 0.8
N_train = (int)(ratio*len(x))
=======
SEQ_LENGTH = 16
x, y = create_sequences(training_data, SEQ_LENGTH)
ratio = 0.8
N_train = (int)(ratio*len(x))

>>>>>>> 3a3385426f5314dfc7f66420212db4c71a8bb3b1
X_train = x[:N_train]
X_test = x[N_train:]
y_train = y[:N_train]
y_test = y[N_train:]
<<<<<<< HEAD
=======

>>>>>>> 3a3385426f5314dfc7f66420212db4c71a8bb3b1
dataX = Variable(torch.Tensor(np.array(x)))
dataY = Variable(torch.Tensor(np.array(y)))
trainX = Variable(torch.Tensor(np.array(X_train)))
trainY = Variable(torch.Tensor(np.array(y_train)))
testX = Variable(torch.Tensor(np.array(X_test)))
testY = Variable(torch.Tensor(np.array(y_test)))
<<<<<<< HEAD
# HYPERPARAMETERS
print("Reading hyperparameters...")

NUM_EPOCHS = [5000,10000,15000,20000,25000]
LEARNING_RATE = 0.0001

NUM_CLASSES = 1
INPUT_SIZE = 1
NUM_LAYERS = 1

HIDDEN_SIZE = 2
    
for num_epoch in NUM_EPOCHS:
    lstm = LSTM(NUM_CLASSES, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, SEQ_LENGTH)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=LEARNING_RATE)
    figure_name = "epochs" + str(num_epoch) + "_lr001"

    train_model(num_epoch, lstm, optimizer, criterion, trainX, trainY, 1, figure_name)


    # EVALUATING ON TEST DATA
    print("Evaluating on test data...")

    lstm.eval()
    train_predict = lstm(dataX)

    data_predict = train_predict.data.numpy()
    data_observed = dataY.data.numpy()
    data_predict = sc.inverse_transform(data_predict)
    data_observed= sc.inverse_transform(data_observed)

    plts = PlotData(data_observed, data_predict, N_train)
    plts.plot_all(figure_name)
    plts.plot_test_data(figure_name)

    # CALCULATE ERRORS
    print(f"Mean absolute error: {mae(data_observed[N_train:],data_predict[N_train:])}")
    print(f"Mean squared error: {mse(data_observed[N_train:],data_predict[N_train:])}")
    print(f"Root mean squared error: {mse(data_observed[N_train:],data_predict[N_train:],squared=False)}")
    print(f"Model standard deviation: {np.std(data_predict[N_train:])}")
    print(f"Observed data standard deviation: {np.std(data_observed[N_train:])}")
    print(f"R2 score: {r2(data_observed[N_train:],data_predict[N_train:])}")

    # TAYLOR DIAGRAM
    mae_val.append(mae(data_observed[N_train:],data_predict[N_train:]))
    std_dev.append(np.std(data_predict[N_train:]))
    r_score.append(r2(data_observed[N_train:],data_predict[N_train:]))
    labels.append("epochs="+str(num_epoch))

STD = np.std(data_observed[N_train:])
plot_taylor_diagram(STD,std_dev,r_score,labels,"first")

fig3, ax3 = plt.subplots(figsize=(12,8))
ax3.plot(NUM_EPOCHS,mae_val)
ax3.set_xlabel("Liczba epok")
ax3.set_ylabel("Średni błąd względny")
fig3.savefig("MAE_lr001.jpg")
=======

# HYPERPARAMETERS
print("Reading hyperparameters...")

NUM_EPOCHS = 1000
LEARNING_RATE = 0.01

NUM_CLASSES = 1
INPUT_SIZE = 1
HIDDEN_SIZE = 2
NUM_LAYERS = 1

lstm = LSTM(NUM_CLASSES, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, SEQ_LENGTH)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=LEARNING_RATE)
figure_name = "hidden" + str(HIDDEN_SIZE) + "_seqlen" + str(SEQ_LENGTH) + "_epochs" + str(NUM_EPOCHS)

train_model(NUM_EPOCHS, lstm, optimizer, criterion, trainX, trainY, 1, figure_name)


# EVALUATING ON TEST DATA
print("Evaluating on test data...")

lstm.eval()
train_predict = lstm(dataX)

data_predict = train_predict.data.numpy()
data_observed = dataY.data.numpy()
data_predict = sc.inverse_transform(data_predict)
data_observed= sc.inverse_transform(data_observed)

plts = PlotData(data_observed, data_predict, N_train)
plts.plot_all()
plts.plot_test_data()

# CALCULATE ERRORS

print(f"Mean absolute error: {mae(data_observed[N_train:],data_predict[N_train:])}")
print(f"Mean squared error: {mse(data_observed[N_train:],data_predict[N_train:])}")
print(f"Root mean squared error: {mse(data_observed[N_train:],data_predict[N_train:],squared=False)}")
print(f"Model standard deviation: {np.std(data_predict[N_train:])}")
print(f"Observed data standard deviation: {np.std(data_observed[N_train:])}")
print(f"R2 score: {r2(data_observed[N_train:],data_predict[N_train:])}")

# TAYLOR DIAGRAM

STD = np.std(data_observed[N_train:])

std_dev = [np.std(data_predict[N_train:])]
r_score = [r2(data_observed[N_train:],data_predict[N_train:])]
label = ["1"]

plot_taylor_diagram(STD,std_dev,r_score,label,"first")
>>>>>>> 3a3385426f5314dfc7f66420212db4c71a8bb3b1
