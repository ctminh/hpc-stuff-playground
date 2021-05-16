import torch
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from mlpack import linear_regression
import sys


#########################################################
###### Self-define Regression Model (Torch-based)
#########################################################

"""Linear Regression model - user-defined model

Define the net-structure for linear-regression models.
The inputs should be normalized or not, or this depends on
what kind of apps. 
"""
class LinearRegression(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(LinearRegression, self).__init__()
        # self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_feature, n_output)   # output layer

    def forward(self, x):
        # x = torch.relu(x) # activation function for hidden layer
        x = self.predict(x) # linear output
        return x


#########################################################
###### Util-Functions
#########################################################
""" Plot dataset
    - input: raw data from log files, num iters to plot
    - output: plot of runtime/load by iters """
def plot_dataset(dataset, num_iters):
    iter_indices = np.arange(num_iters)
    load_arr = (chamtool_log_data["target"])[0:num_iters]
    plt.title('Samoa-Load Dataset (single rank')
    plt.xlabel('Iterations')
    plt.ylabel('Load (in seconds)')
    plt.plot(iter_indices, load_arr)
    plt.grid(True)
    plt.show()


""" Sine-func data generator
    - input: num points
    - output: x, y perspectively """
def sinefunc_data_generator(num_points):
    x = np.arange(0, num_points)
    y = (np.sin(x) + 1) * 10 + 2*x + np.random.rand(num_points) * 5

    # check the generated data
    # print("X: {}".format(x))
    # print("Y: {}".format(y))
    # ------------------ plot the sin-random data
    # plt.title('Given Data')
    # plt.xlabel('Independent varible')
    # plt.ylabel('Dependent varible')
    # plt.plot(x, y)
    # plt.yscale('log')
    # plt.grid(True)
    # plt.show()

    return x, y


""" Chain-on-chain data generator
    - input: dataset, input_length, fr_point, to_point
    - output: a matrix of data (runtimes of first iters
    are inputs for the next iter) based on rolling window """
def cc_dataset_generator(raw_data, input_length, from_point, to_point):
    NUM_POINTS = input_length
    FROM, TO = from_point, to_point
    ccData = (raw_data["target"])[FROM:TO]

    # Based on the formulation we have,
    # num_points/input_len = F, len of the generated data = N = TO - FROM
    # so, the exact num of data-points we have is (N - F) | # lines of matrix
    # For example, the raw dataset we have:
    #       iter     0    1   2   3 ...  N
    #       runtime  r0  r1  r2  r3 ... rN 
    # The cc_dataset could be
    #       r0 r1 r2 r3   r4
    #       r1 r2 r3 r4   r5
    #       r2 r3 ...     r6 """
    labels = []
    for i in range(NUM_POINTS):
        labels.append("a"+str(i))

    length = TO - FROM
    x_ds = []
    y_ds = []

    for i in range(NUM_POINTS, length):
        x_ds.append(ccData[(i-NUM_POINTS):i])
        y_ds.append(ccData[i+FROM])

    tmp_ds = pd.DataFrame(np.array(x_ds), columns=labels)
    final_ds = tmp_ds
    final_ds["target"] = y_ds

    return final_ds


#########################################################
###### Main Function
#########################################################

def main():

    """ Reading logs and generate dataset - using pandas
        and its data frame """
    col_names = []
    for i in range(50):
        if i == 0:
            col_names.append("tid")
        elif i == 49:
            col_names.append("target")
        else:
            col_names.append("bnd_s"+str(i))

    # check arguments
    if len(sys.argv) > 1:
        rank = int(sys.argv[1])
    else:
        rank = 0
    
    # read logfile and store data into chamtool_log_data
    chamtool_log_data = pd.read_csv("./sample-dataset-r"+str(rank)+".csv", names=col_names, header=None)
    
    # plot the samoa-load data
    # plot_dataset(chamtool_log_data, 100)

    # generate random data to test the trial-model
    # sinefunc_data_generator(num_elements)

    # formulate another type of dataset for training
    input_length = 6
    labels = []
    for i in range(6):
        labels.append("a"+str(i))
    TRAIN_FROM = 0
    TRAIN_TO = 20
    VALID_FROM = 100
    VALID_TO = 300
    train_ds = cc_dataset_generator(chamtool_log_data, input_length, TRAIN_FROM, TRAIN_TO)
    valid_ds = cc_dataset_generator(chamtool_log_data, input_length, VALID_FROM, VALID_TO)
    print("lenght of train_ds = {}".format(len(train_ds)))
    print("length of valid_ds = {}".format(len(valid_ds)))

    
    """ Training the regression models """

    """ 1. Using Torch defined-linearregression model
    convert data to tensor_type for training with Torch """
    # convert pandas-dataframe to torch-tensor
    # x_tensor_data = torch.FloatTensor(ds)
    # y_tensor_data = torch.FloatTensor(target)

    # convert to variables
    # x, y = Variable(x_tensor_data), Variable(y_tensor_data)

    # using a given declared model above with Torch
    # net_model = LinearRegression(n_feature=12, n_hidden=10, n_output=1)
    # optimizer = torch.optim.SGD(net_model.parameters(), lr=0.001)
    # loss_fn = torch.nn.MSELoss(reduction='mean')

    # training with visualizing the process
    # n_epoch = 4
    # net_model.train()
        
    # for epoch in range(n_epoch):
    #     optimizer.zero_grad()
    #     prediction = net_model(x)
    #     loss = loss_fn(prediction, y)
        
    #     loss.backward()
    #     optimizer.step()
    #     print("[TRAINING] Epoch {} - Loss = {}".format(epoch, loss))


    """ 2. Using  Scikit-learn fit-model
    use pandas-data frame and train the model """
    # prepare dataset for training
    # x_train_ds = (chamtool_log_data[col_names])[0:50]
    # y_train_ds = (chamtool_log_data["target"])[0:50]
    # linear regression (fit-predict)
    # model = linear_model.LinearRegression()
    # model.fit(x_train_ds, y_train_ds)
    # plot the trained results
    # x_plot = x_train_ds[0:100]
    # y_plot = y_train_ds[0:100]
    # plt.plot(np.arange(len(x_plot)), y_plot, color="blue")
    # plt.plot(np.arange(len(x_plot)), model.predict(x_plot), color="red")
    # plt.grid(True)
    # plt.show()


    """ 3. Using mlpack to train linear-reg model
    could work with pandas-data frame """
    # x_train_ds = (chamtool_log_data[col_names])[0:50]
    # y_train_ds = (chamtool_log_data["target"])[0:50]
    x_train_ds = train_ds[labels]
    y_train_ds = train_ds["target"]
    model = linear_regression(training=x_train_ds, training_responses=y_train_ds)
    mlpack_lr_model = model['output_model']


    """ Validate the predictor """

    # x_trained_plot = (chamtool_log_data[col_names])[800:900]
    # x_trained_plot = (chamtool_log_data["tid"])[800:900]
    # y_trained_plot = (chamtool_log_data["target"])[800:900]
    y_trained_plot = train_ds["target"]

    # x_predict_plot = (chamtool_log_data[col_names])[900:1000]
    # x_predict_plot = (chamtool_log_data["tid"])[900:1000]
    x_valid = valid_ds[labels]
    y_valid = valid_ds["target"]

    # y_predict_plot = model.predict(x_predict_plot) # with scikit-learn or torch-defined-model
    pred_model = linear_regression(input_model=mlpack_lr_model, test=x_valid)
    y_predict = pred_model['output_predictions']


    """ Ploting the validation results """
    s1_point = TRAIN_FROM + input_length
    e1_point = s1_point + len(y_trained_plot)
    s2_point = VALID_FROM + input_length
    e2_point = s2_point + len(y_predict)
    print("Ploting from {} - {} for trained points".format(s1_point, (e1_point-1)))
    print(np.arange(s1_point, e1_point))
    print("Ploting from {} - {} for validated points".format(s2_point, (e2_point-1)))
    print(np.arange(s2_point, e2_point))
    plt.plot(np.arange(s1_point, e1_point), y_trained_plot, color="green")
    plt.plot(np.arange(s2_point, e2_point), y_predict, color="red")
    plt.plot(np.arange(s2_point, e2_point), y_valid, color="blue")  # ground_truth
    plt.grid(True)
    plt.show()

    # ground_truth = (chamtool_log_data["target"])[900:1000]
    # print("Ground-truth:")
    # print(ground_truth)
    # print("Predicted-values:")
    # print(y_predict_plot)



#########################################################
###### Calling main
#########################################################
if __name__ == "__main__":
    main()