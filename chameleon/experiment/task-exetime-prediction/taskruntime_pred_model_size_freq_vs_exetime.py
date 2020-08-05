import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import imageio
import pandas as pd
import torch.optim as optim
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

torch.manual_seed(1)    # reproducible

# define task class
class Task:
    def __init__(self, id, arg_num, prob_size, freq, cpu_type, exe_time):
        self.id = id
        self.arg_num = arg_num
        self.prob_size = prob_size
        self.freq = freq
        self.cpu_type = cpu_type
        self.exe_time = exe_time

# define Linear Regression class
class LinearRegression(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(LinearRegression, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)  # hidden layer
        self.hidden1 = torch.nn.Linear(n_hidden, n_hidden)  # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)  # output layer

        torch.nn.init.xavier_uniform_(self.hidden.weight)
        torch.nn.init.zeros_(self.hidden.bias)
        torch.nn.init.xavier_uniform_(self.hidden1.weight)
        torch.nn.init.zeros_(self.hidden1.bias)

        torch.nn.init.xavier_uniform_(self.predict.weight)
        torch.nn.init.zeros_(self.predict.bias)


    def forward(self, x):
        x = torch.tanh(self.hidden(x))  # activation function for hidden layer
        x = torch.relu(self.hidden1(x))  # activation function for hidden layer
        x = self.predict(x)  # linear output
        return x

# main function
if __name__ == "__main__":
    # read logfile & create type of dataset for training
    logfile = "./logfile.txt"
    tensor_data = torch.tensor(pd.read_csv(logfile, sep=",", header=None).values, dtype=torch.float)
    x_sizefreq_train = tensor_data[0:700,2:4].view(700,2)   # 2 features (size, cpu_frequency)
    y_train = tensor_data[0:700,4].view(700,1)
    x_sizefreq_val = tensor_data[700:1000,2:4].view(300,2)
    y_val = tensor_data[700:1000,4].view(300,1)

    # normalize data for training
    min_size = min(x_sizefreq_train[:,0])
    max_size = max(x_sizefreq_train[:,0])
    min_freq = min(x_sizefreq_train[:,1])
    max_freq = max(x_sizefreq_train[:,1])
    y_min = min(y_train)
    y_max = max(y_train)
    norm_size = (x_sizefreq_train[:,0] - min_size) / (max_size - min_size) * 2 - 1
    norm_freq = (x_sizefreq_train[:,1] - min_freq) / (max_freq - min_freq) * 2 - 1
    norm_sizefreq = []
    for i in range(len(norm_size)):
        norm_sizefreq.append([norm_size[i].item(),norm_freq[i].item()])
    norm_y = (y_train - y_min) / (y_max - y_min)
    x, y = Variable(torch.tensor(norm_sizefreq)), Variable(norm_y)
    # for visualizing
    norm_sizefreq.sort()
    vis_sizefreq = Variable(torch.tensor(norm_sizefreq))
    vis_x = []
    vis_y = []
    for i in range(len(norm_sizefreq)):
        vis_x.append(norm_sizefreq[i][0])
        vis_y.append(norm_sizefreq[i][1])
    vis_x, vis_y = Variable(torch.tensor(vis_x)), Variable(torch.tensor(vis_y))

    # normalize data for validating
    norm_size_val = (x_sizefreq_val[:,0] - min_size) / (max_size - min_size) * 2 - 1
    norm_freq_val = (x_sizefreq_val[:,1] - min_freq) / (max_freq - min_freq) * 2 - 1
    norm_sizefreq_val = []
    for i in range(len(norm_size_val)):
        norm_sizefreq_val.append([norm_size_val[i].item(),norm_freq_val[i].item()])
    norm_y_val = (y_val - y_min) / (y_max - y_min)
    x_val, y_val = Variable(torch.tensor(norm_sizefreq_val)), Variable(norm_y_val)
    norm_sizefreq_val.sort()
    vis_sizefreq_val = Variable(torch.tensor(norm_sizefreq_val))
    vis_x_val = []  # sorted values
    vis_y_val = []  # sorted values
    for i in range(len(vis_sizefreq_val)):
        vis_x_val.append(norm_sizefreq_val[i][0])
        vis_y_val.append(norm_sizefreq_val[i][1])
    vis_x_val, vis_y_val = Variable(torch.tensor(vis_x_val)), Variable(torch.tensor(vis_y_val))

    # declare the network
    learning_rate = 0.01
    net_model = LinearRegression(n_features=2, n_hidden=20, n_output=1)
    loss_fn = torch.nn.MSELoss(reduction='mean')
    optimizer = optim.SGD(net_model.parameters(), lr=learning_rate)

    # for plotting images
    # my_images = []
    # fig, ax = plt.subplots(figsize=(12, 7))
    # fig = plt.figure(figsize=(20,15))
    # ax = plt.subplot(111, projection='3d')

    # training with visualizing the process
    n_epoch = 200
    net_model.train()
    with tqdm(
        total=n_epoch,
        dynamic_ncols=True,
        leave=False,
    ) as plot_running_bar:
        for epoch in range(n_epoch):
            prediction = net_model(x)
            loss = loss_fn(prediction, y)
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # plot and show learning process
            # plt.cla()
            # ax = fig.gca(projection='2d')
            # ax.set_title('Training Regression Analysis', fontsize=14)
            # ax.set_xlabel('Normalized Prob-Size [-1,1]', fontsize=12, labelpad=20)
            # ax.set_ylabel('Normalized Freq [-1,1]', fontsize=12, labelpad=20)
            # ax.set_ylabel('Normalized Exetime [0,1]', fontsize=12, labelpad=10)
            # ax.set_zlabel('Normalized Exetime [0,1]', fontsize=12)
            # ax.scatter3D(norm_size.data.numpy(), norm_freq.data.numpy(), y.data.numpy(), color="red")
            # ax.plot_surface(vis_x.data.numpy(), vis_y.data.numpy(), net_model(vis_sizefreq).data.numpy())
            # ax.scatter(norm_size.data.numpy(), y.data.numpy(), color="orange")
            # ax.plot(vis_x.data.numpy(), net_model(vis_sizefreq).data.numpy(), 'g-', lw=3)
            # ax.scatter(x[:,0].data.numpy(), x[:,1].data.numpy(), net_model(x).data.numpy(), marker=".", color="blue")
            # ax.text(.8, 0.1, 0, 'Step = %d' % epoch, fontdict={'size': 24, 'color': 'red'})
            # ax.text(.8, 0, 0, 'Loss = %.4f' % loss.data.numpy(), fontdict={'size': 24, 'color': 'red'})

            # Used to return the plot as an image array
            # (https://ndres.me/post/matplotlib-animated-gifs-easily/)
            # fig.canvas.draw()  # draw the canvas, cache the renderer
            # image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            # image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            # my_images.append(image)

            # show the running bar
            plot_running_bar.set_description("Epoch: {} - Loss: {}".format(epoch, loss.item()))
            plot_running_bar.update(1)

    # save images as a gif
    # imageio.mimsave('./visualized_train_model2_new.gif', my_images, fps=10)
    # plt.show()

    # net_model.eval()
    # test the model
    # val_res_fig = plt.figure()
    # val_res_ax = plt.subplot(111, projection='3d')
    # val_res_ax.scatter(norm_size_val.data.numpy(), norm_freq_val.data.numpy(), y_val.data.numpy(), marker="o")
    # val_res_ax.scatter(vis_sizefreq_val[:, 0].data.numpy(), vis_sizefreq_val[:, 1].data.numpy(), net_model(vis_sizefreq_val).data.numpy(), marker=".", color="red")
    # val_res_ax.plot(vis_x_val.data.numpy(), net_model(vis_sizefreq_val).data.numpy(), c='b')
    # val_res_ax.set_title('Validation Analysis', fontsize=20)
    # val_res_ax.set_xlabel('Norm Problem-Size', fontsize=16, labelpad=20)
    # val_res_ax.set_ylabel('Norm Freq', fontsize=16, labelpad=15)
    # val_res_ax.set_zlabel('Norm Exetime', fontsize=16)
    # plt.show()

    # Check the real result
    # print("y_max: %f \t | y_min: %f" % (y_max, y_min))
    # print("Task \t size \t\t Mhz \t\t pred_exetime (norm_value) \t real_exetime")
    # for i in range(10):
    #     norm_exetime = net_model(x_val[i]).item()
    #     if norm_exetime < 0:
    #         norm_exetime = norm_exetime * (-1)
    #     pred_exetime = norm_exetime * (y_max - y_min) + y_min
    #     size = (norm_size_val[i].item() + 1) / 2 * (max_size - min_size) + min_size
    #     freq = (norm_freq_val[i].item() + 1) / 2 * (max_freq - min_freq) + min_freq
    #     exetime = norm_y_val[i].item() * (y_max - y_min) + y_min
    #     check_statement = "%d \t %.5f \t %.5f \t %.5f ( %.5f ) \t %.5f" % (i, size, freq, pred_exetime, norm_exetime, exetime)
    #     print(check_statement)

    # save the model state (can only load by Python)
    my_model = net_model(0.5,0.5)
    # torch.save(net_model.state_dict(), 'task-exetime-pred.pth')

    # save the model state for C++
    # run traced module
    # traced_script_module = torch.jit.trace(net_model, vis_sizefreq_val)
    # save the converted model
    # traced_script_module.save("traced_task_exetime_pred.pt")