# coding:utf-8
from GA import GA
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import scipy.io as scio
import time
import os
import h5py
import math
import Test_Functions as fun
from GA import GA
from matplotlib.pyplot import MultipleLocator

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.set_default_dtype(torch.float32)
torch.manual_seed(66)
np.random.seed(66)

class CNN(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, input_kernel_size,input_stride, input_padding):

        super(CNN, self).__init__()

        # input channels of layer includes input_channels and hidden_channels of cells
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.input_kernel_size = input_kernel_size
        self.input_stride = input_stride
        self.input_padding = input_padding

        self.Wh1_x = nn.Conv2d(in_channels=self.input_channels, out_channels=self.input_channels*8, kernel_size=3,
                               stride=self.input_stride, padding=1, bias=True, )
        self.Wh2_x = nn.Conv2d(in_channels=self.input_channels, out_channels=self.input_channels*8, kernel_size=3,
                               stride=self.input_stride, padding=1, bias=True, )
        self.Wh3_x = nn.Conv2d(in_channels=self.input_channels, out_channels=self.input_channels*8, kernel_size=3,
                               stride=self.input_stride, padding=1, bias=True, )
        self.Wh4_x = nn.Conv2d(in_channels=self.input_channels*8, out_channels=1, kernel_size=1,
                               stride=1, padding=0, bias=True)
        self.filter_list = [self.Wh1_x, self.Wh2_x, self.Wh3_x, self.Wh4_x]

        self.init_filter(self.filter_list, c=0.5)
    def forward(self,h):
        f =self.Wh4_x(self.Wh1_x(h) * self.Wh2_x(h) * self.Wh3_x(h))
        for i in range(h.shape[0]):
            for k in range(h.shape[2]):
                for l in range(h.shape[3]):
                    for j in range(h.shape[1]-1):
                        # f[i][0][k][l]+=(50*torch.square(h[i][j+1][k][l]-torch.square(h[i][j][k][l])))
                        f[i][0][k][l] += (10*torch.square(h[i][j+1][k][l]))
        return  f


    def init_filter(self, filter_list, c):
        '''
        :param filter_list: list of filter for initialization
        :param c: constant multiplied on Xavier initialization
        '''

        for filter in filter_list:
            filter.weight.data.uniform_(-c * np.sqrt(1 / np.prod(filter.weight.shape[:-1])),
                                        c * np.sqrt(1 / np.prod(filter.weight.shape[:-1])))
            if filter.bias is not None:
                filter.bias.data.fill_(0.0)
def train(model, input,truth, n_iters, learning_rate, restart):
    best_loss = 10000
    if restart:
        model, optimizer, scheduler = load_model(model)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.97)
    train_loss=[]
    val_loss = []
    for epoch in range(n_iters):
        optimizer.zero_grad()
        pred= model(input)
        mse_loss = nn.MSELoss()
        pred, gt = pred[:, :, :, :], truth[:, :, :, :].cuda()
        idx = int(pred.shape[0] * 0.91)
        pred_tra, pred_val = pred[:idx], pred[idx:]  # prediction
        gt_tra, gt_val = gt[:idx], gt[idx:]  # ground truth
        loss_tra = mse_loss(pred_tra, gt_tra)
        # loss_tra = mse_loss(pred[:pred.shape[0]],gt[:pred.shape[0]])
        loss_val = mse_loss(pred_val, gt_val)
        loss = loss_tra
        loss.backward(retain_graph=True)
        train_loss.append(loss_tra)
        val_loss.append(loss_val)
        optimizer.step()
        scheduler.step()
        print('[%d/%d %d%%] tra: %.4f val: %.4f' % (
        (epoch + 1), n_iters, ((epoch + 1) / n_iters * 100.0),math.sqrt(loss_tra),math.sqrt(loss_val)))

        if loss_val < best_loss and epoch % 5 == 0:
            best_loss = loss_val
            for param_group in optimizer.param_groups:
                print(param_group['lr'])
            print('save model!!!')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, './model/checkpoint.pt')
    return train_loss ,val_loss

def save_model(model, model_name, save_path):
    ''' save the model '''
    torch.save(model.state_dict(), save_path + model_name + '.pt')


def load_model(model):
    # Load model and optimizer state
    checkpoint = torch.load('./model/checkpoint.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(model.parameters(), lr=0.0)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.985)
    return model, optimizer, scheduler

if __name__ == '__main__':
    with h5py.File("./datas/data", 'r') as file:  #load train data(x and f(x))
        x = file.get('x_10dim')[:]
        f = file.get('y1_rosenbrock')[:]

    n_iters =200
    learning_rate = 2e-3
    save_path = './model/'
    x = torch.from_numpy(x).float()
    f = torch.from_numpy(f).float()
    input = x.cuda()

    model = CNN(input_channels=x.shape[1], hidden_channels=4, output_channels=1, input_kernel_size=3,
                input_stride=1, input_padding=1).cuda()

    # train the model
    start = time.time()
    train_loss_list ,val_loss_list= train(model, input, f, n_iters, learning_rate, restart=False)

     dimension = x.shape[1]
     fun = fun.rastrigin
     lower_bound = 0
     upper_bound = 1
     max_iter = 100
     pop_size = 100
     result=np.zeros(20)
     ga = GA(pop_size, dimension=dimension, lower_bound=lower_bound, upper_bound=upper_bound)
     model, optimizer, scheduler = load_model(model)
      for dependent in range(20):
     ga.init_Population()
     for iter in range(max_iter):
         ga.crossover(ga.pc)
         ga.mutation(ga.pm)
         ga.pop = np.unique(ga.pop, axis=0)
         x_pop = np.zeros((int((ga.pop.shape[0]+1)/100)+1)*pop_size*ga.pop.shape[1]).reshape(int((ga.pop.shape[0]+1)/100)+1, ga.pop.shape[1], 10, 10)
         for i in range(x_pop.shape[0]):
             for k in range(10):
                 for l in range(10):
                     for j in range(x_pop.shape[1]):
                         if i * 100 + k * 10 + l < ga.pop.shape[0]:
                             x_pop[i][j][k][l] = ga.pop[i * 100 + k * 10 + l][j]
         x_pop=torch.from_numpy(x_pop).float().cuda()
    
         pred = model(x_pop)
         fit_value=pred.flatten()
         fit_value=fit_value[:ga.pop.shape[0]]
         fit_value=fit_value.cpu().detach().numpy()
          fit_value=fun(ga.pop)
         ga.selection(fit_value)
         fit_value=[]
     optimum = ga.first[-1]
     result[dependent]=fun(optimum)
     print('Optimal value :', fun(optimum ),dependent)
    
     end = time.time()
     result=np.array(result)
     print('mean: %f deviation: %14f ' %(np.mean(result),np.std(result)))


    iter = np.linspace(0, n_iters)
    rmse_tra=list(map(lambda num:math.sqrt(num),train_loss_list))
    rmse_val=list(map(lambda num:math.sqrt(num),val_loss_list))
    y_major_locator = MultipleLocator(20)
    ax=plt.gca()
    ax.yaxis.set_major_locator(y_major_locator)
    plt.ylim(0,200)
    plt.plot( rmse_tra, 'b',label='Train')
    plt.plot( rmse_val, ':r',label='Valid')
    #plt.title("RUNOOB TEST TITLE")
    plt.xlabel("Iter")
    plt.ylabel("Root Mean Square Error")
    plt.legend()
    # plt.show()
    fig_save_path = './figures/'
    plt.savefig(fig_save_path + 'loss_add_weak.jpg', dpi=300)

