import torch
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import torch.nn.functional as F
from torch.optim import AdamW
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm


############# 

##   Set the FNO 3D model

############



class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x
    
class U_net(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, dropout_rate):
        super(U_net, self).__init__()
        self.input_channels = input_channels
        self.conv1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2, dropout_rate = dropout_rate)
        self.conv2 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2, dropout_rate = dropout_rate)
        self.conv2_1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)
        self.conv3 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2, dropout_rate = dropout_rate)
        self.conv3_1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)
        
        self.deconv2 = self.deconv(input_channels, output_channels)
        self.deconv1 = self.deconv(input_channels*2, output_channels)
        self.deconv0 = self.deconv(input_channels*2, output_channels)
    
        self.output_layer = self.output(input_channels*2, output_channels, 
                                         kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)


    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_deconv2 = self.deconv2(out_conv3)
        concat2 = torch.cat((out_conv2, out_deconv2), 1)
        out_deconv1 = self.deconv1(concat2)
        concat1 = torch.cat((out_conv1, out_deconv1), 1)
        out_deconv0 = self.deconv0(concat1)
        concat0 = torch.cat((x, out_deconv0), 1)
        out = self.output_layer(concat0)

        return out

    def conv(self, in_planes, output_channels, kernel_size, stride, dropout_rate):
        return nn.Sequential(
            nn.Conv3d(in_planes, output_channels, kernel_size=kernel_size,
                      stride=stride, padding=(kernel_size - 1) // 2, bias = False),
            nn.BatchNorm3d(output_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate)
        )

    def deconv(self, input_channels, output_channels):
        return nn.Sequential(
            nn.ConvTranspose3d(input_channels, output_channels, kernel_size=4,
                               stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def output(self, input_channels, output_channels, kernel_size, stride, dropout_rate):
        return nn.Conv3d(input_channels, output_channels, kernel_size=kernel_size,
                         stride=stride, padding=(kernel_size - 1) // 2)

class SimpleBlock3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(SimpleBlock3d, self).__init__()
        """
        U-FNO contains 3 Fourier layers and 3 U-Fourier layers.
        
        input shape: (batchsize, x=200, y=96, t=24, c=12)
        output shape: (batchsize, x=200, y=96, t=24, c=1)
        """
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.fc0 = nn.Linear(4, self.width)
        """        
        12 channels for [kr, kz, porosity, inj_loc, inj_rate, 
                         pressure, temperature, Swi, Lam, 
                         grid_x, grid_y, grid_t]
        """
        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv4 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv5 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.w4 = nn.Conv1d(self.width, self.width, 1)
        self.w5 = nn.Conv1d(self.width, self.width, 1)
        self.unet3 = U_net(self.width, self.width, 3, 0)
        self.unet4 = U_net(self.width, self.width, 3, 0)
        self.unet5 = U_net(self.width, self.width, 3, 0)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]
        
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = x1 + x2 
        x = F.gelu(x)
        
        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = x1 + x2 
        x = F.gelu(x)
        
        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = x1 + x2 
        x = F.gelu(x)
        
        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x3 = self.unet3(x) 
        x = x1 + x2 + x3
        x = F.gelu(x)
        
        x1 = self.conv4(x)
        x2 = self.w4(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x3 = self.unet4(x)
        x = x1 + x2 + x3
        x = F.gelu(x)
        
        x1 = self.conv5(x)
        x2 = self.w5(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x3 = self.unet5(x)
        x = x1 + x2 + x3
        x = F.gelu(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        return x

class Net3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(Net3d, self).__init__()

        """
        A wrapper function
        """

        self.conv1 = SimpleBlock3d(modes1, modes2, modes3, width)


    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]
        x = F.pad(F.pad(x, (0,0,0,8,0,8), "replicate"), (0,0,0,0,0,0,0,8), 'constant', 0)
        x = self.conv1(x)
        x = x.view(batchsize, size_x+8, size_y+8, size_z+8, 1)[..., :-8,:-8,:-8, :]
        return x.squeeze()


    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c

## OTHER FUNCTIONS ##
# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001, time_last=True):
        super(UnitGaussianNormalizer, self).__init__()

        if isinstance(x, str):
            tensor_dict = torch.load(x)
            self.mean = tensor_dict['mean']
            self.std = tensor_dict['std']
        else:
            self.mean = torch.mean(x, dim = (0, 1))
            self.std = torch.std(x, dim = (0, 1))

        self.eps = eps
        self.time_last = time_last # if the time dimension is the last dim

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        # sample_idx is the spatial sampling mask
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if self.mean.ndim == sample_idx.ndim or self.time_last:
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if self.mean.ndim > sample_idx.ndim and not self.time_last:
                    std = self.std[...,sample_idx] + self.eps # T*batch*n
                    mean = self.mean[...,sample_idx]
        # x is in shape of batch*(spatial discretization size) or T*batch*(spatial discretization size)
        x = (x * std) + mean
        return x
    
    def to(self, device):
        if torch.is_tensor(self.mean):
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
        else:
            self.mean = torch.from_numpy(self.mean).to(device)
            self.std = torch.from_numpy(self.std).to(device)
        return self

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

    def SaveNormalizer(self, file_name):
        tensor_dict = {'mean': self.mean, 'std': self.std}

        torch.save(tensor_dict, file_name + '_normalizer.pt')
    


def set_seed(seed):
    """Sets the seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    np.random.seed(seed)
    random.seed(seed)
    # If you are using CuDNN, the below two lines can also help reproducibility
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False



class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p,
                                                          1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def forward(self, x, y):
        return self.rel(x, y)

    def __call__(self, x, y):
        return self.forward(x, y)
    

class CustomDataset(Dataset):
    def __init__(self, inputs, outputs):
        """
        Constructor for the Dataset.
        inputs: The input features, including both phi and epsilon_t+1.
        outputs: The target outputs.
        """
        self.inputs = inputs
        self.outputs = outputs


    def __len__(self):
        """Returns the total number of samples."""
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        Retrieves the sample at the given index.
        idx: Index of the sample to retrieve.
        """
        # Assuming that the last 75 values of inputs are phi and the first 75 are epsilon_t+1
        T = 360

        ## permute for FNO input
        taylor_factor = self.inputs[idx,:,:,:,:]
        
        vel = self.outputs[idx,:,:,:,:]


        # Convert them to tensors
        taylor_factor = torch.tensor(taylor_factor, dtype=torch.float32).permute(2,3,1,0)
        vel =torch.tensor(vel, dtype=torch.float32).permute(2,3,1,0)
        return taylor_factor,vel    
    
def plot(matrix1, matrix2, matrix3, matrix4,main_dir,titles=None, cmap='jet', save_dir='.', filename='test.png'):
    """
    Plots three images side by side with separate colorbars and saves the figure to a specified directory.

    Parameters:
    - matrix1, matrix2, matrix3: 2D arrays to be plotted as images
    - titles: List of titles for the three images (default is None)
    - cmap: Colormap to be used for the images (default is 'viridis')
    - save_dir: Directory where the figure will be saved (default is current directory)
    - filename: Name of the file to save the figure (default is 'three_images.png')
    """
    matrices = [matrix1, matrix2, matrix3,matrix4]
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))

    for i, (matrix, ax) in enumerate(zip(matrices, axs)):
        if i == 3:
            cax = ax.imshow(matrix, vmin = 0,cmap='inferno')
        else:
            cax = ax.imshow(matrix, cmap=cmap)
        fig.colorbar(cax, ax=ax)
    
    #fig.set_title("Epoch:"+str(titles))

    plt.tight_layout()
    
    # Save the figure
    save_path = main_dir+"/figures/"+str(filename)
    plt.savefig(save_path)
    plt.show()
    plt.close()
    
def load_data(main_dir,Ndata,batch_size,n_train_frame,split = 10,response_type = 0,H_px = 128,W_px =128):
    random_seed = 2024
    H_0= 128  ## change this if original image res changes
    res = H_0//H_px   ## for square grids ;default is 1

    print("Loading Data....",flush=True)
    frames = np.linspace(0,100,n_train_frame,dtype = int)
    
    vel_data = np.load(main_dir+"output_25g_data.npy")
    micro_data = np.load(main_dir+"micro_image_25g.npy")[:,:1,frames,::res,::res] ## loading only 1st frame for micro
  
    T_max = micro_data.max()
    micro_data = micro_data/T_max

    V_impact = 300

    if response_type == 0 :  ## 0 is velocity field 1 is PEEQ
        response_data_array =vel_data[:,0:1,frames,::res,::res]/V_impact #np.zeros((vel_data.shape[0],1,32,H_px,W_px))
    else:
        response_data_array =vel_data[:,1:2,frames,::res,::res]/2
    print(vel_data.shape,response_data_array.shape)
    ## splt into train-val 

    inp_train, inp_val, out_train, out_val = train_test_split(micro_data[:Ndata,:,:,:,:],response_data_array[:Ndata,:,:,:,:], test_size=split/100, random_state=random_seed)
    print(inp_train.shape,inp_val.shape)


    T = n_train_frame
    #N = micro_data.shape[0]
    N, channels, T, H, W = inp_train.shape
    inp_train_aug = np.empty((2 * N, channels, T, H, W), dtype=inp_train.dtype)
    # Place original images in even indices
    inp_train_aug[0::2] = inp_train
    # Place reflected images (flip along the last axis, i.e. width) in odd indices
    inp_train_aug[1::2] = np.flip(inp_train, axis=-2)

    N, channels, T, H, W = out_train.shape
    out_train_aug = np.empty((2 * N, 1, T, H, W), dtype=out_train.dtype)
    # Place original images in even indices
    out_train_aug[0::2] = out_train
    # Place reflected images (flip along the last axis, i.e. width) in odd indices
    out_train_aug[1::2] = np.flip(out_train, axis=-2)


    

    # pad locations (x,y,t)
    # Create grid for x-coordinates:
# We want x_grid to be the same for every spatial location in H and for every timestep.
    gridx = np.linspace(0, 1, W_px)              # shape: (W_px,)
    gridx = gridx.reshape(1, 1, W_px, 1, 1)       # shape: (1, 1, W_px, 1, 1)
    gridx = np.tile(gridx, (1, H_px, 1, T, 1))     # shape: (1, H_px, W_px, T, 1)

# y_grid: values from 0 to 1 along the height axis.
    gridy = np.linspace(0, 1, H_px)              # shape: (H_px,)
    gridy = gridy.reshape(1, H_px, 1, 1, 1)       # shape: (1, H_px, 1, 1, 1)
    gridy = np.tile(gridy, (1, 1, W_px, T, 1))     # shape: (1, H_px, W_px, T, 1)

# t_grid: values from 0 to 1 along the time axis.
# Here we create T+1 points and drop the first one (if desired) to get exactly T points.
    gridt = np.linspace(0, 1, T + 1)[1:]      # shape: (T,)
    gridt = gridt.reshape(1, 1, 1, T, 1)       # shape: (1, 1, 1, T, 1)
    gridt = np.tile(gridt, (1, H_px, W_px, 1, 1))     # shape: (1, H_px, W_px, T, 1)

# ----------------------------
# Rearrange dimensions to match the target order
# The model expects inputs in the order [N, channels, T, H_px, W_px].
# Our grids are currently in shape [1, H_px, W_px, T, 1], so we need to transpose axes.
    gridx = gridx.transpose(0, 4, 3, 1, 2)  # from (1, H_px, W_px, T, 1) to (1, 1, T, H_px, W_px)
    gridy = gridy.transpose(0, 4, 3, 1, 2)  # (1, 1, T, H_px, W_px)
    gridt = gridt.transpose(0, 4, 3, 1, 2)  # (1, 1, T, H_px, W_px)

# ----------------------------
# Repeat (tile) the grid arrays along the batch (N) dimension
# Currently, each grid has shape (1, 1, T, H_px, W_px); we need them to have shape (N, 1, T, H_px, W_px)
    gridx1 = np.tile(gridx, (inp_train_aug.shape[0], 1, 1, 1, 1))  # shape: (N, 1, T, H_px, W_px)
    gridy1 = np.tile(gridy, (inp_train_aug.shape[0], 1, 1, 1, 1))  # shape: (N, 1, T, H_px, W_px)
    gridt1 = np.tile(gridt, (inp_train_aug.shape[0], 1, 1, 1, 1))  # shape: (N, 1, T, H_px, W_px)

    inp_train1 = np.concatenate([inp_train_aug, gridx1, gridy1, gridt1], axis=1)


    gridx2 = np.tile(gridx, (inp_val.shape[0], 1, 1, 1, 1))  # shape: (N, 1, T, H_px, W_px)
    gridy2 = np.tile(gridy, (inp_val.shape[0], 1, 1, 1, 1))  # shape: (N, 1, T, H_px, W_px)
    gridt2 = np.tile(gridt, (inp_val.shape[0], 1, 1, 1, 1))  # shape: (N, 1, T, H_px, W_px)

    inp_val1 = np.concatenate([inp_val, gridx2, gridy2, gridt2], axis=1)

    print("final data shape: ",inp_train1.shape,out_train_aug.shape)

    plot(inp_train1[0,0,-1,:,:],out_train_aug[0,0,-1,:,:],inp_train1[0,1,-1,:,:],out_train_aug[0,0,-1,:,:],"/scratch4/lgraham1/Indrashish/Unet_time_march",)
    plot(inp_train1[1,0,-1,:,:],out_train_aug[1,0,-1,:,:],inp_train1[1,1,-1,:,:],out_train_aug[1,0,-1,:,:],"/scratch4/lgraham1/Indrashish/Unet_time_march",filename="test1.png")


    inp_test = micro_data[Ndata:]

    gridx3 = np.tile(gridx, (inp_test.shape[0], 1, 1, 1, 1))  # shape: (N, 1, T, H_px, W_px)
    gridy3 = np.tile(gridy, (inp_test.shape[0], 1, 1, 1, 1))  # shape: (N, 1, T, H_px, W_px)
    gridt3 = np.tile(gridt, (inp_test.shape[0], 1, 1, 1, 1))  # shape: (N, 1, T, H_px, W_px)

    inp_test1 = np.concatenate([inp_test, gridx3, gridy3, gridt3], axis=1)
    out_test = response_data_array[Ndata:]

    print(inp_train1.shape,out_train_aug.shape,inp_val.shape,out_val.shape,inp_test.shape,out_test.shape)

    train_dataset = CustomDataset(inputs=inp_train1, outputs=out_train_aug)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = CustomDataset(inputs = inp_val1,outputs=out_val)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = CustomDataset(inputs=inp_test1, outputs=out_test)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)


    return train_dataloader,val_dataloader,test_dataloader