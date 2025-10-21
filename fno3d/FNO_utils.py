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



# import torch
# import numpy as np
# import torch.nn as nn
# import torch.nn.functional as F

# import matplotlib.pyplot as plt

# import operator
# from functools import reduce
# from functools import partial

# from timeit import default_timer

# # torch.manual_seed(0)
# # np.random.seed(0)


# ################################################################
# # 3d fourier layers
# ################################################################

# class SpectralConv3d(nn.Module):
#     def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
#         super(SpectralConv3d, self).__init__()

#         """
#         3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
#         """

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
#         self.modes2 = modes2
#         self.modes3 = modes3

#         self.scale = (1 / (in_channels * out_channels))
#         self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
#         self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
#         self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
#         self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

#     # Complex multiplication
#     def compl_mul3d(self, input, weights):
#         # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
#         return torch.einsum("bixyz,...ioxyz->boxyz", input, weights)

#     def forward(self, x):
#         batchsize = x.shape[0]
#         #Compute Fourier coeffcients up to factor of e^(- something constant)
#         x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

#         # Multiply relevant Fourier modes
#         out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
#         out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
#             self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
#         out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
#             self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
#         out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
#             self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
#         out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
#             self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

#         #Return to physical space
#         x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
#         return x

# class FNO3d(nn.Module):
#     def __init__(self, modes1, modes2, modes3, width):
#         super(FNO3d, self).__init__()

#         """
#         The overall network. It contains 4 layers of the Fourier layer.
#         1. Lift the input to the desire channel dimension by self.fc0 .
#         2. 4 layers of the integral operators u' = (W + K)(u).
#             W defined by self.W; K defined by self.conv .
#         3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
#         input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
#         input shape: (batchsize, x=64, y=64, t=40, c=13)
#         output: the solution of the next 40 timesteps
#         output shape: (batchsize, x=64, y=64, t=40, c=1)
#         """

#         self.modes1 = modes1
#         self.modes2 = modes2
#         self.modes3 = modes3
#         self.width = width
#         self.fc0 = nn.Linear(4, self.width)
#         # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

#         self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
#         self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
#         self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
#         self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
#         self.w0 = nn.Conv1d(self.width, self.width, 1)
#         self.w1 = nn.Conv1d(self.width, self.width, 1)
#         self.w2 = nn.Conv1d(self.width, self.width, 1)
#         self.w3 = nn.Conv1d(self.width, self.width, 1)
#         self.bn0 = torch.nn.BatchNorm3d(self.width)
#         self.bn1 = torch.nn.BatchNorm3d(self.width)
#         self.bn2 = torch.nn.BatchNorm3d(self.width)
#         self.bn3 = torch.nn.BatchNorm3d(self.width)

#         self.fc1 = nn.Linear(self.width, 128)
#         self.fc2 = nn.Linear(128, 1)

#     def forward(self, x):
#         batchsize = x.shape[0]
#         size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]

#         x = self.fc0(x)
#         x = x.permute(0, 4, 1, 2, 3)

#         x1 = self.conv0(x)
#         x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
#         x = x1 + x2
#         x = F.gelu(x)
#         x1 = self.conv1(x)
#         x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
#         x = x1 + x2
#         x = F.gelu(x)
#         x1 = self.conv2(x)
#         x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
#         x = x1 + x2
#         x = F.gelu(x)
#         x1 = self.conv3(x)
#         x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
#         x = x1 + x2

#         x = x.permute(0, 2, 3, 4, 1)
#         x = self.fc1(x)
#         x = F.gelu(x)
#         x = self.fc2(x)
#         return x

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

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv3d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv3d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class FNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width, input_size=4, output_size=1, layer=4):
        super(FNO3d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.layer = layer
        self.padding = 6 # pad the domain if input is non-periodic

        self.p = nn.Linear(input_size, self.width)# input channel: the solution of the first n timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)
        self.conv_layers = nn.ModuleList([SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3) for _ in range(self.layer)])
        self.mlp_layers = nn.ModuleList([MLP(self.width, self.width, self.width) for _ in range(self.layer)])
        self.w_layers = nn.ModuleList([nn.Conv3d(self.width, self.width, 1) for _ in range(self.layer)])
        self.q = MLP(self.width, output_size, self.width * 4) # output channel is 1: u(x, y)

    def forward(self, x):
        # grid = self.get_grid(x.shape, x.device)
        # x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        for i in range(self.layer):
            x1 = self.conv_layers[i](x)
            x1 = self.mlp_layers[i](x1)
            x2 = self.w_layers[i](x)
            x = x1 + x2
            if i < self.layer -1:
                x = F.gelu(x)

        x = x[..., :-self.padding] # pad the domain if input is non-periodic
        x = self.q(x)
        x = x.permute(0, 2, 3, 4, 1) 
        return x



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
    
# def load_data(main_dir,Ndata,batch_size,n_train_frame,response_type = 0,H_px = 128,W_px =128):
#     random_seed = 2024
#     H_0= 128  ## change this if original image res changes
#     res = H_0//H_px   ## for square grids ;default is 1

#     print("Loading Data....",flush=True)
#     frames = np.linspace(0,100,n_train_frame,dtype = int)
#     vel_data = np.load(main_dir+"output_25g_data.npy")
#     micro_data = np.load(main_dir+"micro_image_25g.npy")[:,:1,frames,::res,::res] ## loading only 1st frame for micro

#     T_max = micro_data.max()
#     micro_data = micro_data/T_max

#     V_impact = 300

#     if response_type == 0 :  ## 0 is velocity field 1 is PEEQ
#         response_data_array =vel_data[:,0:1,frames,::res,::res]/V_impact #np.zeros((vel_data.shape[0],1,32,H_px,W_px))
#     else:
#         response_data_array =vel_data[:,1:2,frames,::res,::res]


#     T = n_train_frame
#     #N = micro_data.shape[0]
#     N, channels, T, H, W = micro_data.shape
#     micro_data_aug = np.empty((2 * N, channels, T, H, W), dtype=micro_data.dtype)
#     # Place original images in even indices
#     micro_data_aug[0::2] = micro_data
#     # Place reflected images (flip along the last axis, i.e. width) in odd indices
#     micro_data_aug[1::2] = np.flip(micro_data, axis=-2)

#     N, channels, T, H, W = response_data_array.shape
#     resp_data_aug = np.empty((2 * N, channels, T, H, W), dtype=response_data_array.dtype)
#     # Place original images in even indices
#     resp_data_aug[0::2] = response_data_array
#     # Place reflected images (flip along the last axis, i.e. width) in odd indices
#     resp_data_aug[1::2] = np.flip(response_data_array, axis=-2)


    

#     # pad locations (x,y,t)
#     # Create grid for x-coordinates:
# # We want x_grid to be the same for every spatial location in H and for every timestep.
#     gridx = np.linspace(0, 1, W_px)              # shape: (W_px,)
#     gridx = gridx.reshape(1, 1, W_px, 1, 1)       # shape: (1, 1, W_px, 1, 1)
#     gridx = np.tile(gridx, (1, H_px, 1, T, 1))     # shape: (1, H_px, W_px, T, 1)

# # y_grid: values from 0 to 1 along the height axis.
#     gridy = np.linspace(0, 1, H_px)              # shape: (H_px,)
#     gridy = gridy.reshape(1, H_px, 1, 1, 1)       # shape: (1, H_px, 1, 1, 1)
#     gridy = np.tile(gridy, (1, 1, W_px, T, 1))     # shape: (1, H_px, W_px, T, 1)

# # t_grid: values from 0 to 1 along the time axis.
# # Here we create T+1 points and drop the first one (if desired) to get exactly T points.
#     gridt = np.linspace(0, 1, T + 1)[1:]      # shape: (T,)
#     gridt = gridt.reshape(1, 1, 1, T, 1)       # shape: (1, 1, 1, T, 1)
#     gridt = np.tile(gridt, (1, H_px, W_px, 1, 1))     # shape: (1, H_px, W_px, T, 1)

# # ----------------------------
# # Rearrange dimensions to match the target order
# # The model expects inputs in the order [N, channels, T, H_px, W_px].
# # Our grids are currently in shape [1, H_px, W_px, T, 1], so we need to transpose axes.
#     gridx = gridx.transpose(0, 4, 3, 1, 2)  # from (1, H_px, W_px, T, 1) to (1, 1, T, H_px, W_px)
#     gridy = gridy.transpose(0, 4, 3, 1, 2)  # (1, 1, T, H_px, W_px)
#     gridt = gridt.transpose(0, 4, 3, 1, 2)  # (1, 1, T, H_px, W_px)

# # ----------------------------
# # Repeat (tile) the grid arrays along the batch (N) dimension
# # Currently, each grid has shape (1, 1, T, H_px, W_px); we need them to have shape (N, 1, T, H_px, W_px)
#     gridx = np.tile(gridx, (2*N, 1, 1, 1, 1))  # shape: (N, 1, T, H_px, W_px)
#     gridy = np.tile(gridy, (2*N, 1, 1, 1, 1))  # shape: (N, 1, T, H_px, W_px)
#     gridt = np.tile(gridt, (2*N, 1, 1, 1, 1))  # shape: (N, 1, T, H_px, W_px)


#     micro_data1 = np.concatenate([micro_data_aug, gridx, gridy, gridt], axis=1)


#     print("final data shape: ",micro_data1.shape,resp_data_aug.shape)
#     plot(micro_data1[0,0,-1,:,:],resp_data_aug[0,0,-1,:,:],micro_data1[0,1,-1,:,:],micro_data1[0,2,-1,:,:],"/scratch4/lgraham1/Indrashish/Unet_time_march",)
#     plot(micro_data1[1,0,-1,:,:],resp_data_aug[1,0,-1,:,:],micro_data1[1,1,-1,:,:],micro_data1[1,2,-1,:,:],"/scratch4/lgraham1/Indrashish/Unet_time_march",filename="test1.png")




# #inp_train, inp_test, out_train, out_test = train_test_split( inp_scaled, out_scaled, test_size=0.25, random_state=random_seed)
   
#     inp_train, inp_test, out_train, out_test = train_test_split(micro_data1[:Ndata,:,:,:,:],resp_data_aug[:Ndata,:,:,:,:], test_size=0.1, random_state=random_seed)
    
#     # if response_type == 0:
#     #     out_train_max = np.max(out_train,axis=(0,2,3,4),keepdims=True)
#     #     out_train_min = np.min(out_train,axis = (0,2,3,4),keepdims=True)
#     #     out_train = (out_train-out_train_min)/(out_train_max-out_train_min)
#     #     out_test = (out_test-out_train_min)/(out_train_max-out_train_min)

#     #     scaling_params ={}
#     #     scaling_params['out_max'] = out_train_max
#     #     scaling_params['out_min'] = out_train_min
#     #     np.savez( "/scratch4/lgraham1/Indrashish/Unet_time_march/FNO_vel_scaling_params.npz", **scaling_params)


#     print(inp_train.shape,out_train.shape,inp_test.shape,out_test.shape)
#     train_dataset = CustomDataset(inputs=inp_train, outputs=out_train)
#     train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

#     test_dataset = CustomDataset(inputs=inp_test, outputs=out_test)
#     test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#     return train_dataloader,test_dataloader

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