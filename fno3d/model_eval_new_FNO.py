import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from FNO_utils import FNO3d, set_seed
from torch.optim import AdamW
import torch.nn as nn
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm
from matplotlib import pyplot as plt

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

def load_data(main_dir,Ndata,batch_size,n_train_frame,split = 10,response_type = 0,H_px = 128,W_px =128):
    random_seed = 2024
    H_0= 128  ## change this if original image res changes
    res = H_0//H_px   ## for square grids ;default is 1

    print("Loading Data....",flush=True)
    frames = np.linspace(0,100,n_train_frame,dtype = int)
    
    vel_data = np.load(main_dir+"output_50g_data.npy")
    micro_data = np.load(main_dir+"micro_image_50g.npy")[:,:1,frames,::res,::res] ## loading only 1st frame for micro
  
    T_max = micro_data.max()
    micro_data = micro_data/T_max

    V_impact = 300

    if response_type == 0 :  ## 0 is velocity field 1 is PEEQ
        response_data_array =vel_data[:,0:1,frames,::res,::res]/V_impact #np.zeros((vel_data.shape[0],1,32,H_px,W_px))
    else:
        response_data_array =vel_data[:,1:2,frames,::res,::res]/2
    #print(vel_data.shape,response_data_array.shape)
    ## splt into train-val 



    T = n_train_frame
    #N = micro_data.shape[0]
    N, channels, T, H, W = micro_data.shape


    N, channels, T, H, W = response_data_array.shape


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
    gridx1 = np.tile(gridx, (micro_data.shape[0], 1, 1, 1, 1))  # shape: (N, 1, T, H_px, W_px)
    gridy1 = np.tile(gridy, (micro_data.shape[0], 1, 1, 1, 1))  # shape: (N, 1, T, H_px, W_px)
    gridt1 = np.tile(gridt, (micro_data.shape[0], 1, 1, 1, 1))  # shape: (N, 1, T, H_px, W_px)

    inp_train1 = np.concatenate([micro_data, gridx1, gridy1, gridt1], axis=1)


    print(inp_train1.shape,response_data_array.shape)


    test_dataset = CustomDataset(inputs=inp_train1, outputs=response_data_array)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)


    return test_dataloader




# ── Settings ────────────────────────────────────────────────────────────────
random_seed   = 2025
split         = 10      # 10,20,30,40
set_seed(seed=random_seed)

data_dir      = "/scratch4/lgraham1/Indrashish/Plate_Impact_Data/New_microstructures/200_grains/grid_data/"
main_dir      = "/scratch4/lgraham1/Indrashish/Unet_time_march/"
batch_size    = 5
H_px, W_px    = 128, 128
resp_type     = 0        # 0 = velocity, 1 = plasticity
n_train_frame = 32
# ────────────────────────────────────────────────────────────────────────────

# Data
test_dataloader = load_data(
    data_dir, Ndata=750, batch_size=batch_size,
    n_train_frame=n_train_frame,
    split=split,
    response_type=resp_type,
    H_px=H_px, W_px=W_px,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modes = 16
width = 40
# Model
model = FNO3d(modes, modes, modes, width).to(device)
model.load_state_dict(torch.load(
    main_dir + f"saved_model/FNO3d-1shot-25g-{resp_type}_{split}.pth",map_location=device
))
model.eval()

loss_f = nn.MSELoss()

# Pre-allocate output array
# dims: [n_batches, 3 (x,y,pred), depth=32, height=128, width=128]
test_data = torch.zeros(
    len(test_dataloader), 3, 32, 128, 128,
    dtype=torch.float32, device=device
)

total_loss = 0.0
t = np.linspace(0,100,32)

with torch.no_grad():
    for i, (x_test, y_test) in enumerate(test_dataloader):
        x_test = x_test.to(device)   # e.g. [B, C, D, H, W]
        y_test = y_test.to(device)

        start_time = time.time()
        out_test = model(x_test)
        end_time = time.time()     # same shape as x_test/y_test
        b = x_test.shape[0]
        t_loss = loss_f(
            out_test.reshape(b, -1),
            y_test.reshape(b, -1),
        )
        total_time = end_time-start_time

        total_loss += t_loss.item()
        print(x_test.shape,y_test.shape,out_test.shape)
        # take the first sample of the batch, channel 0
        x0 = x_test[0,:,:,:,0]    # shape [D, H, W]
        y0 = y_test[0,:,:,:,0] 
        o0 = out_test[0,:,:,:,0]

        # permute if needed (here: (D,H,W)->(W,D,H))
        x0 = x0.permute(2, 0, 1)
        y0 = y0.permute(2, 0, 1)
        o0 = o0.permute(2, 0, 1)

        test_data[i, 0] = x0
        test_data[i, 1] = y0
        test_data[i, 2] = o0
        if i ==0: 
            error  =  abs(test_data[0,1,-1,:,:].cpu().numpy()*300-test_data[0,2,-1,:,:].cpu().numpy()*300)/np.max(test_data[0,1,-1,:,:].cpu().numpy()*300)
            plt.imshow(error, vmin = 0,vmax = 0.2)
            plt.colorbar()
            plt.savefig(main_dir+"figures/fno50g_2_error.jpg",dpi = 300)
            plt.close()
            plt.imshow(test_data[0,0,-1,:,:].cpu().numpy(),cmap = "coolwarm")
            plt.colorbar()
            plt.savefig(main_dir+"figures/fno50g_2_micro.jpg",dpi = 300)
            plt.close()
            plt.imshow(test_data[0,1,-1,:,:].cpu().numpy(),cmap = "coolwarm")
            plt.colorbar()
            plt.savefig(main_dir+"figures/fno50g_2_true.jpg",dpi = 300)
            plt.close()
            plt.plot(t,np.mean(test_data[0,1,:,54:75,127].cpu().numpy(),axis = 1)*300,label = "True 50g",color ="green",marker='x')
            plt.plot(t,np.mean(test_data[0,2,:,54:75,127].cpu().numpy(),axis = 1)*300,label = "Predicted 50g",color = "green",ls ="--",marker = 'x')
            plt.savefig(main_dir+"figures/fno50g_2_freesurf_vel.jpg",dpi = 300)
            plt.close()
        print(f"[Batch {i+1}/{len(test_dataloader)}] loss = {t_loss.item():.4f}| time = {total_time}",flush = True)

avg_test_loss = total_loss / len(test_dataloader)
print(f"Mean Test loss for split {split}%: {avg_test_loss:.4f}")

# Save
np.save(
    main_dir + f"test_data/test_FNO_all50g_{resp_type}_{split}.npy",
    test_data.cpu().numpy()
)
