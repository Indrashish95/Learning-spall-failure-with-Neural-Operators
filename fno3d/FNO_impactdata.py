import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse

from FNO_utils import FNO3d, CustomDataset, LpLoss,set_seed,load_data,plot

from timeit import default_timer
from torch.optim import AdamW
import torch.nn as nn
from tqdm import tqdm
from matplotlib import pyplot as plt

## Setting Seed
random_seed=2025
def report_all_parameters(model):
    print(f"{'Name':40s} | {'Trainable':9s} | {'Shape':20s} | {'#Elements':>10s} | {'ElemSize (B)':>12s} | {'TotalSize (B)':>14s}")
    print("-" * 110)
    
    total_params = 0
    total_bytes = 0
    
    for name, p in model.named_parameters():
        numel = p.numel()
        elem_size = p.element_size()
        total_size = numel * elem_size
        trainable = "Yes" if p.requires_grad else "No"
        
        print(f"{name:40s} | {trainable:9s} | {str(tuple(p.shape)):20s} | {numel:10,d} | {elem_size:12,d} | {total_size:14,d}")
        
        total_params += numel
        total_bytes += total_size

    print("-" * 110)
    print(f"{'Total':40s} | {'':9s} | {'':20s} | {total_params:10,d} | {'':12s} | {total_bytes:14,d}")
    print(f"Total size in MB: {total_bytes / (1024**2):.2f} MB")
# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--split", type=int, default=10,
#     help="Which split to run (e.g. 10,20,30,40)"
# )
# args = parser.parse_args()

split  = 10 #args.split
set_seed(seed=random_seed)

# batch_size = 10
# batch_size2 = batch_size

## Data loading

Ndata = 800 # number of training samples

data_dir = "./data/"
main_dir = "./" 

batch_size = 5
H_px = 128
W_px = 128
resp_type = 0 ## 0 velocity 1 plasticity
n_train_frame = 32

#train_dataloader,test_dataloader = load_data(data_dir,Ndata,batch_size,n_train_frame,response_type=resp_type ,H_px=H_px,W_px=W_px)

train_dataloader,val_dataloader,test_dataloader = load_data(data_dir,Ndata,batch_size,n_train_frame,split = split,response_type=resp_type ,H_px=H_px,W_px=W_px)


## model hyperparams and definition
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

modes1 = 40
modes2 = 40
modes3 = 16

width = 40
model = FNO3d(modes1, modes2, modes3, width).to(device)
### calculate number of paramters
report_all_parameters(model)
# if torch.cuda.device_count() > 1:
#         print(f"Using {torch.cuda.device_count()} GPUs")
#         model = nn.DataParallel(model)
# exit()
## Optimizer parameters

learning_rate = 0.001
epochs = 250
gamma = 0.5
scheduler_step =50
scheduler_gamma = 0.9

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)


## loss function
loss_f = LpLoss(size_average=False) 
## TRAINING LOOP
train_err = np.zeros((epochs,))
val_err = np.zeros((epochs,))
best_loss = 1e9
test_eval_freq = 50
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    test_loss = 0
    val_loss = 0
    tr_mem = 0
    val_mem = 0
    for x, y in tqdm(train_dataloader):
        x, y = x.to(device), y.to(device)
        #print(x.shape,y.shape,flush = True)
        optimizer.zero_grad()
        torch.cuda.reset_peak_memory_stats()
        out = model(x)
        peak_memory = torch.cuda.max_memory_allocated()
        tr_mem += peak_memory/(1024**2)
        # print(x.shape,y.shape,out.shape)
        l2 =  loss_f(out.reshape(out.shape[0], -1), y.reshape(out.shape[0], -1))

        l2.backward()

        optimizer.step()
        # train_mse += mse.item()
        train_l2 += l2.item()

    # np.save(main_dir+"train_data/"+f"train_FNO_micro25g1_{resp_type}",x.cpu().numpy())   
    # np.save(main_dir+"train_data/"+f"train_FNO_true25g1_{resp_type}",y.cpu().numpy())  
    # np.save(main_dir+"train_data/"+f"train_FNO_pred25g1_{resp_type}",out.detach().cpu().numpy())  

    scheduler.step()
    train_err[ep] = train_l2/len(train_dataloader)
    tr_mem = tr_mem/len(train_dataloader)


    for x_val,y_val in val_dataloader:
            x_val,y_val = x_val.to(device),y_val.to(device)
            model.eval()
            torch.cuda.reset_peak_memory_stats()
            out_val = model(x_val)  
            peak_memory = torch.cuda.max_memory_allocated()
            # print(x_val.shape,y_val.shape,out_val.shape,flush = True)
            t_loss = loss_f(out_val.reshape(x_val.shape[0], -1), y_val.reshape(x_val.shape[0], -1))
            val_loss = val_loss+t_loss.item()
            val_mem +=peak_memory/(1024**2)
    # print("test loss calculated:",test_loss,flush=True)

    val_err[ep]=val_loss/len(val_dataloader)
    val_mem = val_mem/len(val_dataloader)
    # time_after_epoch = time.time()     ## time at the end of epoch
    # total_time = time_after_epoch - start_time
    # if (ep+1)%100 == 0:
    #     error = abs(y[0,:,:,-1,0].cpu().numpy()-out[0,:,:,-1].detach().cpu().numpy())
    #     plot(x[0,:,:,-1,0].cpu().numpy(),y[0,:,:,-1,0].cpu().numpy(),out[0,:,:,-1].detach().cpu().numpy(),error,main_dir,titles = ep,filename=f"Epoch_{ep}_UFNOimpacttrain_{resp_type}.png")
    print(f'Epoch {ep + 1}| Train Loss: {train_err[ep]:.4f}| val Loss: {val_err[ep]:.4f} | train_memory : {tr_mem} | val_memory: {val_mem}',flush=True)


    if val_err[ep] < best_loss:
            best_loss = val_err[ep]
            print("Test Loss improved. Saving model ........",flush=True)
            torch.save(model.state_dict(), main_dir+"saved_model/"+f"FNO3d-1shot-25g1-{resp_type}_{split}_modes40.pth")
            eps = 0.1

            # error = abs(y[0,:,:,-1,0].cpu().numpy()-out[0,:,:,-1,0].detach().cpu().numpy())

            # plot(x[0,:,:,-1,0].cpu().numpy(),y[0,:,:,-1,0].cpu().numpy(),out[0,:,:,-1,0].detach().cpu().numpy(),error,main_dir,titles = ep,filename=f"Epoch_{ep}_UFNOimpacttrain_{resp_type}.png")
            
            error = abs(y_val[0,:,:,-1,0].cpu().numpy()-out_val[0,:,:,-1,0].detach().cpu().numpy())

            # plot(x_val[0,:,:,-1,0].cpu().numpy(),y_val[0,:,:,-1,0].cpu().numpy(),out_val[0,:,:,-1,0].detach().cpu().numpy(),error,main_dir,titles = ep,filename=f"Epoch_{ep}_FNOimpact_val_{resp_type}_{split}40.png")

            # np.save(main_dir+"test_data/"+f"val_FNO_micro25g2_{resp_type}_{split}",x_val.cpu().numpy())   
            # np.save(main_dir+"test_data/"+f"val_FNO_true25g2_{resp_type}_{split}",y_val.cpu().numpy())  
            # np.save(main_dir+"test_data/"+f"val_FNO_pred25g2_{resp_type}_{split}",out_val.detach().cpu().numpy()) 
    
    np.savetxt(main_dir+f"FNO3d-train_loss2_{resp_type}_{split}",train_err)
    np.savetxt(main_dir+f"FNO3d-test_loss2_{resp_type}_{split}",val_err)

    plt.plot(train_err[:ep],label="Train Loss")
    plt.plot(val_err[:ep],label = "Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale('log')
        #plt.xscale('log')
    plt.xlim(-10,epochs+10)
    plt.legend(loc="upper right")
    plt.savefig(main_dir+f"FNO3d_loss25g2_{resp_type}_{split}.png")
    plt.close()
     
test_loss = 0
for x_test,y_test in test_dataloader:
    x_test,y_test = x_test.to(device),y_test.to(device)
    model.eval()
    out_test = model(x_test)  
            # print(out_test.shape,y_test.shape,flush = True)
    t_loss = loss_f(out_test.reshape(x_test.shape[0], -1), y_test.reshape(x_test.shape[0], -1))
    test_loss = test_loss+t_loss.item()

test_loss = test_loss/len(test_dataloader)

print(f"Test loss for split = {split/100}:",test_loss,flush=True)
eps = 0.1
error = abs(y_test[0,:,:,-1,0].cpu().numpy()-out_test[0,:,:,-1].detach().cpu().numpy())
plot(x_test[0,:,:,-1,0].cpu().numpy(),y_test[0,:,:,-1,0].cpu().numpy(),out_test[0,:,:,-1].detach().cpu().numpy(),
     error,main_dir,titles = ep,filename=f"FNOimpacttest_{resp_type}_{split}.png")
np.save(main_dir+"test_data/"+f"test_FNO_micro25g2_{resp_type}_{split}",x_test.cpu().numpy())   
np.save(main_dir+"test_data/"+f"test_FNO_true25g2_{resp_type}_{split}",y_test.cpu().numpy())  
np.save(main_dir+"test_data/"+f"test_FNO_pred25g2_{resp_type}_{split}",out_test.detach().cpu().numpy()) 
    
      




    







    



