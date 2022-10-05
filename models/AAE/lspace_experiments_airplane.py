import os
import time
import argparse
import torch
import json

with open('models/AAE/ckptselection.json') as f1:
    config1 = json.load(f1)
    

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

if config1['type']=="objects":
    from models.AAE.AAE_model import *
    # print("yes")
elif config1['type']=="parts":
    from models.AAE.AAE_model_parts import *

ckpt_path= "AAE_" + config1['category']+"_" + config1['type']

encoder_path=os.path.join("models/AAE/Trained_Models",ckpt_path,"E.pth")
gen_path=os.path.join("models/AAE/Trained_Models",ckpt_path,"G.pth")

#Latent Space for 512
with open('models/AAE/hyperparams_objects.json') as f:
    config = json.load(f)

def weights_init(m):
    classname = m.__class__.__name__
    if classname in ('Conv1d', 'Linear'):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

random.seed(config['seed'])
torch.manual_seed(config['seed'])
torch.cuda.manual_seed_all(config['seed'])

device = cuda_setup(config['cuda'], config['gpu'])


G_airplane = Generator(config).to(device)
E_airplane = Encoder(config).to(device)
D = Discriminator(config).to(device)

G_airplane.apply(weights_init)
E_airplane.apply(weights_init)
D.apply(weights_init)



EG_optim = getattr(optim, config['optimizer']['EG']['type'])
EG_optim = EG_optim(chain(E.parameters(), G.parameters()),
                    **config['optimizer']['EG']['hyperparams'])

D_optim = getattr(optim, config['optimizer']['D']['type'])
D_optim = D_optim(D.parameters(),
                  **config['optimizer']['D']['hyperparams'])

E_airplane.load_state_dict(torch.load(encoder_path))
G_airplane.load_state_dict(torch.load(gen_path))

E_airplane.eval()
G_airplane.eval()

print("model_load_succesful")

def encode(chair):
    # chair=pc_normalize(chair)
    av=np.reshape(chair, (config['z_size'],3,1)).T
    codes=torch.from_numpy(av).to('cuda')
    with torch.no_grad():
        X_rec = E_airplane(codes.float())
    return X_rec[0].cpu().numpy()

def reconstruct_from_code(code):
    av=np.reshape(code, (config['z_size'],1,1)).T
    codes=torch.from_numpy(av).to('cuda')
    with torch.no_grad():
        X_rec = G_airplane(codes.float()).data.cpu().numpy()
    return X_rec[0]