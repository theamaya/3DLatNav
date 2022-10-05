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
    print("yes")
elif config1['type']=="parts":
    from models.AAE.AAE_model_parts import *

ckpt_path= "AAE_" + config1['category']+"_" + config1['type']

encoder_path=os.path.join("models/AAE/Trained_Models",ckpt_path,"E.pth")
gen_path=os.path.join("models/AAE/Trained_Models",ckpt_path,"G.pth")

E.load_state_dict(torch.load(encoder_path))
G.load_state_dict(torch.load(gen_path))

E.eval()
G.eval()

print("model_load_succesful")

def encode(chair):
    chair=pc_normalize(chair)
    av=np.reshape(chair, (config['z_size'],3,1)).T
    codes=torch.from_numpy(av).to('cuda')
    with torch.no_grad():
        X_rec = E(codes.float())
    return X_rec[0].cpu().numpy()

def reconstruct_from_code(code):
    av=np.reshape(code, (config['z_size'],1,1)).T
    codes=torch.from_numpy(av).to('cuda')
    with torch.no_grad():
        X_rec = G(codes.float()).data.cpu().numpy()
    return X_rec[0]