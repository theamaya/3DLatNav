import numpy as np
import torch
import argparse
# import matplotlib.pyplot as plt
from models.Diffusion.utils.dataset import *
from models.Diffusion.utils.misc import *
from models.Diffusion.utils.data import *
from models.Diffusion.models.autoencoder import *
# from evaluation import EMD_CD
# from latent_space import *
# from models.autoencoder import *

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='models/Diffusion/Trained_Models/DPM_chair_objects.pt')
parser.add_argument('--categories', type=str, default='chair')
parser.add_argument('--device', type=str, default='cuda')
# Datasets and loaders
parser.add_argument('--batch_size', type=int, default=1) #please update this to the number of pcs
parser.add_argument('--num_workers', type=int, default=0)
args = parser.parse_args("")

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

ckpt = torch.load(args.ckpt)
seed_all(ckpt['args'].seed)#test_loader=test_loader
model = AutoEncoder(ckpt['args']).to(args.device)
model.load_state_dict(ckpt['state_dict'])
model.to('cuda')
model.eval()

print("model_load_successful")

def encode(chair):
    #pc=np.expand_dims(chair,axis=0)
    pc=pc_normalize(chair)
    pc=torch.from_numpy(pc)
    ref = pc.to(args.device)
    shift = pc.mean(dim=0).reshape(1, 3)
    scale = pc.flatten().std().reshape(1, 1)
    shift = shift.to(args.device)
    scale = scale.to(args.device)
    ref=torch.unsqueeze(ref, axis=0)
    ref = (ref - shift) / scale
    model.eval()
    with torch.no_grad():
        code = model.encode(ref)
    code=code.detach().cpu().numpy()
    return(code)

def reconstruct_from_code(chair):
    chair= torch.from_numpy(chair.astype(np.float32)).to('cuda')
    #print(chair)
    recons = model.decode(chair,2048, flexibility=ckpt['args'].flexibility).detach().cpu().numpy()
    #recons = np.expand_dims(recons,0)
    return(recons[0].T)