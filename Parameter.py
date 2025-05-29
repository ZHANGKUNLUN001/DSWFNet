import torch


BATCH_SIZE= 2
NUM_WORKER= 2
FILE_NAME='Train'
LEARN_RATE=1e-4
WEIGHT_DECAY=1e-5
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# SET_NAME='Massachusetts_Roads_Dataset'
SET_NAME='CHN6-CUG'
PRE_EPOCH=0
START_EPOCH=0
END_EPOCH=100
sever_root=''