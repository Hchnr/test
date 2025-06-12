import random
import time

import numpy as np
import torch
import tqdm
from transformers import AutoModel, AutoTokenizer
from modeling import Ernie45TModel


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


MODEL_PATH = "/share/project/hcr/models/wenxinyiyan/paddle_internal/ERNIE-45-Turbo"
NUM_PROMPTS = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = Ernie45TModel.from_pretrained(
    MODEL_PATH, trust_remote_code=True, attn_implementation="eager"
)
model.to(DEVICE)
model.eval()

import pdb

pdb.set_trace()

text = "Replace me by any text you'd like."
text *= 64
encoded_input = tokenizer(text, return_tensors="pt").to(DEVICE)
