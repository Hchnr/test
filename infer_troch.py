import random
import time

import numpy as np
import torch
import tqdm
from transformers import AutoTokenizer, AutoModel
from modeling.modeling_torch import ErnieForCausalLM, ErnieModel


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


MODEL_PATH = "/share/project/hcr/models/wenxinyiyan/paddle_internal/ERNIE-45-Turbo"
MODEL_PATH = "/share/project/hcr/models/Qwen/Qwen3-0.6B"
NUM_PROMPTS = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True)

import pdb

pdb.set_trace()

# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = ErnieModel.from_pretrained(MODEL_PATH, trust_remote_code=True)
model.to(DEVICE)
model.eval()

text = "Replace me by any text you'd like."
text *= 64
# encoded_input = tokenizer(text, return_tensors="pt").to(DEVICE)
