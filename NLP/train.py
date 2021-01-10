import yaml
import torch
import numpy as np
import os
import random
import pandas as pd
import string

from trainer import Trainer
from datasets.base_dataset import BaseDataset, ArticleDataset
from utils import get_embedder, get_common_names
from pytorch_metric_learning import losses, miners, samplers


# from brats_competition.model_training.common.augmentations import get_transforms

# random.seed(42)
# np.random.seed(42)
# torch.manual_seed(42)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

with open(os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')) as config_file:
    config = yaml.full_load(config_file)

# train_transform = get_transforms(config['train']['transform'])
# val_transform = get_transforms(config['val']['transform'])

base_ds = BaseDataset(config['data']['article_length'], embed_model=get_embedder(config['data']['embed_model']), names=get_common_names(config['data']['dblp_path']))

train_ds = ArticleDataset(base_ds, train=True)
val_ds = ArticleDataset(base_ds, train=False)
num_triplets = config['num_triplets']
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, num_workers=12, sampler=samplers.FixedSetOfTriplets(val_ds.base_ind(), num_triplets))


train_dl = torch.utils.data.DataLoader(train_ds, batch_size=config['batch_size'], shuffle=False, num_workers=12, sampler=samplers.FixedSetOfTriplets(train_ds.base_ind(), num_triplets))

# train_dl = torch.utils.data.DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=12)
# val_dl = torch.utils.data.DataLoader(val_ds, batch_size=config['batch_size'], shuffle=True, num_workers=12)

trainer = Trainer(config, train_dl, val_dl)
trainer.train()