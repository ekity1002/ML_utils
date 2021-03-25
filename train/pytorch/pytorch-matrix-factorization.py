#!/usr/bin/env python
# coding: utf-8

# In[22]:


import os
import csv
import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

DATA_DIR = '/kaggle/input/movielens-100k-dataset/ml-100k'
OUTPUT_DIR = './'

class Config:
    device='cpu'
    epochs=40
    seed=17
    train_bs=8
    valid_bs=8
    embedding_dim=20
    lr=1e-2
    num_workers=None       
    verbose_step=100
    
def torch_seed_everything(seed_value=777):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

config=Config()
torch_seed_everything(config.seed)


# # load data

# In[23]:


df = pd.read_csv(os.path.join(DATA_DIR, 'u.data'), sep='\t', header=None)
df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
#df = df.sort_values('timestamp').reset_index(drop=True)
n_user = df.user_id.nunique()
n_item = df.item_id.nunique()
df


# In[24]:


print('user_num', n_user)
print('item_num', n_item)


# # split data

# In[25]:


train_df, valid_df = train_test_split(df, test_size=0.2, stratify=df['user_id'], random_state=config.seed)
assert train_df.user_id.nunique() == valid_df.user_id.nunique()
print(train_df.shape, valid_df.shape)
#print(valid_df.user_id.nunique())


# # Dataset

# In[26]:


class MovieLensDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        user_id, item_id, rating, _ = self.df.iloc[idx]
        # index starts with 0
        sample = {"user": user_id - 1, "item": item_id - 1, "rating": rating}
        return sample


# # model

# In[27]:


class MatrixFactorizationPyTorch(nn.Module):
    def __init__(self, n_user, n_item, k=20):
        """
        n_user: user num
        n_item: item num
        k: embedding dim
        """
        super().__init__()
        self.user_factors = nn.Embedding(n_user, k, sparse=True)
        self.item_factors = nn.Embedding(n_item, k, sparse=True)

    def forward(self, user, item):
        #print(user, item)
        u_emb = self.user_factors(user)
        i_emb = self.item_factors(item)
        # print(u_emb.shape, i_emb.shape)
        # print((u_emb * i_emb).shape)
        # print((u_emb * i_emb).sum(axis=1).shape)
        return (u_emb * i_emb).sum(axis=1)


# In[28]:


train_loader = DataLoader(MovieLensDataset(train_df), batch_size=2, shuffle=True,)
next(iter(train_loader))


# In[29]:


data = next(iter(train_loader))
user, item = data['user'], data['item']
model = MatrixFactorizationPyTorch(n_user, n_item, k=config.embedding_dim)
model(user, item)


# # train

# In[30]:


def train_one_epoch(epoch, model, loss_fn, optimizer,
                    train_loader, device, scheduler=None):
    model.train()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    data_cnt = 0
    total_loss = 0.0

    # 学習データをシャッフルしてループ
    for step, data in pbar:
        user = data['user']
        item = data['item']
        rating = data['rating']
        data_cnt += user.shape[0]

        # 勾配リセット
        optimizer.zero_grad()

        #順伝搬、逆伝搬
        outputs = model(user, item)
        #print('outupts', outputs)
        #print(rating)
        loss = loss_fn(outputs,  rating.float())
        #print('loss', loss)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        #print(total_loss)
        if ((step + 1) % config.verbose_step == 0) or ((step + 1) == len(train_loader)):
            description = f'train epoch {epoch} loss: {total_loss / data_cnt:.4f}'
            pbar.set_description(description)

    total_loss = total_loss / len(train_loader)
    print('train loss = {:.4f}'.format(total_loss))

def valid_one_epoch(epoch, model, loss_fn, val_loader, device):

    model.eval()
    total_loss = 0.0
    data_cnt = 0
    #preds = []
    pbar = tqdm(enumerate(val_loader), total=len(val_loader))

    for step, data in pbar:
        user = data['user']
        item = data['item']
        rating = data['rating']
        data_cnt += user.shape[0]

        outputs = model(user, item)
        loss = loss_fn(outputs, rating)
        total_loss += loss
        
        # preds.append(outputs.detach().cpu().numpy())

        if ((step + 1) % config.verbose_step == 0) or ((step + 1) == len(val_loader)):
            description = f'val epoch {epoch} loss: {total_loss / data_cnt:.4f}'
            pbar.set_description(description)
        

    valid_loss = total_loss / len(val_loader)
    print('val loss = {:.4f}'.format(valid_loss))
    return valid_loss 

def run_train(train_loader, valid_loader):
    device = torch.device(config.device)
    model = MatrixFactorizationPyTorch(n_user, n_item, k=config.embedding_dim)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=config.lr)
    best_loss=1e10
    for epoch in range(config.epochs):
        train_one_epoch(epoch, model, loss_fn, optimizer, train_loader, device)

        with torch.no_grad():
            val_loss = valid_one_epoch(epoch, model, loss_fn, valid_loader, device)
        
        if best_loss > val_loss:
            best_loss = val_loss
            best_rmse = torch.sqrt(best_loss)
            best_epoch = epoch
            # TODO: save model,  figure
            best_path =  os.path.join(OUTPUT_DIR,f'best_model.bin')
            torch.save({'model':model.state_dict(),},
                           best_path)
    print(f'----- result ------')
    print(f'Best epoch: {epoch}')
    print(f'Best loss: {best_loss}, RMSE: {best_rmse}')


# In[31]:


train_loader = DataLoader(MovieLensDataset(train_df), batch_size=config.train_bs, shuffle=True,)
valid_loader = DataLoader(MovieLensDataset(valid_df), batch_size=config.valid_bs, shuffle=False,)
run_train(train_loader, valid_loader)


# # get recommendation

# In[32]:


m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
item_df = pd.read_csv(os.path.join(DATA_DIR, 'u.item'), sep='|', encoding="iso-8859-1", usecols=range(5), names=m_cols)
item_df.head()


# In[109]:


def load_model():
    best_path = os.path.join(OUTPUT_DIR, 'best_model.bin')
    model = MatrixFactorizationPyTorch(n_user, n_item, k=20)
    model.load_state_dict(torch.load(best_path)['model'])
    return model

def predict_rating(rec_df):
    """
    predict unwatched item ratings
    """
    model = load_model()
    model.eval()
    dataloader = DataLoader(MovieLensDataset(rec_df), batch_size=10, shuffle=False,)
    pbar = tqdm(dataloader, total=len(dataloader))
    preds = []
    for data in pbar:
        user_id = data['user']
        item_id = data['item']
        rating = data['rating']

        preds += model(user_id, item_id)

    return torch.stack(preds).detach().numpy()

def recommend_for_user(user_id, rating_df, item_df, top_n=10):
    """
    """
    rec_df = rating_df.query("user_id != @user_id")
    rec_df['user_id'] = user_id
    rec_df = rec_df.drop_duplicates(subset=['user_id','item_id'])
    rec_df['rating'] = predict_rating(rec_df)
    
    # clip rating
    rec_df = rec_df.query('0.5 <= rating <= 5.5 ')

    # add title column 
    d = dict(zip(item_df.movie_id, item_df.title))
    rec_df['title'] = rec_df['item_id'].map(d)
    rec_df = rec_df.sort_values('rating', ascending=False)

    # show recommend movies
    print('-'*30 + 'recommendations' + '-'*30)
    print(rec_df[['title','rating']].head(top_n))
#     for i, row in rec_df.head(top_n).iterrows():
#         title, rating = row['title'],row['rating']
#         print(f'{i:}: title:{title}  score:{rating}')

    # show movies which user have watched before
    user_df = rating_df.query("user_id == @user_id")
    user_df['title'] = user_df['item_id'].map(d)
    user_df = user_df.sort_values('rating', ascending=False)

    print('-'*30 + 'watched_movies' + '-'*30)
    print(user_df[['title','rating']].head(top_n))
#     for i, row in user_df.head(top_n).iterrows():
#         title, rating = row['title'], row['rating']
#         print(f'{i}: title:{title}  score:{rating}')


# In[112]:


user_id = random.choice(df.user_id.values)
print(user_id)
recommend_for_user(user_id, df, item_df)


# In[ ]:




