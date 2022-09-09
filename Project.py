#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis with Deep Learning using BERT

# ### Prerequisites

# - Intermediate-level knowledge of Python 3 (NumPy and Pandas preferably, but not required)
# - Exposure to PyTorch usage
# - Basic understanding of Deep Learning and Language Models (BERT specifically)

# ### Project Outline

# **Task 1**: Introduction (this section)
# 
# **Task 2**: Exploratory Data Analysis and Preprocessing
# 
# **Task 3**: Training/Validation Split
# 
# **Task 4**: Loading Tokenizer and Encoding our Data
# 
# **Task 5**: Setting up BERT Pretrained Model
# 
# **Task 6**: Creating Data Loaders
# 
# **Task 7**: Setting Up Optimizer and Scheduler
# 
# **Task 8**: Defining our Performance Metrics
# 
# **Task 9**: Creating our Training Loop
# 
# **Task 10**: Loading and Evaluating our Model

# ## Task 1: Introduction

# ### What is BERT
# 
# BERT is a large-scale transformer-based Language Model that can be finetuned for a variety of tasks.
# 
# For more information, the original paper can be found [here](https://arxiv.org/abs/1810.04805). 
# 
# [HuggingFace documentation](https://huggingface.co/transformers/model_doc/bert.html)
# 
# [Bert documentation](https://characters.fandom.com/wiki/Bert_(Sesame_Street) ;)

# <img src="Images/BERT_diagrams.pdf" width="1000">

# ## Task 2: Exploratory Data Analysis and Preprocessing

# We will use the SMILE Twitter dataset.
# 
# _Wang, Bo; Tsakalidis, Adam; Liakata, Maria; Zubiaga, Arkaitz; Procter, Rob; Jensen, Eric (2016): SMILE Twitter Emotion dataset. figshare. Dataset. https://doi.org/10.6084/m9.figshare.3187909.v2_

# In[4]:


import torch
import pandas as pd
from tqdm.notebook import tqdm


# In[5]:


df = pd.read_csv(
    'Data/smile-annotations-final.csv',
    names = ['id', 'text', 'category'])
df.set_index('id', inplace = True)


# In[6]:


df.head()


# In[7]:


df.text.iloc[0]


# In[8]:


df.category.value_counts()


# In[9]:


df = df[~df.category.str.contains('\|')]


# In[10]:


df = df[df.category != 'nocode']


# In[11]:


df.category.value_counts()
#class imbalance


# In[12]:


possible_labels = df.category.unique()


# In[13]:


possible_labels


# In[14]:


label_dict = {}
for index, possible_label in enumerate(possible_labels):
    label_dict[possible_label] = index


# In[15]:


label_dict


# In[16]:


df['label'] = df.category.replace(label_dict)
df.head()


# In[17]:


df.head(15)


# ## Task 3: Training/Validation Split

# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


X_train, X_val, y_train, y_val = train_test_split(
    df.index.values,
    df.label.values,
    test_size = 0.15,
    random_state = 17,
    stratify = df.label.values
)


# In[20]:


df['data_type'] = ['not_set']*df.shape[0]


# In[21]:


df.head()


# In[22]:


df.loc[X_train, 'data_type'] = 'train'
df.loc[X_val, 'data_type'] = 'val'


# In[23]:


df.groupby(['category', 'label', 'data_type']).count()


# ## Task 4: Loading Tokenizer and Encoding our Data

# In[24]:


from transformers import BertTokenizer
from torch.utils.data import TensorDataset


# In[25]:


tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased',
    do_lower_case = True
)


# In[26]:


encoded_data_train = tokenizer.batch_encode_plus(
    df[df.data_type == 'train'].text.values,
    # bert to know where the sentence begin and end using line below
    add_special_tokens = True,
    # fix input need to set max number of letters in a twitter
    return_attention_mask = True,
    pad_to_max_length =  True,
    # twitter max length
    max_length = 256,
    # pytorch
    return_tensors = 'pt'
)


encoded_data_val = tokenizer.batch_encode_plus(
    df[df.data_type == 'val'].text.values,
    # bert to know where the sentence begin and end using line below
    add_special_tokens = True,
    # fix input need to set max number of letters in a twitter
    return_attention_mask = True,
    pad_to_max_length =  True,
    # twitter max length
    max_length = 256,
    # pytorch
    return_tensors = 'pt'
)


input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train =  torch.tensor(df[df.data_type=='train'].label.values)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val =  torch.tensor(df[df.data_type=='val'].label.values)


# In[27]:


dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)


# In[28]:


len(dataset_train)


# In[29]:


len(dataset_val)


# ## Task 5: Setting up BERT Pretrained Model

# In[30]:


from transformers import BertForSequenceClassification


# In[31]:


model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', 
    num_labels = len(label_dict),
    output_attentions = False,
    output_hidden_states = False
)


# ## Task 6: Creating Data Loaders

# In[32]:


from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


# In[33]:


batch_size = 4 #32

dataloader_train = DataLoader(
    dataset_train,
    sampler = RandomSampler(dataset_train),
    batch_size = batch_size
)

dataloader_val = DataLoader(
    dataset_val,
    sampler = RandomSampler(dataset_val),
    batch_size = 32
)


# ## Task 7: Setting Up Optimizer and Scheduler

# In[34]:


from transformers import AdamW, get_linear_schedule_with_warmup


# In[35]:


optimizer = AdamW(
    model.parameters(),
    lr = 1e-5, #2e-5>5e-5
    eps = 1e-8
)


# In[36]:


epochs = 10

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps = 0,
    num_training_steps = len(dataloader_train)*epochs
)


# ## Task 8: Defining our Performance Metrics

# Accuracy metric approach originally used in accuracy function in [this tutorial](https://mccormickml.com/2019/07/22/BERT-fine-tuning/#41-bertforsequenceclassification).

# In[37]:


import numpy as np


# In[38]:


from sklearn.metrics import f1_score


# In[39]:


#preds = [0.9, 0.05, 0.05, 0,0,0]
#preds = [1,0,0,0,0,0]


# In[62]:


def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis = 1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average = 'weighted')
# weight depends on sample size


# In[63]:


def accuracy_per_class(preds, labels):
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    
    preds_flat = np.argmax(preds, axis = 1).flatten()
    labels_flat = labels.flatten()
    
    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')
        


# ## Task 9: Creating our Training Loop

# Approach adapted from an older version of HuggingFace's `run_glue.py` script. Accessible [here](https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128).

# In[43]:


import random

seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


# In[44]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(device)


# In[56]:


def evaluate(dataloader_val):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in tqdm(dataloader_val):
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals


# In[51]:


for epoch in tqdm(range(1, epochs+1)):
    model.train()
    loss_train_total = 0
    progress_bar = tqdm(dataloader_train, 
                        desc = 'Epoch {:1d}'.format(epoch),
                        leave = False, 
                        disable = False)
    for batch in progress_bar:
        
        model.zero_grad()
        batch = tuple(b.to(device) for b in batch)
        inputs = {
            'input_ids'      : batch[0],
            'attention_mask' : batch[1],
            'labels'         : batch[2]
            
        }
        
        outputs = model(**inputs)
        
        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'training_loss':'{:.3f}'.format(loss.item()/len(batch))})
    
    torch.save(model.state_dict(), f'Model/BERT_ft_epoch{epoch}.model')
    tqdm.write('\nEpoch {epoch}')
    loss_train_avg = loss_train_total/len(dataloader)
    tqdm.write('Training loss: {loss_train_avg}')
    
    val_loss, predictions, true_vals = evaluate(dataloader_val)
    val_f1 = f1_score_func(predictions, true_vals)
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'F1 score (weighted): {val_f1}')


# ## Task 10: Loading and Evaluating our Model

# In[52]:


model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)


# In[53]:


model.to(device)
pass


# In[57]:


model.load_state_dict(
    torch.load('Models/finetuned_bert_epoch_1_gpu_trained.model',
              map_location = torch.device('cpu')))


# In[61]:


_,predictions, true_vals = evaluate(dataloader_val)


# In[64]:


accuracy_per_class(predictions, true_vals)


# In[ ]:


# Google Colab -- GPU Instance(K80)
# batch_size = 32
# epoch = 10


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




