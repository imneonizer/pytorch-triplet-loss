import re
import torch
from pytorch_metric_learning import testers
import numpy as np

def get_all_embeddings(dataset, model, device):
    tester = testers.BaseTester(data_device=device)
    return tester.get_all_embeddings(dataset, model)

def log_to_file(msg, path="training.log"):
    with open(path, "a") as f:
        f.write(msg+"\n")

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def get_accuracy(val_dataset, train_dataset, model, device):
    # Please make sure val dataset is passed as first argument
    # Other wise false accuracy might be reported
    # due to brodcasting with a small number of samples
    
    with torch.no_grad():
        val_embeddings, val_labels = get_all_embeddings(val_dataset, model, device=device)
        train_embeddings, train_labels = get_all_embeddings(train_dataset, model, device=device)
        # for each val embedding, find distance with all embeddings in train embeddings
        dist = torch.cdist(val_embeddings, train_embeddings)
    
    query_labels = np.array(val_dataset.labels)
    # Find index of closesest matching embedding
    matched_idx = torch.argmin(dist, axis=1).cpu().numpy()
    matched_labels = np.array(train_dataset.labels)[matched_idx]

    accuracy = (query_labels == matched_labels).mean()
    return accuracy