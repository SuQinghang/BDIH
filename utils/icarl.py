import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from torch.utils.data import DataLoader

def random(dataset, memory_size):
    gallery = deepcopy(dataset)

    # Collect exemplars
    cat_dict = {}
    for dat, tar in zip(gallery.data, gallery.targets):
        tar = int(tar)
        if tar not in cat_dict:
            cat_dict[tar] = []
        cat_dict[tar].append(dat)
    num_class = len(cat_dict.keys())
    mem_per_cls = memory_size // num_class
    exemplar_set = {}
    for class_id in cat_dict:
        current_images = np.array(cat_dict[class_id])
        sample_index = np.random.choice(len(current_images), mem_per_cls, replace=False)
        exemplar_set[class_id] = list(current_images[sample_index])
    return exemplar_set


def icarl(net, dataset, memory_size, device="cuda"):
    gallery = deepcopy(dataset)#prevent from modification
    def addNewExemplars(net,loader,device,exem_num):
        # extract embeddings per class
        net.eval()
        embedding_set, image_set = [], []
        with torch.no_grad():
            for images, class_labels, idx in loader:
                images = images.to(device)
                embeddings = net(images)
                embedding_set.append(embeddings)
                image_set.extend(np.array(loader.dataset.data)[idx])
            embedding_set = torch.cat(embedding_set)
            embedding_set = F.normalize(embedding_set.detach(),p=2,dim=1).cpu().numpy()
            embedding_mean = np.mean(embedding_set, axis=0)
        embedding_set = np.array(embedding_set)
        now_embedding_sum = np.zeros_like(embedding_mean)
        # collect exemplars per class
        images = []
        for i in range(exem_num):
            x = embedding_mean - (now_embedding_sum + embedding_set) / (i + 1)
            x = np.linalg.norm(x, axis=1)
            index = np.argmin(x)
            now_embedding_sum += embedding_set[index]
            images.append(image_set[index])
            embedding_set = np.delete(embedding_set, index, axis=0)
            image_set = np.delete(image_set, index, axis=0)
            if len(embedding_set) == 0: break
        return images
    # Collect exemplars
    cat_dict = {}
    for dat, tar in zip(gallery.data, gallery.targets):
        tar = int(tar)
        if tar not in cat_dict:
            cat_dict[tar] = []
        cat_dict[tar].append(dat)
    num_class = len(cat_dict.keys())
    mem_per_cls = memory_size // num_class
    exemplar_set = {}
    for class_id in cat_dict:
        gallery.data = cat_dict[class_id]
        gallery.targets = [class_id] * len(gallery.data)
        loader = DataLoader(gallery, batch_size=100, shuffle=False, num_workers=0)
        images = addNewExemplars(net, loader, device, mem_per_cls)
        exemplar_set[class_id] = images
    return exemplar_set