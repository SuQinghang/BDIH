import sys
sys.path.append('.')
import torch, os, pdb, collections, pdb, json, PIL, pickle
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from data.transform import (Onehot, encode_onehot, query_transform,
                            train_transform)
from loguru import logger
import copy

NUM_CLASSES_PER_SESSION=20
NUM_CLASSES=100
class CIFAR100(Dataset):
    def __init__(self,
        root = "Dataset/cifar100_png",
        prefix = "data/collections/cifar100",
        mode=None,
        session_id = None,
        joint_train = False,
        exp_name = "general10",
    ):
        assert mode in ["train","gallery","val","test"], "mode should be {train, gallery, val, test}"
        self.session_id = session_id
        test_files = [os.path.join(prefix,"cifar100_test_rand1_cls20_task%d.json"%(i)) for i in range(5)]
        train_files = [os.path.join(prefix,"cifar100_train_%s_rand1_cls20_task%d.json"%(exp_name,i)) for i in range(5)]
        val_files = [os.path.join(prefix,"cifar100_val_%s_rand1_cls20_task%d.json"%(exp_name,i)) for i in range(5)]
        files = train_files if mode in ["train","gallery"] else  val_files if mode == "val" else test_files
        if not joint_train and mode in ["train","gallery"]:
            with open(files[session_id]) as f:#collect current session data only
                datalist = json.load(f)
        else:
            datalist = []
            session_id = 4 if "blur" in exp_name and mode in ["val","test"] else session_id
            for sid in range(session_id+1):#collect session data seen so far
                with open(files[sid]) as f:
                    datalist.extend(json.load(f))
        self.data = np.array([item["file_name"] for item in datalist])
        self.targets = np.array([item["label"] for item in datalist])
        self.category_list = list(set(self.targets))
        self.root = root
        self.mode = mode
        if mode == "train":
            self.desc = "training set"
        elif mode == "gallery":
            self.desc = "gallery set"
        elif mode == "val":
            self.desc = "validation query set"
        elif mode == "test":
            self.desc = "testing query set"
        # =================================================================================
        # Transform Definition
        if mode == "train":
            self.transform = train_transform()
            # print("Using train-transforms:",self.transform)
        else:#mode in ["gallery", "val", "test"]
            self.transform = query_transform()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.data[idx]
        label = self.targets[idx]
        img_path = os.path.join(self.root, img_name)
        imgpil = PIL.Image.open(img_path).convert("RGB")
        if self.transform:
            trans_img = self.transform(imgpil)
        return trans_img, label, idx
    
    def get_onehot_targets(self):
        self.onehot_targets = encode_onehot(np.asarray(self.targets), num_classes=NUM_CLASSES)
        return torch.from_numpy(self.onehot_targets).float()

    def add_memory(self, exem_data, exem_labels):

        tmp = copy.deepcopy(exem_data)
        tmp.extend(self.data.tolist())
        self.data = np.array(tmp)

        tmp = copy.deepcopy(exem_labels)
        tmp.extend(self.targets.tolist())
        self.targets = np.array(tmp)

        logger.info("##### [add memory] #####")
        cat_dict = {}
        for cat in exem_labels:
            if cat not in cat_dict:
                cat_dict[cat] = 0
            cat_dict[cat] += 1
        logger.info(cat_dict)
        logger.info("########################")
        
    def add_memory_prev(self, exem_data, exem_labels):

        tmp = self.data.tolist()
        tmp.extend(exem_data)
        self.data = np.array(tmp)

        tmp = self.targets.tolist()
        tmp.extend(exem_labels)
        self.targets = np.array(tmp)
        logger.info("##### [add memory] #####")
        cat_dict = {}
        for cat in exem_labels:
            if cat not in cat_dict:
                cat_dict[cat] = 0
            cat_dict[cat] += 1
        logger.info(cat_dict)
        logger.info("########################")
    
    def show(self,verbose=True):
        print("-----------------")
        print("[%s]"%(self.desc))
        print("class label from %d to %d"%(np.min(self.targets),np.max(self.targets)))
        print("number of data: ",self.data.shape," with dtype %s"%(self.data.dtype))
        if verbose: print({tar:cnt for tar, cnt in zip(*np.unique(self.targets, return_counts=True))})
        print("-----------------")
