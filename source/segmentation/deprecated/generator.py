import numpy as np
from collections import defaultdict

def base_inheritance(cls):
    def inherit_base(dataset, *args, **kwargs):
        return type(cls.__name__, (cls, *dataset.__class__.mro(),), {})(dataset, *args, **kwargs)

    return inherit_base

@base_inheritance
class TFDataGenerator:
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.__dict__ = dataset.__dict__.copy()
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(self.dataset.__len__())
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return -(-self.dataset.__len__() // self.batch_size)

    def __getitem__(self, batch_index):
        indices = self.indices[batch_index*self.batch_size:(batch_index+1)*self.batch_size]
        result = self.__data_generation(indices)
        
        return result["image"], result["mask"]
        
    def __data_generation(self, indices):
        result = defaultdict(list)
        for index in indices:
            sample = self.dataset[index]
            for k, v in sample.items():
                result[k].append(v)
        for k, v in result.items():
            result[k] = np.stack(v, axis=0)
        return result