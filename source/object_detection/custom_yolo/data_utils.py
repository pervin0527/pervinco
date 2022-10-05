import numpy as np
import pandas as pd
from functools import reduce

def read_txt_file(path):
    f = open(path, "r", encoding="utf-8")
    lines = f.readlines()

    return lines


def read_label_file(path):
    df = pd.read_csv(path, sep=",", index_col=False, header=None)
    classes = df[0].tolist()

    return classes


def compose(*funcs):
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image

def preprocess_input(image):
    image /= 255.0
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])

    return image