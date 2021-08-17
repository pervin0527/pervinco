#-*- coding:utf-8 -*-

import torch
import torch.utils.data
from collections import OrderedDict
import numpy as np

def load_state_dict(model, state_dict):
    model_names = [n for n in model.state_dict().keys()]
    state_names = [n for n in state_dict.keys()]

    # find prefix for the model and state dicts from the first param name
    if model_names[0].find(state_names[0]) >= 0:
        model_prefix = model_names[0].replace(state_names[0], '')
        state_prefix = None
    elif state_names[0].find(model_names[0]) >= 0:
        state_prefix = state_names[0].replace(model_names[0], '')
        model_prefix = None
    else:
        print('ERROR between {:s} and {:s}'.format(model_names[0], state_names[0]))
        raise KeyError

    new_state_dict = OrderedDict()
    for i,(k,v) in enumerate(state_dict.items()):
        if state_prefix is None:
            new_name = model_prefix + model_names[i]
        else:
            new_name = model_names[i].replace(state_prefix, '')

        new_state_dict[new_name] = v

    model_state = model.state_dict()
    model_state.update(new_state_dict)
    model.load_state_dict(model_state)


def quaternion_angular_error(q1, q2):
    """
    angular error between two quaternions
    :param q1: (4, )
    :param q2: (4, )
    :return:
    """
    d = abs(np.dot(q1, q2))
    d = min(1.0, max(-1.0, d))
    theta = 2 * np.arccos(d) * 180 / np.pi
    return theta


def qexp(q):
    """
    Applies the exponential map to q
    :param q: (3,)
    :return: (4,)
    """
    n = np.linalg.norm(q)
    q = np.hstack((np.cos(n), np.sinc(n/np.pi)*q))
    return q



def _find_yaw_from_q(q):

	"""
	Applies exponential map to log quaternion
	:param q: N x 3
	:return: N x 4
	"""
	n = np.linalg.norm(q)
	n = max(n , 1e-8)
	q = q / n

	# roll (x-axis rotation)
	sinr_cosp = +2.0 * (q[0] * q[1] + q[2] * q[3])
	cosr_cosp = +1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2])
	roll = np.arctan2(sinr_cosp, cosr_cosp)

	# pitch (y-axis rotation)
	sinp = 2.0 * (q[0] * q[2] - q[3] * q[1])
	if (abs(sinp) >= 1):
		pitch = math.copysign(np.pi / 2, sinp) #  // use 90 degrees if out of range
	else:
		pitch = np.arcsin(sinp)

	# yaw (z-axis rotation)
	siny_cosp = 2.0 * (q[0] * q[3] + q[1] * q[2])
	cosy_cosp = 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]) #np.cos(n)*np.cos(n) + q[0]*q[0] - q[1] * q[1] - q[2] * q[2]

	yaw = np.arctan2(siny_cosp, cosy_cosp)

	return yaw




def qlog(q):
    """
    dim(q)=4 -> dim(q)=3
    """
    if all(q[1:] == 0):
        print(q[1:])
        q = np.zeros(3)
    else:
        # 역삼각함수 arccos()
        q = np.arccos(q[0]) * q[1:] / np.linalg.norm(q[1:])
    return q




import pyproj

def latlon_to_utm_with_pyproj_package(lon, lat, number, letter, east_mean, north_mean):
    proj = '{:02d}{}'.format(number, letter)
    x1, y1 = pyproj.transform(pyproj.Proj('+proj=latlong'),
                              pyproj.Proj('+proj=utm +zone={}'.format(proj)),
                              lon, lat)
    return x1-east_mean, y1-north_mean



def utm_to_latlon_with_pyproj_package(x, y, number, letter, east_mean, north_mean):
    proj = '{:02d}{}'.format(number, letter)
    x = x + east_mean
    y = y + north_mean
    lon, lat = pyproj.transform(pyproj.Proj('+proj=utm +zone={}'.format(proj)),
                                pyproj.Proj('+proj=latlong'), x, y)
    return lon, lat
