import sys
import os.path as osp

sys.path.insert(0, '../')


from loss import MecNetLoss, MecNetANGLoss # for pos+ang: MecNetANGLoss
from models.mecnet import MECNet, MECANGNet # for pos+ang: MECANGNET
from models import mobilenet
from lsar import LSAR_POS_SOFT, LSAR_POSANG_SOFT # for pos+ang: LSAR_POSANG_SOFT

import os.path as osp
import numpy as np
import argparse
import configparser
import json
import torch
from torch import nn
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from utils import load_state_dict, quaternion_angular_error, _find_yaw_from_q, qexp
import matplotlib.pyplot as plt

## args
class MECNetArgs():
    def __init__(self):

        self.dataset = 'LSAR_POS_SOFT' # 'LSAR_POSANG_SOFT' for pos+angle
        self.model_name = 'menet'
        self.learn_alpha = True

        self.expflag = 'test'
        self.device = '0'
        self.maxflr = 3
        self.inflr = 1
        self.flrflag = True

        self.resume = True
        self.weights = 'results/test/epoch_005.pth.tar'

        self.datapath = osp.join('..', 'traintest_dataset_12_f1') # dataset path

## parameters
class MECNetConfig():
    def __init__(self):

        self.n_epochs = 15
        self.batch_size = 12
        self.seed = 7
        self.shuffle = True
        self.num_workers = 6
        self.snapshot = 5
        self.val_freq = 1
        self.max_grad_norm = 0
        self.cuda = True

        self.lr = 1e-4
        self.weight_decay = 0.0005
        self.lr_decay = 0.1
        self.lr_stepvalues = [10]
        self.print_freq = 20
        self.logdir = './results'


args = MECNetArgs()
cfg = MECNetConfig()

alpha = -3.0
dropout = 0.5
color_jitter = 0.7

alphap=alpha
alphaa=alpha
alphaf=alpha


data_dir = args.datapath
stats_file = osp.join(data_dir, 'stats.txt')


stats = np.loadtxt(stats_file)

# transformers
tforms = [transforms.Resize(256)]
if color_jitter > 0:
    assert color_jitter <= 1.0
    print('ColorJitter data augmentation')
    tforms.append(transforms.ColorJitter(brightness=color_jitter,
        contrast=color_jitter, saturation=color_jitter, hue=0.5))
tforms.append(transforms.ToTensor())
tforms.append(transforms.Normalize(mean=stats[0], std=np.sqrt(stats[1])))
data_transform = transforms.Compose(tforms)


# data loader
kwargs = dict(data_path=data_dir, train= True, transform=data_transform, \
            seed=cfg.seed, maxflr=args.maxflr, inflr=args.inflr, flrflag=args.flrflag)
if args.dataset.find('ANG')>=0:
    data_set = LSAR_POSANG_SOFT(**kwargs)
else:
    data_set = LSAR_POS_SOFT(**kwargs)


# loader (batch_size MUST be 1)
batch_size = 1
assert batch_size == 1
loader = DataLoader(data_set, batch_size=batch_size, shuffle=False,
num_workers=0, pin_memory=True)

# load num classes
numcls = data_set.num_classes
numang = data_set.num_angles
numflr = data_set.num_floors


# model
feature_extractor = mobilenet.__dict__['mobilenetv2'](pretrained=True)
if args.dataset.find('ANG')>=0:
    model = MECANGNet(feature_extractor, droprate=dropout, pretrained=True,
          num_classes=numcls,num_angles=numang,num_floors=numflr)
else:
    model = MECNet(feature_extractor, droprate=dropout, pretrained=True,
          num_classes=numcls,num_angles=numang,num_floors=numflr)


model.eval()

# loss functions
t_criterion = lambda t_pred, t_gt: np.linalg.norm(t_pred - t_gt)
q_criterion = quaternion_angular_error

# load weights
weights_filename = osp.expanduser(args.weights)
if osp.isfile(weights_filename):
	loc_func = lambda storage, loc: storage
	checkpoint = torch.load(weights_filename, map_location=loc_func)
	load_state_dict(model, checkpoint['model_state_dict'])
	print('Loaded weights from {:s}'.format(weights_filename))
else:
	print('Could not load weights from {:s}'.format(weights_filename))
	sys.exit(-1)


# activate GPUs
CUDA = torch.cuda.is_available()
torch.manual_seed(cfg.seed)
if CUDA:
	torch.cuda.manual_seed(cfg.seed)
	model.cuda()


pose_stats_filename = osp.join(data_dir, 'pose_stats_cls_soft_flr'+str(args.inflr)+'.txt')
pose_m, pose_s, max_t, min_t = np.loadtxt(pose_stats_filename)


L = len(data_set)


# make grid map
grid_size = 2
grid_x = int((max_t[0] - min_t[0]) / grid_size)+1 + 2
grid_y = int((max_t[1] - min_t[1]) / grid_size)+1 + 2


cls_ind = np.arange(numcls)
rows = ((cls_ind.astype('int') / grid_y - 1 )*grid_size + min_t[0])
cols = ((cls_ind.astype('int') % grid_y - 1 )*grid_size + min_t[1])

pred_poses = np.zeros((L, 7))  # store all predicted poses
cls_poses = np.zeros((L, 2))  # store all predicted poses
targ_poses = np.zeros((L, 7))  # store all target poses
u = np.zeros((L, 1))
v = np.zeros((L, 1))

# inference loop
for batch_idx, (data, target) in enumerate(loader):
	if batch_idx % 200 == 0:
		print('Image {:d} / {:d}'.format(batch_idx, len(loader)))

	idx = [batch_idx]
	# idx = idx[len(idx) / 2]

	# output : 1 x 6 or 1 x STEPS x 6
	data_var = Variable(data, requires_grad=False)
	if cfg.cuda:
		data_var = data_var.cuda(non_blocking=True)
		target = target.cuda(non_blocking=True)

	target_var = Variable(target, requires_grad=False)

    # predict pos
	with torch.set_grad_enabled(False):
		output = model(data_var)

    # split outputs
	output_pos = output[:,:6]
	output_cls = output[:,6:numcls+6]
	output_ang = output[:,numcls+6:numcls+numang+6]
	output_flr = output[:,numcls+numang+6:-1]
	target_pos = target[:,:6]
	target_cls = target[:,6:]
	# target_cls = target[:,:,6:]

	s = output_pos.size()
	s = target_pos.size()
	output_pos = output_pos.cpu().data.numpy().reshape((-1, s[-1]))
	target_pos = target_pos.cpu().data.numpy().reshape((-1, s[-1]))

	# normalize the predicted quaternions
	q = [qexp(p[3:]) for p in output_pos]
	output_pos = np.hstack((output_pos[:, :3], np.asarray(q)))
	q = [qexp(p[3:]) for p in target_pos]
	target_pos = np.hstack((target_pos[:, :3], np.asarray(q)))


	# un-normalize the predicted and target translations
	output_pos[:, :3] = (output_pos[:, :3] * pose_s) + pose_m
	target_pos[:, :3] = (target_pos[:, :3] * pose_s) + pose_m

	# take the middle prediction
	pred_poses[idx, :] = output_pos #[len(output_pos)/2]
	targ_poses[idx, :] = target_pos #[len(target_pos)/2]

	# s = output_ang.size()
	# output_ang = output_ang.cpu().data.numpy().reshape((-1, s[-1]))
	# # output_ang = output_ang[len(output_ang)/2]
	# outang = np.argmax(output_ang)

	output_cls = F.softmax(output_cls, dim=1).cpu().data.numpy()
	outcls = np.argmax(output_cls)


	## Compute weighted sum of indexes around current index which has the maximum score
	outcls = outcls
	cur_row = int((outcls.astype('int') / grid_y) )
	cur_col = int((outcls.astype('int') % grid_y) )

	ti = np.array(range(min(grid_x, max(cur_row-2,0)),min(grid_x, max(cur_row+3,0)) ))
	tj = np.array(range(min(grid_y, max(cur_col-2,0)),min(grid_y, max(cur_col+3,0)) ))

	cur_rows, cur_cols = np.meshgrid(ti, tj)
	#

	cur_outcls = (np.array(cur_rows))  * grid_y + (np.array(cur_cols))
	cur_outcls = cur_outcls.flatten()


	cur_outputcls = np.zeros_like(output_cls.flatten())
	cur_outputcls[cur_outcls] = output_cls[:,cur_outcls]

	cur_outputcls = cur_outputcls / sum(cur_outputcls)

	clsx = sum(rows*cur_outputcls.flatten())
	clsy = sum(cols*cur_outputcls.flatten())

	cls_poses[idx,0] = clsx
	cls_poses[idx,1] = clsy


# calculate losses
c_loss = np.asarray([t_criterion(p, t) for p, t in zip(cls_poses[:, :2], targ_poses[:, :2])])

print('Error in translation (classification): median {:3.2f} m,  mean {:3.2f} m\n'.format(np.median(c_loss), np.mean(c_loss)))


# create figure object
fig = plt.figure()
if args.dataset != '3D':
	ax = fig.add_subplot(111)
else:
	ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0, bottom=0, right=1, top=1)

# plot on the figure object
ss = max(1, int(len(data_set) / 100))  # 100 for stairs
# scatter the points and draw connecting line
x = np.vstack((cls_poses[::ss, 0].T, targ_poses[::ss, 0].T))
y = np.vstack((cls_poses[::ss, 1].T, targ_poses[::ss, 1].T))


if args.dataset != '3D':  # 2D drawing
	ax.plot(x, y, c='b')
	ax.scatter(x[0, :], y[0, :], c='r')
	ax.scatter(x[1, :], y[1, :], c='g')

	## for angle

    # grid_ang = np.pi / 36
    #
    # pi = outang * grid_ang
    # u[idx,:] = 1*np.cos(pi)
    # v[idx,:] = 1*np.sin(pi)

	# heading_pi = _find_yaw_from_q(pred_poses[idx,3:])
	# pdb.set_trace()
	# u = 1*np.cos(heading_pi)
	# v = 1*np.sin(heading_pi)
	# ax.quiver(pred_poses[:,0],pred_poses[:,1],u,v, angles='xy',  scale=20, color='blue')

else:
	z = np.vstack((pred_poses[::ss, 2].T, targ_poses[::ss, 2].T))
	for xx, yy, zz in zip(x.T, y.T, z.T):
		ax.plot(xx, yy, zs=zz, c='b')
	ax.scatter(x[0, :], y[0, :], zs=z[0, :], c='r', depthshade=0)
	ax.scatter(x[1, :], y[1, :], zs=z[1, :], c='g', depthshade=0)
	ax.view_init(azim=119, elev=13)


plt.show(block=True)
plt.savefig('result.png', bbox_inches='tight')
