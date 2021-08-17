#-*- coding:utf-8 -*-

"""
Data loader for the LSAR dataset
"""
import os
import os.path as osp
import numpy as np
from torch.utils import data
from torchvision.datasets.folder import default_loader
from torchvision import transforms, models
import sys
import pickle
from PIL import Image
import torch
from utils import qlog, _find_yaw_from_q, latlon_to_utm_with_pyproj_package, utm_to_latlon_with_pyproj_package
import utm
import pandas as pd


# define grid size
GRID_SIZE = 2 # for position
GRID_ANG_RPY = 6 # for angle


class PosDecodeOutput():
    def __init__(self, fname, rownum=0):
        dt = pd.read_csv(fname).iloc[rownum]
        self.east_mean = dt.east_mean
        self.north_mean = dt.north_mean
        self.zone_number = dt.zone_number
        self.zone_letter = dt.zone_letter
        self.floor_scaling = float(dt.floor_scaling)
        if self.floor_scaling < 0.001:
            self.floor_scaling = 0.001

    def decode_floor(self, est):
        return int(round(est/self.floor_scaling))

    def decode_WGS84(self, east, north):
        e = east + self.east_mean
        n = north + self.north_mean
        return utm.to_latlon(e, n, self.zone_number, self.zone_letter)

    def encode_floor(self, floor):
        return floor*self.floor_scaling

    def encode_WGS84(self, lon, lat):
        # print('lon: {}, lat: {}'.format(lon, lat))

        east, north, zone_num, zone_letter = utm.from_latlon(lat, lon)
        assert zone_num == self.zone_number
        assert zone_letter == self.zone_letter
        return (east - self.east_mean), (north - self.north_mean)



def load_image(filename, loader=default_loader):
  try:
    img = loader(filename)
  except IOError as e:
    print('Could not load image {:s}, IOError: {:s}'.format(filename, e))
    return None
  except:
    print('Could not load image {:s}, unexpected error'.format(filename))
    return None

  return img

class LSAR_POS_SOFT(data.Dataset):
    def __init__(self, data_path, train, transform=None, seed=7, maxflr=3, inflr=10, flrflag=True):
      """
      data_path: root data directory.
      train: for training
      transform: data transform
      """

      self.transform = transform
      self.target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())
      np.random.seed(seed)
      self.flrflag = flrflag
      self.colordrop = True

      # directories
      base_dir = osp.join(osp.expanduser(data_path))
      self.bdir = base_dir

      d = PosDecodeOutput(osp.join(base_dir, 'etri_dataset_e12_f123_c_m01_transforms.csv'))

      # decide which sequences to use
      if train:
        split_file = osp.join(base_dir, 'dataset_train.txt')
      else:
        split_file = osp.join(base_dir, 'dataset_test.txt')
      with open(split_file, 'r') as f:
          seqs = []
          xs = []
          ys = []
          zs = []
          qs = []
          qs4 = []
          cnt = 0
          for l in f:
              # print(l)

              seq = (l.split(' ')[0]) # image file path
              x = float(l.split(' ')[1]) # 경도
              y = float(l.split(' ')[2]) # 위도
              z = float(l.split(' ')[3]) # 층 floor
              
              # 1차원 벡터
              q = [float(l.split(' ')[4]), float(l.split(' ')[5]), float(l.split(' ')[6]), float(l.split(' ')[7])] # Quaternion list x, y, z, w
              # 1차원 벡터q / 벡터 q의 크기(norm)
              q = q / np.linalg.norm(q)

              if self.flrflag: ## floor flag?? floor에 대한 정보가 있을때를 구분하는 것???
                  if z == inflr:
                      if (cnt % 1) == 0:
                          # 데이터 각 리스트별 append
                          seqs.append(seq)

                          # pyproj로 위도, 경도 값 utm 좌표로 변환. latitude, longitude -> utm
                          new_x, new_y = latlon_to_utm_with_pyproj_package(x, y, d.zone_number, d.zone_letter, d.east_mean, d.north_mean)
                          xs.append(new_x)
                          ys.append(new_y)
                          # xs.append(x)
                          # ys.append(y)
                          zs.append(z)
                          qs.append(qlog(q))
                          qs4.append(q)

                  cnt += 1
              else:
                  if (cnt % 1) == 0:
                      seqs.append(seq)
                      new_x, new_y = latlon_to_utm_with_pyproj_package(x, y, d.zone_number, d.zone_letter, d.east_mean, d.north_mean)
                      xs.append(new_x)
                      ys.append(new_y)
                      # xs.append(x)
                      # ys.append(y)
                      zs.append(z)
                      qs.append(qlog(q)) # 4차원 -> 3차원으로 차원축소
                      qs4.append(q)

                  cnt += 1

          # 각각 images, utm_x, utm_y, floor, log quaternion(4D -> 3D), quaternion(4D)
          print(np.array(seqs).shape, np.array(xs).shape, np.array(ys).shape, np.array(zs).shape, np.array(qs).shape, np.array(qs4).shape)
      f.close()

      # read poses and collect image names
      self.c_imgs = []
      self.d_imgs = []
      self.gt_idx = np.empty((0,), dtype=np.int)
      ps = {}

      for i, seq in enumerate(seqs):
        seq_dir = osp.join(base_dir, seq)
        self.c_imgs = seqs

      """
      average, standard_deviation, max, min -> pose_stats_cls_soft_flt_1.txt 에 저장.
        - average : utm_x, utm_y, floor 각각의 평균들을 담고 있다.(None, 3)
        - std : utm_x, utm_y, 1(또는 floor) 의 표준 편차
        - max : utm_x 중 최댓값
        - min : utm_y 중 최솟값
      """

      # 내부적으로 계산된 x,y,z,q에 대한 max, min, mean, std 값을 pose_stats*.txt에 저장
      if self.flrflag:
          pose_stats_filename = osp.join(base_dir, 'pose_stats_cls_soft_flr'+str(inflr)+'.txt')
      else:
          pose_stats_filename = osp.join(base_dir, 'pose_stats_cls.txt')

      if train:
        mean_t = [np.mean(np.asarray(xs)), np.mean(np.asarray(ys)), np.mean(np.asarray(zs))]
        if self.flrflag:
            std_t = [np.std(np.asarray(xs)), np.std(np.asarray(ys)), 1]
        else:
            std_t = [np.std(np.asarray(xs)), np.std(np.asarray(ys)), np.std(np.asarray(zs))]

        max_t = [np.max(np.asarray(xs)), np.max(np.asarray(ys)), 1] # np.std(np.asarray(zs))]
        min_t = [np.min(np.asarray(xs)), np.min(np.asarray(ys)), 1] # np.std(np.asarray(zs))]

        np.savetxt(pose_stats_filename, np.vstack((mean_t, std_t, max_t, min_t)), fmt='%8.7f')
      else:
        mean_t, std_t, max_t, min_t = np.loadtxt(pose_stats_filename)

      GRID_SIZE = 2
      grid_x = int((max_t[0] - min_t[0]) / GRID_SIZE) + 1 + 2
      grid_y = int((max_t[1] - min_t[1]) / GRID_SIZE) + 1 + 2
      # print(grid_x, grid_y) # 18, 3
      grid_ang = np.pi / 36 # 3.14159 / 36


      self.num_classes = (grid_x)*(grid_y) # 54
      self.num_floors = maxflr # 3
      self.num_angles = 72

      """
      grid 공간에 맞게 data processing => 절댓값.
      convert pose to translation + log quaternion
      """
      self.poses = np.empty((0, 6))
      self.classes = np.empty((0, self.num_classes))
      self.floors = np.empty((0, 1))
      self.angles = np.empty((0, 1))
      self.mclasses = np.empty((0, 1))
      for seq,x,y,z,q,q4 in zip(seqs,xs,ys,zs,qs,qs4):
        pss = [(x-mean_t[0])/std_t[0], (y-mean_t[1])/std_t[1], (z-mean_t[2])/std_t[2], q[0], q[1], q[2]]

        c1 = int(np.floor((x - min_t[0]) / GRID_SIZE)) + 1
        r1 = int(np.floor((y - min_t[1]) / GRID_SIZE)) + 1
        c2 = int(np.ceil((x - min_t[0]) / GRID_SIZE)) + 1
        r2 = int(np.ceil((y - min_t[1]) / GRID_SIZE)) + 1

        dc1 = 1 - ( x - ((c1-1) * GRID_SIZE + min_t[0]) ) / GRID_SIZE #  [0,1]
        dc2 = 1 - dc1 #( (c2 * GRID_SIZE + min_t[0]) - x ) / GRID_SIZE #  [0,1]
        dr1 = 1 - ( y - ((r1-1) * GRID_SIZE + min_t[1]) ) / GRID_SIZE #  [0,1]
        dr2 = 1 - dr1 #( (r2 * GRID_SIZE + min_t[1]) - y ) / GRID_SIZE #  [0,1]

        pcls = np.zeros((1,self.num_classes))
        id_c1r1 = c1 * grid_y + r1
        id_c2r1 = min(c2, grid_x-1) * grid_y + r1
        id_c1r2 = c1 * grid_y + min(r2, grid_y-1)
        id_c2r2 = min(c2, grid_x-1) * grid_y + min(r2, grid_y-1)

        pcls[:,id_c1r1] += dc1*dr1
        pcls[:,id_c2r1] += dc2*dr1
        pcls[:,id_c1r2] += dc1*dr2
        pcls[:,id_c2r2] += dc2*dr2

        pmcls =  int(np.floor((x - min_t[0]) / (GRID_SIZE))+1)  * grid_y + int(np.floor((y - min_t[1]) / (GRID_SIZE))+1)
        pang =  int(np.floor(_find_yaw_from_q(q4) / grid_ang))
        pflr = int(z/10)-1

        self.poses = np.vstack((self.poses, pss)) # 3dim으로 회귀를 통한 위치와 각도에 대한 절대측위값.
        self.classes = np.vstack((self.classes, pcls)) # 위치 positions
        self.floors = np.vstack((self.floors, pflr)) # floor
        self.angles = np.vstack((self.angles, pang)) # 방향 Orientations
        self.mclasses = np.vstack((self.mclasses, pmcls))

      print(self.poses.shape, self.classes.shape, self.floors.shape, self.angles.shape, self.mclasses.shape)


    def __getitem__(self, index):
        img = None
        while img is None:
            img = load_image(osp.join(self.bdir, self.c_imgs[index]))

            pose = self.poses[index]
            clss = self.classes[index]
            flrs = self.floors[index]
            angs = self.angles[index]
            mcls = self.mclasses[index]
            index += 1
        index -= 1

        if self.target_transform is not None:
            pose = self.target_transform(pose)
            clss = self.target_transform(clss)
            angs = self.target_transform(angs)
            flrs = self.target_transform(flrs)
            mcls = self.target_transform(mcls)


        if self.transform is not None:
            img = self.transform(img)
            if self.colordrop:
                zeroout_id = np.random.permutation(3)
                zeroout_id = zeroout_id[0]
                new_img = torch.zeros_like(img)
                new_img[zeroout_id,:,:] = img[zeroout_id,:,:]
                img = new_img

        return img, torch.cat([pose, clss, angs, flrs, mcls])

    def __len__(self):
      return self.poses.shape[0]




class LSAR_POSANG_SOFT(data.Dataset):
    def __init__(self, data_path, train, transform=None,
                 seed=7, maxflr=3, inflr=10, flrflag=True):
      """

      data_path: root data directory.
      train: for training
      transform: data transform

      """

      self.transform = transform
      self.target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())
      np.random.seed(seed)
      self.flrflag = flrflag
      self.colordrop = True

      # directories
      base_dir = osp.join(osp.expanduser(data_path))
      self.bdir = base_dir

      # decide which sequences to use
      if train:
        split_file = osp.join(base_dir, 'dataset_train.txt')
      else:
        split_file = osp.join(base_dir, 'dataset_test.txt')


      d = PosDecodeOutput(osp.join(base_dir, 'etri_dataset_e12_f123_c_m01_transforms.csv'))

      with open(split_file, 'r') as f:
          seqs = []
          xs = []
          ys = []
          zs = []
          qs = []
          qs4 = []
          cnt = 0
          for l in f:

              seq = (l.split(' ')[0])
              x = float(l.split(' ')[1])
              y = float(l.split(' ')[2])
              z = float(l.split(' ')[3])
              q = [float(l.split(' ')[4]), float(l.split(' ')[5]), float(l.split(' ')[6]), float(l.split(' ')[7])]
              q = q / np.linalg.norm(q)


              if self.flrflag:
                  if z == inflr: #
                      if (cnt % 1) == 0:

                          seqs.append(seq)
                          new_x, new_y = latlon_to_utm_with_pyproj_package(x, y, d.zone_number, d.zone_letter, d.east_mean, d.north_mean)
                          xs.append(new_x)
                          ys.append(new_y)
                          # xs.append(x)
                          # ys.append(y)
                          zs.append(z)
                          qs.append(qlog(q))
                          qs4.append(q)

                  cnt += 1
              else:
                  if (cnt % 1) == 0:
                      seqs.append(seq)
                      new_x, new_y = latlon_to_utm_with_pyproj_package(x, y, d.zone_number, d.zone_letter, d.east_mean, d.north_mean)
                      xs.append(new_x)
                      ys.append(new_y)
                      # xs.append(x)
                      # ys.append(y)
                      zs.append(z)
                      qs.append(qlog(q))
                      qs4.append(q)

                  cnt += 1

      f.close()

      # read poses and collect image names
      self.c_imgs = []
      self.d_imgs = []
      self.gt_idx = np.empty((0,), dtype=np.int)
      ps = {}

      for i, seq in enumerate(seqs):
        seq_dir = osp.join(base_dir, seq)
        self.c_imgs = seqs

      if self.flrflag:
          pose_stats_filename = osp.join(base_dir, 'pose_stats_cls_soft_flr'+str(inflr)+'.txt') # 아래에서 구한 평균값, 표준 편차, 최대값, 최소값을 저장하는 파일
      else:
          pose_stats_filename = osp.join(base_dir, 'pose_stats_cls.txt')

      if train:
        mean_t = [np.mean(np.asarray(xs)), np.mean(np.asarray(ys)), np.mean(np.asarray(zs))] # 학습 시퀀스일 때 x, y, z 각각에 대한 평균값들을 하나의 리스트에 담아 둔다.

        if self.flrflag:
            std_t = [np.std(np.asarray(xs)), np.std(np.asarray(ys)), 1] # 표준 편차
        else:
            std_t = [np.std(np.asarray(xs)), np.std(np.asarray(ys)), np.std(np.asarray(zs))]


        max_t = [np.max(np.asarray(xs)), np.max(np.asarray(ys)), 1] # np.std(np.asarray(zs))] 최대값
        min_t = [np.min(np.asarray(xs)), np.min(np.asarray(ys)), 1] # np.std(np.asarray(zs))] 최소값

        print('mean', mean_t, 'standard deviation', std_t, 'max', max_t, 'min', min_t)
        np.savetxt(pose_stats_filename, np.vstack((mean_t, std_t, max_t, min_t)), fmt='%8.7f')

      else:
        mean_t, std_t, max_t, min_t = np.loadtxt(pose_stats_filename)


      grid_x = int((max_t[0] - min_t[0]) / GRID_SIZE) + 1 + 2
      grid_y = int((max_t[1] - min_t[1]) / GRID_SIZE) + 1 + 2

      grid_yaw = GRID_ANG_RPY + 2 + 1
      grid_roll = GRID_ANG_RPY + 2 + 1
      grid_pitch = GRID_ANG_RPY + 2 + 1

      grid_ang_roll = (max_q[0] - min_q[0]) / (grid_roll-3)
      grid_ang_pitch = (max_q[1] - min_q[1]) / (grid_pitch-3)
      grid_ang_yaw = (max_q[2] - min_q[2]) / (grid_yaw-3)
      min_ang_roll = min_q[0]
      min_ang_pitch = min_q[1]
      min_ang_yaw = min_q[2]

      self.num_classes = (grid_x)*(grid_y)
      self.num_floors = 1
      self.num_angles = grid_roll * grid_pitch * grid_yaw #72 #int(np.max(np.asarray(self.angles))+1)

      # convert pose to translation + log quaternion
      self.poses = np.empty((0, 6))
      self.classes = np.empty((0, self.num_classes))
      self.floors = np.empty((0, 1))
      self.angles = np.empty((0, self.num_angles))
      self.mclasses = np.empty((0, 1))
      self.mangclasses = np.empty((0, 1))

      for seq,x,y,z,q,q4 in zip(seqs,xs,ys,zs,qs,qs4):
        pss = [(x-mean_t[0])/std_t[0], (y-mean_t[1])/std_t[1], (z-mean_t[2])/std_t[2], q[0], q[1], q[2]]


        # for position
        c1 = int(np.floor((x - min_t[0]) / GRID_SIZE)) + 1
        r1 = int(np.floor((y - min_t[1]) / GRID_SIZE)) + 1
        c2 = int(np.ceil((x - min_t[0]) / GRID_SIZE)) + 1
        r2 = int(np.ceil((y - min_t[1]) / GRID_SIZE)) + 1

        dc1 = 1 - ( x - ((c1-1) * GRID_SIZE + min_t[0]) ) / GRID_SIZE #  [0,1]
        dc2 = 1 - dc1 #( (c2 * GRID_SIZE + min_t[0]) - x ) / GRID_SIZE #  [0,1]
        dr1 = 1 - ( y - ((r1-1) * GRID_SIZE + min_t[1]) ) / GRID_SIZE #  [0,1]
        dr2 = 1 - dr1 #( (r2 * GRID_SIZE + min_t[1]) - y ) / GRID_SIZE #  [0,1]

        pcls = np.zeros((1,self.num_classes))##
        id_c1r1 = c1 * grid_y + r1
        id_c2r1 = min(c2, grid_x-1) * grid_y + r1
        id_c1r2 = c1 * grid_y + min(r2, grid_y-1)
        id_c2r2 = min(c2, grid_x-1) * grid_y + min(r2, grid_y-1)

        pcls[:,id_c1r1] += dc1*dr1
        pcls[:,id_c2r1] += dc2*dr1
        pcls[:,id_c1r2] += dc1*dr2
        pcls[:,id_c2r2] += dc2*dr2

        pmcls =  int(np.floor((x - min_t[0]) / (GRID_SIZE))+1)  * grid_y + int(np.floor((y - min_t[1]) / (GRID_SIZE))+1)




        # for angle
        angle = q

        roll1 = int(np.floor((angle[0] - min_ang_roll) / grid_ang_roll )) + 1
        pitch1 = int(np.floor((angle[1] - min_ang_pitch) / grid_ang_pitch )) + 1
        yaw1 = int(np.floor((angle[2] - min_ang_yaw) / grid_ang_yaw )) + 1
        roll2 = int(np.ceil((angle[0] - min_ang_roll) / grid_ang_roll )) + 1
        pitch2 = int(np.ceil((angle[1] - min_ang_pitch) / grid_ang_pitch )) + 1
        yaw2 = int(np.ceil((angle[2] - min_ang_yaw) / grid_ang_yaw )) + 1

        droll1 = 1 - ( angle[0] - ((roll1-1) * grid_ang_roll + min_ang_roll) ) / grid_ang_roll #  [0,1]
        droll2 = 1 - droll1 #( (c2 * grid_size + min_t[0]) - x ) / grid_size #  [0,1]
        dpitch1 = 1 - ( angle[1] - ((pitch1-1) * grid_ang_pitch + min_ang_pitch) ) / grid_ang_pitch #  [0,1]
        dpitch2 = 1 - dpitch1 #( (r2 * grid_size + min_t[1]) - y ) / grid_size #  [0,1]
        dyaw1 = 1 - ( angle[2] - ((yaw1-1) * grid_ang_yaw + min_ang_yaw) ) / grid_ang_yaw #  [0,1]
        dyaw2 = 1 - dyaw1 #( (r2 * grid_size + min_t[1]) - y ) / grid_size #  [0,1]



        pang = np.zeros((1,self.num_angles))##
        id_r1p1y1 = (roll1 * grid_pitch + pitch1) * grid_yaw + yaw1
        id_r2p1y1 = (min(roll2, grid_roll-1) * grid_pitch + pitch1) * grid_yaw + yaw1
        id_r1p2y1 = (roll1 * grid_pitch + min(pitch2, grid_pitch-1)) * grid_yaw + yaw1
        id_r2p2y1 = (min(roll2, grid_roll-1) * grid_pitch + min(pitch2, grid_pitch-1)) * grid_yaw + yaw1

        id_r1p1y2 = (roll1 * grid_pitch + pitch1) * grid_yaw + min(yaw2, grid_yaw-1)
        id_r2p1y2 = (min(roll2, grid_roll-1) * grid_pitch + pitch1) * grid_yaw + min(yaw2, grid_yaw-1)
        id_r1p2y2 = (roll1 * grid_pitch + min(pitch2, grid_pitch-1)) * grid_yaw + min(yaw2, grid_yaw-1)
        id_r2p2y2 = (min(roll2, grid_roll-1) * grid_pitch + min(pitch2, grid_pitch-1)) * grid_yaw + min(yaw2, grid_yaw-1)


        pang[:,id_r1p1y1] += droll1*dpitch1*dyaw1
        pang[:,id_r2p1y1] += droll2*dpitch1*dyaw1
        pang[:,id_r1p2y1] += droll1*dpitch2*dyaw1
        pang[:,id_r2p2y1] += droll2*dpitch2*dyaw1

        pang[:,id_r1p1y2] += droll1*dpitch1*dyaw2
        pang[:,id_r2p1y2] += droll2*dpitch1*dyaw2
        pang[:,id_r1p2y2] += droll1*dpitch2*dyaw2
        pang[:,id_r2p2y2] += droll2*dpitch2*dyaw2


        pangmcls =  (int(np.floor(((angle[0] - min_ang_roll) / grid_ang_roll ) +1)  * grid_pitch) + int(np.floor((angle[1] - min_ang_pitch) / grid_ang_pitch)+1)) * grid_yaw + int(np.floor((angle[2] - min_ang_yaw) / grid_ang_yaw)+1)

        # pang =  0 #int(np.floor(_find_yaw_from_q(q[) / grid_ang))
        pflr = 0 #int(z/10)-1

        self.poses = np.vstack((self.poses, pss))
        self.classes = np.vstack((self.classes, pcls))
        self.floors = np.vstack((self.floors, pflr))
        self.angles = np.vstack((self.angles, pang))
        self.mclasses = np.vstack((self.mclasses, pmcls))
        self.mangclasses = np.vstack((self.mangclasses, pangmcls))




    def __getitem__(self, index):
        img = None
        while img is None:
            img = load_image(osp.join(self.bdir, self.c_imgs[index]))

            pose = self.poses[index]
            clss = self.classes[index]
            flrs = self.floors[index]
            angs = self.angles[index]
            mcls = self.mclasses[index]
            mangcls = self.mangclasses[index]

            index += 1
        index -= 1

        if self.target_transform is not None:
            pose = self.target_transform(pose)
            clss = self.target_transform(clss)
            angs = self.target_transform(angs)
            flrs = self.target_transform(flrs)
            mcls = self.target_transform(mcls)
            mangcls = self.target_transform(mangcls)


        if self.transform is not None:
            img = self.transform(img)
            if self.colordrop:
                zeroout_id = np.random.permutation(3)
                zeroout_id = zeroout_id[0]
                new_img = torch.zeros_like(img)
                new_img[zeroout_id,:,:] = img[zeroout_id,:,:]
                img = new_img

        return img, torch.cat([pose, clss, angs, flrs, mcls, mangcls])

    def __len__(self):
      return self.poses.shape[0]
