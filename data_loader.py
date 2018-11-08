
# -*- coding: utf-8 -*-
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
from pathlib import Path

from feature_extraction import resnet_transform
import h5py
import numpy as np
import json
from ipdb import set_trace

class VideoData(Dataset):
    # def __init__(self, root, mode, cross_index, preprocessed=True, transform=resnet_transform, with_name=False, valid=False):
    def __init__(self, root, mode, preprocessed=True, transform=resnet_transform, with_name=False, valid=False):
        self.root = root
        self.mode = mode
        self.preprocessed = preprocessed
        self.transform = transform
        self.with_name = with_name
        self.valid = valid
        # self.cross_index = cross_index
        #self.video_list = list(self.root.iterdir())
        fpath = self.root.joinpath('tvsum_splits.json')
        with open(fpath, 'r') as f:
            video_name = json.load(f)
            video_name = video_name[0]
            if self.mode.lower() == 'train':
                video_name = video_name['train_keys']
                # self.video_len = len(video_name)/5
            else:
                video_name = video_name['test_keys']
                # self.video_len = len(video_name)/5

        # fpath = self.root.joinpath('summe_all_train_splits.json')
        # with open(fpath, 'r') as f:
        #     video_name2 = json.load(f)
        #     video_name2 = video_name2[0]['train_keys']
        #     # set_trace()

        # fpath = self.root.joinpath('ovp_splits.json')
        # with open(fpath, 'r') as f:
        #     video_name3 = json.load(f)
        #     video_name3 = video_name3[0]['train_keys']

        # fpath = self.root.joinpath('youtube_splits.json')
        # with open(fpath, 'r') as f:
        #     video_name4 = json.load(f)
        #     video_name4 = video_name4[0]['train_keys']
        


        # self.video_box = []
        # for a in range(5):
        #     self.video_box.append(video_name[int(a*self.video_len):int((a+1)*self.video_len)])
        
        # self.video_list = []
        # if self.mode.lower() == 'train':
        #     for i in range(5):
        #         if i != self.cross_index:
        #             self.video_list = self.video_list + self.video_box[cross_index]
        # elif self.mode.lower() == 'valid':
        #     self.video_list = self.video_box[cross_index]
        # else:
        #     self.video_list = video_name
        




        self.video_list = video_name
        # self.video_list2 = video_name2
        # self.video_list3 = video_name3
        # self.video_list4 = video_name4
        self.vpath = self.root.joinpath('eccv16_dataset_tvsum_google_pool5.h5')
        # self.v2path = self.root.joinpath('eccv16_dataset_summe_google_pool5.h5')
        # self.v3path = self.root.joinpath('eccv16_dataset_ovp_google_pool5.h5')
        # self.v4path = self.root.joinpath('eccv16_dataset_youtube_google_pool5.h5')
        if self.mode.lower() == 'train':
            self.features_list = []
            self.gtsummary_list = []
            with h5py.File(self.vpath, 'r') as f:
                for name in self.video_list:
                    self.features_list.append(f[name]['features'][...])
                    self.gtsummary_list.append(f[name]['gtsummary'][...])
            # with h5py.File(self.v2path, 'r') as f:
            #     for name2 in self.video_list2:
            #         self.features_list.append(f[name2]['features'][...])
            #         self.gtsummary_list.append(f[name2]['gtsummary'][...])
            # with h5py.File(self.v3path, 'r') as f:
            #     for name3 in self.video_list3:
            #         self.features_list.append(f[name3]['features'][...])
            #         self.gtsummary_list.append(f[name3]['gtsummary'][...])
            # with h5py.File(self.v4path, 'r') as f:
            #     for name4 in self.video_list4:
            #         self.features_list.append(f[name4]['features'][...])
            #         self.gtsummary_list.append(f[name4]['gtsummary'][...])
    def __len__(self):
        return len(self.video_list)
        # if self.mode.lower() == 'train':
        #     # print(len(self.features_list))
        #     return len(self.features_list)
        # else:
        #     # print(len(self.video_list))
        #     return len(self.video_list)

    def __getitem__(self, index):
        # name = self.video_list[index]
        with h5py.File(self.vpath, 'r') as f:
            if self.with_name:
                name = self.video_list[index]
                # pos = np.arange(len(f[name]['features'][...]))+1
                gtsummary = f[name]['gtsummary'][...]
                gtscore = f[name]['gtscore'][...]
                feature = f[name]['features'][...]
                cps = f[name]['change_points'][...]
                num_frames = f[name]['n_frames'][()]
                nfps = f[name]['n_frame_per_seg'][...]
                positions = f[name]['picks'][...]
                user_summary = f[name]['user_summary'][...]
                
                return feature, gtsummary, gtscore, cps, num_frames, nfps, positions, user_summary, name
                # return pos, feature, cps, num_frames, nfps, positions, user_summary, name

            else:
                # pos = np.arange(len(f[name]['features'][...]))+1
                # features = f[name]['features'][...]
                # gtsummary = f[name]['gtsummary'][...]
                features = self.features_list[index]
                gtsummary = self.gtsummary_list[index]
                pos = np.arange(len(features))+1
                
                return features, gtsummary
                # return pos, features, gtsummary

        # if self.with_name:
        #     return torch.Tensor(np.array(self.features_list[index])), \
        #            torch.Tensor(np.array(self.gtsummary[index])), \
        #            self.video_list[index]
        # else:
        #     return torch.Tensor(np.array(self.features_list[index])), \
        #            torch.Tensor(np.array(self.gtsummary[index]))


# def get_loader(root, mode, cross_index):
def get_loader(root, mode):

    if mode.lower() == 'train':
        # return DataLoader(VideoData(root, mode, cross_index), batch_size=1, shuffle=True)
        return DataLoader(VideoData(root, mode), batch_size=1, shuffle=True)
    # elif mode.lower() == 'valid':
    #     return DataLoader(VideoData(root, mode, cross_index, with_name=True), batch_size=1, shuffle=False)
    else:
        # return DataLoader(VideoData(root, mode, cross_index, with_name=True), batch_size=1, shuffle=False)
        return DataLoader(VideoData(root, mode, with_name=True), batch_size=1, shuffle=False)


if __name__ == '__main__':
    pass
