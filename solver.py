# -*- coding: utf-8 -*-
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Bernoulli

import numpy as np
import pandas as pd
import seaborn as sn
import json
from tqdm import tqdm, trange
from layers.yen_summarizer import simple_encoder_LSTM, Encoder, attentive_encoder_LSTM, attentive_encoder_decoder_LSTM  # , apply_weight_norm
from utils import TensorboardWriter
from ipdb import set_trace
from tqdm_logger import logger
from tqdm_logger import seclog, log
from tensorboardX import SummaryWriter
import vsum_tools
from matplotlib import pyplot as plt
from knapsack import knapsack_dp
# from feature_extraction import ResNetFeature


class Solver(object):
    def __init__(self, config=None, train_loader=None, test_loader=None):
        """Class that Builds, Trains and Evaluates SUM-GAN model"""
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader

    def build(self):

        # Build Modules
        # self.summarizer = simple_encoder_LSTM(
        #     input_size=self.config.input_size,
        #     hidden_size=self.config.hidden_size,
        #     num_layers=self.config.num_layers).cuda()
        self.summarizer = attentive_encoder_LSTM(
            input_size=self.config.input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers).cuda()
        # self.summarizer = attentive_encoder_decoder_LSTM(
        #     input_size=self.config.input_size,
        #     hidden_size=self.config.hidden_size,
        #     num_layers=self.config.num_layers).cuda()
        

        if self.config.mode == 'train':
            # Build Optimizers
            self.optimizer = optim.Adam(
                self.summarizer.parameters(),
                lr=self.config.lr)

            self.summarizer.train()

            # Tensorboard
            self.writer = TensorboardWriter(self.config.log_dir)

    #@staticmethod

    def classify_loss(self, predict, gt):
        loss = nn.BCELoss()
        # loss = nn.CrossEntropyLoss()
        return loss(predict, gt)

    def weighted_binary_cross_entropy(self, output, target, weights=[0.5,2]):    
        
        if weights is not None:
            assert len(weights) == 2
            
            loss = weights[1] * (target * torch.log(torch.clamp((output), min=1e-5, max=1-1e-5))) + \
                   weights[0] * ((1 - target) * torch.log(torch.clamp((1 - output), min=1e-5, max=1-1e-5)))
        else:
            loss = target * torch.log(output) + (1 - target) * torch.log(torch.clamp((1 - output), min=1e-5, max=1-1e-5))

        return torch.neg(torch.mean(loss))

    def diversity_similarity_loss(self, seq, actions, ignore_far_sim=True, temp_dist_thre=20, use_gpu=True):
        """
        Compute diversity reward and representativeness reward

        Args:
            seq: sequence of features, shape (1, seq_len, dim)
            actions: binary action sequence, shape (1, seq_len, 1)
            ignore_far_sim (bool): whether to ignore temporally distant similarity (default: True)
            temp_dist_thre (int): threshold for ignoring temporally distant similarity (default: 20)
            use_gpu (bool): whether to use GPU
        """
        # set_trace()
        _seq = seq.detach()
        _actions = actions.detach()
        pick_idxs = _actions.squeeze().nonzero().squeeze()
        num_picks = len(pick_idxs) if pick_idxs.ndimension() > 0 else 1
        
        if num_picks == 0:
            # give zero reward is no frames are selected
            reward = torch.tensor(0.)
            if use_gpu: reward = reward.cuda()
            return reward

        _seq = _seq.squeeze()
        n = _seq.size(0)


        # compute diversity reward
        if num_picks == 1:
            reward_div = torch.tensor(0.)
            if use_gpu: reward_div = reward_div.cuda()
        else:
            normed_seq = _seq / _seq.norm(p=2, dim=1, keepdim=True)
            dissim_mat = 1. - torch.matmul(normed_seq, normed_seq.t()) # dissimilarity matrix [Eq.4]
            dissim_submat = dissim_mat[pick_idxs,:][:,pick_idxs]
            if ignore_far_sim:
                # ignore temporally distant similarity
                pick_mat = pick_idxs.expand(num_picks, num_picks)
                temp_dist_mat = torch.abs(pick_mat - pick_mat.t())
                dissim_submat[temp_dist_mat > temp_dist_thre] = 1.
            reward_div = dissim_submat.sum() / (num_picks * (num_picks - 1.)) # diversity reward [Eq.3]

        # compute representativeness reward
        if num_picks == 1:
            reward_rep = torch.tensor(0.)
            if use_gpu: reward_rep = reward_rep.cuda()
            seclog(['WARNING:NUM_PICK==1','red'])
        else:
            dist_mat = torch.pow(_seq, 2).sum(dim=1, keepdim=True).expand(n, n)
            dist_mat = dist_mat + dist_mat.t()
            dist_mat.addmm_(1, -2, _seq, _seq.t())
            dist_mat1 = dist_mat[:,pick_idxs]
            dist_mat2 = dist_mat1.min(1, keepdim=True)[0]
            #reward_rep = torch.exp(torch.FloatTensor([-dist_mat.mean()]))[0] # representativeness reward [Eq.5]
            reward_rep = torch.exp(-dist_mat2.mean())

        # combine the two rewards
        reward = (reward_div + reward_rep) * 0.5

        return -reward

    def f_score(self, scores, video_gt, printf=False):
        # set_trace()
        # prob>0.5 set to 1
        scores = np.asarray(scores)
        video_gt = video_gt.numpy()
        scores[scores>=0.5]=1;scores[scores<0.5]=0
        acc = np.zeros(len(scores))
        acc[np.equal(scores,np.ones(len(scores)))&np.equal(video_gt,np.ones(len(video_gt)))] = 1
        if printf:
            seclog([f'scores_number: {np.sum(scores)}','cyan'])
            seclog([f'video_number: {np.sum(video_gt)}','blue'])

        P = np.sum(acc)/(np.sum(scores)+1e-12)
        R = np.sum(acc)/(np.sum(video_gt))
        f_score = 200*P*R/(P+R+1e-12)
        return P, R, f_score


    def train(self):
        logdir = './log/tvsum_11.5/'
        writer = SummaryWriter(logdir)
        step = 0
        old = 0
        early_stop = 0
        tbar = tqdm(total=self.config.n_epochs)
        for epoch_i in range(self.config.n_epochs):

            self.summarizer.train()
            # classify_loss_history = []
            loss_history = []
            ds_loss_history = []
            cls_loss_history = []   
            cost_history = []
            P_history = []
            R_history = []
            f_score_history = []
            for batch_i, [image_features, image_gt] in enumerate(self.train_loader):
                # image_pos_ = Variable(pos).cuda()
                image_features_ = Variable(image_features).cuda()

                # scores = self.summarizer(image_pos_, image_features_)
                scores, = self.summarizer(image_features_, return_attns=True)
                m = Bernoulli(scores)
                actions = m.sample()
                classify_loss = self.classify_loss(scores, Variable(image_gt.view(-1, 1, 1)).cuda())
                # classify_loss = self.weighted_binary_cross_entropy(scores, Variable(image_gt.view(-1,1,1)).cuda())
                ds_loss = self.diversity_similarity_loss(image_features_, actions)
                cost = 0.01 * (scores.mean() - 0.5)**2
                # log(f'loss = {classify_loss:.6f}')
                loss = classify_loss + ds_loss + cost
                
                # tbar.set_description(f'BCE_loss = {classify_loss.item():.4f}')
                tbar.set_description(f'loss={loss.item():.4f}')

                self.optimizer.zero_grad()
                # classify_loss.backward()
                loss.backward()
                self.optimizer.step()

                # classify_loss_history.append(classify_loss.item())
                loss_history.append(loss.item())
                cls_loss_history.append(classify_loss.item())
                ds_loss_history.append(ds_loss.item())
                cost_history.append(cost.item())
                # set_trace()
                scores = np.array(scores.data)
                # set_trace()
                P, R, train_f_score = self.f_score(scores.squeeze(), image_gt.squeeze())
                P_history.append(P)
                R_history.append(R)
                f_score_history.append(train_f_score)

                # writer.add_scalar('train_loss',classify_loss.cpu(),step)
                writer.add_scalar('train_loss', loss.cpu(), step)
                writer.add_scalar('ds_loss', ds_loss.cpu(), step)
                writer.add_scalar('classify_loss', classify_loss.cpu(), step)
                writer.add_scalar('train_P', P, step)
                writer.add_scalar('train_R', R, step)
                writer.add_scalar('train_f_score', train_f_score, step)

                step += 1

            # train_loss = np.mean(classify_loss_history)
            train_loss = np.mean(loss_history)
            
            seclog([
                f'loss = {train_loss.item():.3f} train_P: {np.mean(P_history):.3f} train_R: {np.mean(R_history):.3f} train_f-score: {np.mean(f_score_history):.3f}', 'green'])
            seclog([f'cls_loss = {np.mean(cls_loss_history):.3f} ds_loss = {np.mean(ds_loss_history):.3f} cost = {np.mean(cost_history):.3f}', 'green'])
            # Save parameters at checkpoint

            test_score = self.evaluate(epoch_i, step, writer)
            if not os.path.exists(logdir+'model'):
                os.mkdir(logdir+'model')
            if test_score > old:
                ckpt_path = str(logdir) + f'model/epoch{epoch_i}_score-{test_score:.5f}.pkl'
                torch.save(self.summarizer.state_dict(), ckpt_path)
                old = test_score
                early_stop = 0
            else:
                early_stop += 1
                log(early_stop)

            if early_stop == 20:
                break

            tbar.update(1)

    def evaluate(self, epoch_i, step, writer):
        self.summarizer.eval()
        

        # ======================== testing set test ================================ #
        # ========================================================================== #
        out_dict = {}
        acc_list = []
        loss_list = []
        # for [video_tensor, video_gt, video_name] in self.test_loader:
        for [video_tensor, gtsummary, gtscore, cps, num_frames, nfps, positions, user_summary, name] in self.test_loader:
            # video_name = video_name[0]
            video_name = name[0]
            video_gt = gtsummary
            # video_pos = Variable(pos).cuda()
            video_feature = Variable(video_tensor).cuda()
            scores, = self.summarizer(video_feature)
            # scores = self.summarizer(video_pos, video_feature)
            classify_loss = self.classify_loss(scores, Variable(video_gt).view(-1,1, 1).cuda())
            # classify_loss = self.weighted_binary_cross_entropy(scores, Variable(video_gt.view(-1,1,1)).cuda())
            scores = scores.cpu().detach().numpy().squeeze()
            cps = cps.numpy().squeeze(0)
            num_frames = num_frames.numpy().squeeze(0)
            nfps = nfps.numpy().squeeze(0).tolist()
            positions = positions.numpy().squeeze(0)
            user_summary = user_summary.numpy().squeeze(0)
            video_name = name[0]
            # print(user_summary.shape[0])
            
            machine_summary = vsum_tools.generate_summary(scores, cps, num_frames, nfps, positions)
            fm, _, _ = vsum_tools.evaluate_summary(machine_summary, user_summary, 'avg')



            # out_dict[video_name] = scores.squeeze(1).tolist()
            # P, R, f_score = self.f_score(scores.squeeze(), video_gt.squeeze(), True)
            loss_list.append(classify_loss.item())
            # acc_list.append(f_score)
            acc_list.append(fm)
            # log(f'video_name: {video_name:<9} P: {P:.3f} R:{R:.3f} f_score:{f_score:.3f}')
            log(f'video_name: {video_name:<9} f_score:{fm:.3f}')

        seclog([f'testing loss : {np.mean(loss_list):.3f} mean of f_score : {np.mean(acc_list):.3f}', 'light_red'])
        # seclog([f'testing f_score: {np.mean(acc_list):.3f}', 'blue'])
        # writer.add_scalar('test_loss',np.mean(loss_list),step)
        writer.add_scalar('test_f_score', np.mean(acc_list), step)
        writer.add_scalar('test_loss', np.mean(loss_list), step)

        return np.mean(acc_list)

    def pretrain(self):
        pass

    def visualize(self):
        # model_path = 'meeting2/tvsum/tvsum_standard_3layer18head/model/score-0.60574.pkl'
        model_path = 'log/tvsum_11.5_atten_only_posffn/model/epoch12_score-0.18391.pkl'
        self.summarizer.load_state_dict(torch.load(model_path))
        self.summarizer.eval()
        

        # ======================== testing set test ================================ #
        # ========================================================================== #
        out_dict = {}
        acc_list = []
        loss_list = []
        for [video_tensor, gtsummary, gtscore, cps, num_frames, nfps, positions, user_summary, name] in self.test_loader:
            video_name = name[0]
            video_feature = Variable(video_tensor).cuda()
            scores, att_map = self.summarizer(video_feature, return_attns=True)
            scores = scores.cpu().detach().numpy().squeeze()
            gtsummary = gtsummary.numpy().squeeze(0)
            gtscore = gtscore.numpy().squeeze(0)
            cps = cps.numpy().squeeze(0)
            num_frames = num_frames.numpy().squeeze(0)
            nfps = nfps.numpy().squeeze(0).tolist()
            positions = positions.numpy().squeeze(0)
            user_summary = user_summary.numpy().squeeze(0)
            save_path = f'log/tvsum_2layer8head_11.1/feature_map/'
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            save_path = save_path + f'{video_name}/'
            
            machine_summary = vsum_tools.generate_summary(scores, cps, num_frames, nfps, positions)
            fm, P, R = vsum_tools.evaluate_summary(machine_summary, user_summary, 'avg')

            user_score = np.zeros(len(user_summary[0]))
            for user in user_summary:
                user_score += user

            # [seq, head, layer, seq]

            


            # =========================== Encoder attentive Decoder ================================== #
            # [seq, head, layer, seq]
            attention_map = np.zeros((len(att_map), att_map[0][0][0].shape[0], len(att_map[0][0]), len(att_map)))
            for i in range(len(att_map)):
                for j in range(len(att_map[0][0])):
                    attention_map[i, :, j, :] = att_map[i][0][j].cpu().detach().numpy().squeeze()
            for layer in range(attention_map.shape[2]):
                for h in range(attention_map.shape[1]):


                    df_cm = pd.DataFrame(attention_map[60:,h,layer,:], index = [i for i in range(attention_map.shape[0]-60)],
                          columns = [i for i in range(attention_map.shape[0])])
                    # plt.figure(figsize = (10,7))
                    # sn.heatmap(df_cm, annot=True)
                    f, ax= plt.subplots(figsize = (14*2, 14*2))

                    sn.heatmap(df_cm,cmap='YlGnBu', linewidths = 0.05, ax = ax)
                    # sn.heatmap(df_cm, annot=True, ax = ax)
                    # 設定Axes的標題
                    ax.set_title(f'Accuracy = {fm*100:.2f}')

                    if not os.path.exists(save_path):
                        os.mkdir(save_path)
                    f.savefig(save_path+f'layer{layer}head_{h}.jpg', dpi=100, bbox_inches='tight')
                    plt.close()


            # ======================================================================================== #





            # =========================== original =================================================== #
            # att_map = att_map[0]
            # for i in range(3):
            #     att_map0 = att_map[i].cpu().detach().numpy()
            #     for h in range(len(att_map0)):


            #         df_cm = pd.DataFrame(att_map0[h], index = [i for i in range(att_map0[h].shape[0])],
            #               columns = [i for i in range(att_map0[h].shape[1])])
            #         # plt.figure(figsize = (10,7))
            #         # sn.heatmap(df_cm, annot=True)
            #         f, ax= plt.subplots(figsize = (14*2, 14*2))

            #         sn.heatmap(df_cm,cmap='YlGnBu', linewidths = 0.05, ax = ax)
            #         # sn.heatmap(df_cm, annot=True, ax = ax)
            #         # 設定Axes的標題
            #         ax.set_title(f'Accuracy = {fm*100:.2f}')

                    
            #         if not os.path.exists(save_path):
            #             os.mkdir(save_path)
            #         f.savefig(save_path+f'layer{i}head_{h}.jpg', dpi=100, bbox_inches='tight')
            #         plt.close()
            # ======================================================================================= #



            # plot score vs gtscore
            fig, axs = plt.subplots(3)
            n = len(gtscore)

            limits = int(math.floor(len(scores) * 0.15))
            order = np.argsort(scores)[::-1].tolist()
            picks = []
            total_len = 0
            for i in order:
                if total_len < limits:
                    picks.append(i)
                    total_len += 1
            y_scores = np.zeros(len(scores))
            y_scores[picks] = gtscore[picks]

            y_summary = np.zeros(len(scores))
            y_summary[picks] = gtsummary[picks]

            # machine_summary = user_score*machine_summary
            # set_trace()
            axs[0].bar(range(n), gtsummary, width=1, color='lightgray')
            axs[0].bar(range(n), y_summary, width=1, color='orange')
            axs[0].set_title("tvsum {} F-score {:.1%}".format(video_name, fm))
            
            axs[1].bar(range(n), gtscore, width=1, color='lightgray')
            axs[1].bar(range(n), y_scores, width=1, color='orange')
            plt.xticks(np.linspace(0,n,n/20, endpoint=False,dtype=int))
            
            axs[2].bar(range(n), scores.tolist(), width=1, color='orange')
            plt.xticks(np.linspace(0,n,n/20, endpoint=False,dtype=int))
            # axs[2].bar(range(len(user_score)), user_score, width=1, color='lightgray')
            # axs[2].bar(range(len(user_score)), user_score*machine_summary, width=1, color='orange')
            
            # for i in range(15):
            #     axs[i+3].bar(range(len(user_score)), user_summary[i], width=1, color='lightgray')
            #     axs[i+3].bar(range(len(user_score)), user_summary[i]*machine_summary, width=1, color='orange')
                # print(i)

            # fig = plt.figure(figsize=(10,60))
            fig.tight_layout() 
            fig.savefig(save_path+f'visualization3.png', bbox_inches='tight')
            plt.close()






            acc_list.append(fm)
            log(f'video_name: {video_name:<9} P: {P:.3f} R:{R:.3f} f_score:{fm:.3f}')
            break

        seclog([f'testing f_score: {np.mean(acc_list):.3f}', 'blue'])




if __name__ == '__main__':
    pass
