# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Salesforce.com, Inc
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause#

import numpy as np
from tqdm import tqdm
import random

import torch
import torch.nn as nn
from torch.optim import Adam

from torch.utils.data import DataLoader, RandomSampler
from datasets import RecWithContrastiveLearningDataset
from modules import NCELoss
from utils import recall_at_k, ndcg_k, get_metric, get_user_seqs, nCr

class Trainer:
    def __init__(self, model, train_dataloader,
                 eval_dataloader,
                 test_dataloader, 
                 args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model

        self.total_augmentaion_pairs = nCr(self.args.n_views, 2)
        if self.cuda_condition:
            self.model.cuda()
        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        self.cf_criterion = NCELoss(self.args.temperature, self.device)
        print("self.cf_criterion:", self.cf_criterion.__class__.__name__)
        
    def __refresh_training_dataset(self, item_embeddings):
        """
        use for updating item embedding
        """
        user_seq, _, _, _ = get_user_seqs(self.args.data_file)
        # training data for node classification
        train_dataset = RecWithContrastiveLearningDataset(self.args, user_seq, data_type='train')
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.args.batch_size)
        return train_dataloader
        
    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, full_sort=full_sort, train=False)

    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, full_sort=full_sort, train=False)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError

    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": '{:.4f}'.format(HIT_1), "NDCG@1": '{:.4f}'.format(NDCG_1),
            "HIT@5": '{:.4f}'.format(HIT_5), "NDCG@5": '{:.4f}'.format(NDCG_5),
            "HIT@10": '{:.4f}'.format(HIT_10), "NDCG@10": '{:.4f}'.format(NDCG_10),
            "MRR": '{:.4f}'.format(MRR),
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix)

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
            "HIT@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
            "HIT@20": '{:.4f}'.format(recall[3]), "NDCG@20": '{:.4f}'.format(ndcg[3])
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size) # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1) # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float() # [batch*seq_len]
        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss

    def predict_sample(self, seq_out, test_neg_sample):
        # [batch 100 hidden_size]
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        # [batch hidden_size]
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred

class CoSeRecTrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, 
                 args):
        super(CoSeRecTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, 
            args
        )

    def _one_pair_contrastive_learning(self, inputs):
        '''
        contrastive learning given one pair sequences (batch)
        inputs: [batch1_augmented_data, batch2_augmentated_data]
        '''
        cl_batch = torch.cat(inputs, dim=0)
        # print("contrastive learning batches:",  cl_batch.shape)
        cl_batch = cl_batch.to(self.device)
        # cl_sequence_output = self.model.transformer_encoder(cl_batch,isTrain=True)
        if self.args.model_augmentation:
            cl_sequence_output = self.model.transformer_encoder(cl_batch,isTrain=True)
        else:
            cl_sequence_output = self.model.transformer_encoder(cl_batch)
        # cf_sequence_output = cf_sequence_output[:, -1, :]
        cl_sequence_flatten = cl_sequence_output.view(cl_batch.shape[0], -1)
        # cf_output = self.projection(cf_sequence_flatten)
        batch_size = cl_batch.shape[0]//2
        cl_output_slice = torch.split(cl_sequence_flatten, batch_size)
        cl_loss = self.cf_criterion(cl_output_slice[0], 
                                cl_output_slice[1])
        return cl_loss
    
    def _one_pair_cl_model_aug(self, inputs):
        '''the contrastive learning with model augmentation:
        each branch adopts different models
        '''
        cl_sequence_outout = []
        for cl_batch in inputs: # each encoder has distinct model structures 
            cl_batch = cl_batch.to(self.device)
            cl_sequence_embedding = self.model.transformer_encoder(cl_batch, isTrain=True)
            cl_sequence_flatten = cl_sequence_embedding.view(cl_batch.shape[0], -1)
            cl_sequence_outout.append(cl_sequence_flatten)
            # print('cl_sequence_flatten:', cl_sequence_flatten.shape)
        cl_loss = self.cf_criterion(cl_sequence_outout[0], cl_sequence_outout[1])
        return cl_loss
    
    def _cl_encoder(self, inputs, original=None, encoder=None, en_weight=0.1):
        '''
        the contrastive learning loss with static encoder
        '''
        cl_batch = torch.cat(inputs, dim=0)
        # print("contrastive learning batches:",  cl_batch.shape)
        cl_batch = cl_batch.to(self.device)
        # cl_sequence_output = self.model.transformer_encoder(cl_batch,isTrain=True)
        if self.args.model_augmentation:
            cl_sequence_output = self.model.transformer_encoder(cl_batch,isTrain=True)
        else:
            cl_sequence_output = self.model.transformer_encoder(cl_batch)
        # generate the embedding of the original sequence output from the encoder. 
        ori_sequence_output = encoder(original)
        
        cl_sequence_flatten = cl_sequence_output.view(cl_batch.shape[0], -1)
        # cf_output = self.projection(cf_sequence_flatten)
        batch_size = cl_batch.shape[0]//2
        cl_output_slice = torch.split(cl_sequence_flatten, batch_size)
        cl_loss = self.cf_criterion(cl_output_slice[0], 
                                cl_output_slice[1])
        for slice in cl_output_slice:                        
            cl_loss += en_weight*self.cf_criterion(slice, ori_sequence_output)
        return cl_loss
            


    def iteration(self, epoch, dataloader, full_sort=True, train=True):

        str_code = "train" if train else "test"

        # Setting the tqdm progress bar

        if train:
            self.model.train()
            rec_avg_loss = 0.0
            cl_individual_avg_losses = [0.0 for i in range(self.total_augmentaion_pairs)]
            cl_sum_avg_loss = 0.0
            joint_avg_loss = 0.0

            print(f"rec dataset length: {len(dataloader)}")
            rec_cf_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))

            for i, (rec_batch, cl_batches) in rec_cf_data_iter:
                '''
                rec_batch shape: key_name x batch_size x feature_dim
                cl_batches shape: 
                    list of n_views x batch_size x feature_dim tensors
                '''
                # 0. batch_data will be sent into the device(GPU or CPU)
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                _, input_ids, target_pos, target_neg, _ = rec_batch

                # ---------- recommendation task ---------------#
                if self.args.model_augmentation:
                    sequence_output = self.model.transformer_encoder(input_ids, model_aug=self.args.rec_model_aug, isTrain=True)
                else:
                    sequence_output = self.model.transformer_encoder(input_ids)
                    

                rec_loss = self.cross_entropy(sequence_output, target_pos, target_neg)

                # ---------- contrastive learning task -------------#
                cl_losses = []
                for cl_batch in cl_batches:
                    # print("contrastive learning batches:",  cl_batch[0].shape)
                    if self.args.model_aug_in_batch == 'same':
                        cl_loss = self._one_pair_contrastive_learning(cl_batch)
                    elif self.args.model_aug_in_batch == 'distinct':
                        cl_loss = self._one_pair_cl_model_aug(cl_batch)
                    elif self.args.model_aug_in_batch == 'sasrec-static':
                        cl_loss = self._cl_encoder(cl_batch,input_ids,self.args.pretrain_encoder,en_weight=self.args.en_weight)
                    else:
                        raise ValueError("no %s model augmentation methods" %self.args.model_aug_in_batch)
                    cl_losses.append(cl_loss)

                joint_loss = self.args.rec_weight * rec_loss
                for cl_loss in cl_losses:
                    joint_loss += self.args.cf_weight * cl_loss
                self.optim.zero_grad()
                joint_loss.backward()
                self.optim.step()

                rec_avg_loss += rec_loss.item()

                for i, cl_loss in enumerate(cl_losses):
                    cl_individual_avg_losses[i] += cl_loss.item()
                    cl_sum_avg_loss += cl_loss.item()
                joint_avg_loss += joint_loss.item()


            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / len(rec_cf_data_iter)),
                "joint_avg_loss": '{:.4f}'.format(joint_avg_loss / len(rec_cf_data_iter)),
                "cl_avg_loss": '{:.4f}'.format(cl_sum_avg_loss / (len(rec_cf_data_iter)*self.total_augmentaion_pairs)),
            }
            for i, cl_individual_avg_loss in enumerate(cl_individual_avg_losses):
                post_fix['cl_pair_'+str(i)+'_loss'] = '{:.4f}'.format(cl_individual_avg_loss / len(rec_cf_data_iter))

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

        else:
            rec_data_iter = tqdm(enumerate(dataloader),
                                  desc="Recommendation EP_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
            self.model.eval()

            pred_list = None

            if full_sort:
                answer_list = None
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers = batch
                    recommend_output = self.model.transformer_encoder(input_ids)

                    recommend_output = recommend_output[:, -1, :]
                    # recommendation results

                    rating_pred = self.predict_full(recommend_output)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                    # argpartition T: O(n)  argsort O(nlogn)
                    ind = np.argpartition(rating_pred, -20)[:, -20:]
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                return self.get_full_sort_score(epoch, answer_list, pred_list)

            else:
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers, sample_negs = batch
                    recommend_output = self.model.finetune(input_ids)
                    test_neg_items = torch.cat((answers, sample_negs), -1)
                    recommend_output = recommend_output[:, -1, :]

                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)

                return self.get_sample_scores(epoch, pred_list)

class SRMATrainer(Trainer):
    
    def __init__(self, model, cs_encoder,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, 
                 args):
        super(SRMATrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, 
            args
        )
        self.cs_encoder = cs_encoder
        if self.cuda_condition:
            self.cs_encoder.cuda()

    def _one_pair_contrastive_learning(self, inputs):
        '''
        contrastive learning given one pair sequences (batch)
        inputs: [batch1_augmented_data, batch2_augmentated_data]
        '''
        cl_batch = torch.cat(inputs, dim=0)
        cl_batch = cl_batch.to(self.device)
        if self.args.model_augmentation:
            cl_sequence_output = self.model.transformer_encoder(cl_batch,isTrain=True)
        else:
            cl_sequence_output = self.model.transformer_encoder(cl_batch)
        cl_sequence_flatten = cl_sequence_output.view(cl_batch.shape[0], -1)
        batch_size = cl_batch.shape[0]//2
        cl_output_slice = torch.split(cl_sequence_flatten, batch_size)
        cl_loss = self.cf_criterion(cl_output_slice[0], 
                                cl_output_slice[1])
        return cl_loss
    
    def _one_pair_cl_model_aug(self, inputs):
        '''the contrastive learning with model augmentation:
        each branch adopts different models
        '''
        cl_sequence_outout = []
        for cl_batch in inputs: # each encoder has distinct model structures 
            cl_batch = cl_batch.to(self.device)
            cl_sequence_embedding = self.model.transformer_encoder(cl_batch, isTrain=True)
            cl_sequence_flatten = cl_sequence_embedding.view(cl_batch.shape[0], -1)
            cl_sequence_outout.append(cl_sequence_flatten)
        cl_loss = self.cf_criterion(cl_sequence_outout[0], cl_sequence_outout[1])
        return cl_loss
    
    def _cl_encoder(self, inputs, original=None, encoder=None, en_weight=0.1):
        '''
        the contrastive learning loss with static encoder
        '''
        cl_batch = torch.cat(inputs, dim=0)
        # print("contrastive learning batches:",  cl_batch.shape)
        cl_batch = cl_batch.to(self.device)
        # cl_sequence_output = self.model.transformer_encoder(cl_batch,isTrain=True)
        if self.args.model_augmentation:
            cl_sequence_output = self.model.transformer_encoder(cl_batch,isTrain=True)
        else:
            cl_sequence_output = self.model.transformer_encoder(cl_batch)
        
        cl_sequence_flatten = cl_sequence_output.view(cl_batch.shape[0], -1)
        # cf_output = self.projection(cf_sequence_flatten)
        batch_size = cl_batch.shape[0]//2
        cl_output_slice = torch.split(cl_sequence_flatten, batch_size)
        cl_loss = self.cf_criterion(cl_output_slice[0], 
                                cl_output_slice[1])
        # generate the embedding of the original sequence output from the pretrained encoder. 
        ori_sequence_output = encoder.transformer_encoder(original)
        ori_sequence_flatten = ori_sequence_output.view(ori_sequence_output.shape[0], -1)
        for slice in cl_output_slice:                        
            cl_loss += en_weight*self.cf_criterion(slice, ori_sequence_flatten)
        return cl_loss
            
    def iteration(self, epoch, dataloader, full_sort=True, train=True):

        str_code = "train" if train else "test"

        # Setting the tqdm progress bar

        if train:
            self.model.train()
            rec_avg_loss = 0.0
            cl_individual_avg_losses = [0.0 for i in range(self.total_augmentaion_pairs)]
            cl_sum_avg_loss = 0.0
            joint_avg_loss = 0.0

            print(f"rec dataset length: {len(dataloader)}")
            rec_cf_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))

            for i, (rec_batch, cl_batches) in rec_cf_data_iter:
                '''
                rec_batch shape: key_name x batch_size x feature_dim
                cl_batches shape: 
                    list of n_views x batch_size x feature_dim tensors
                '''
                # 0. batch_data will be sent into the device(GPU or CPU)
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                _, input_ids, target_pos, target_neg, _ = rec_batch

                # ---------- recommendation task ---------------#
                if self.args.model_augmentation:
                    sequence_output = self.model.transformer_encoder(input_ids, model_aug=self.args.rec_model_aug, isTrain=True)
                else:
                    sequence_output = self.model.transformer_encoder(input_ids)
                    

                rec_loss = self.cross_entropy(sequence_output, target_pos, target_neg)

                # ---------- contrastive learning task -------------#
                cl_losses = []
                for cl_batch in cl_batches:
                    # print("contrastive learning batches:",  cl_batch[0].shape)
                    if self.args.model_aug_in_batch == 'same':
                        cl_loss = self._one_pair_contrastive_learning(cl_batch)
                    elif self.args.model_aug_in_batch == 'distinct':
                        cl_loss = self._one_pair_cl_model_aug(cl_batch)
                    elif self.args.model_aug_in_batch == 'sasrec-static':
                        cl_loss = self._cl_encoder(cl_batch, input_ids, encoder=self.cs_encoder, en_weight=self.args.en_weight)
                    else:
                        raise ValueError("no %s model augmentation methods" %self.args.model_aug_in_batch)
                    cl_losses.append(cl_loss)

                joint_loss = self.args.rec_weight * rec_loss
                for cl_loss in cl_losses:
                    joint_loss += self.args.cf_weight * cl_loss
                self.optim.zero_grad()
                joint_loss.backward()
                self.optim.step()

                rec_avg_loss += rec_loss.item()

                for i, cl_loss in enumerate(cl_losses):
                    cl_individual_avg_losses[i] += cl_loss.item()
                    cl_sum_avg_loss += cl_loss.item()
                joint_avg_loss += joint_loss.item()


            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / len(rec_cf_data_iter)),
                "joint_avg_loss": '{:.4f}'.format(joint_avg_loss / len(rec_cf_data_iter)),
                "cl_avg_loss": '{:.4f}'.format(cl_sum_avg_loss / (len(rec_cf_data_iter)*self.total_augmentaion_pairs)),
            }
            for i, cl_individual_avg_loss in enumerate(cl_individual_avg_losses):
                post_fix['cl_pair_'+str(i)+'_loss'] = '{:.4f}'.format(cl_individual_avg_loss / len(rec_cf_data_iter))

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

        else:
            rec_data_iter = tqdm(enumerate(dataloader),
                                  desc="Recommendation EP_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
            self.model.eval()

            pred_list = None

            if full_sort:
                answer_list = None
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers = batch
                    recommend_output = self.model.transformer_encoder(input_ids)

                    recommend_output = recommend_output[:, -1, :]
                    # recommendation results

                    rating_pred = self.predict_full(recommend_output)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                    # argpartition T: O(n)  argsort O(nlogn)
                    ind = np.argpartition(rating_pred, -20)[:, -20:]
                    # 
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    # 
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    # 
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                return self.get_full_sort_score(epoch, answer_list, pred_list)

            else:
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers, sample_negs = batch
                    recommend_output = self.model.finetune(input_ids)
                    test_neg_items = torch.cat((answers, sample_negs), -1)
                    recommend_output = recommend_output[:, -1, :]

                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)

                return self.get_sample_scores(epoch, pred_list)


