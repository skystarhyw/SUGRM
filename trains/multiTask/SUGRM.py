import os
import logging
import pickle as plk
from tqdm import tqdm

import torch
from torch import optim

from utils.functions import dict_to_str
from utils.metricsTop import MetricsTop

logger = logging.getLogger('MSA')

class SUGRM():
    def __init__(self, args):
        assert args.train_mode == 'regression'

        self.args = args
        self.args.tasks = "MTAV"
        self.metrics = MetricsTop(args.train_mode).getMetics(args.datasetName)
        self.feat_p3 = []
        self.feat_n3 = []

        self.feature_map = {
            'fusion': torch.zeros(args.train_samples, args.common_dim, requires_grad=False).to(args.device)
        }

        self.center_map = {
            'fusion': {
                'pos': torch.zeros(args.common_dim, requires_grad=False).to(args.device),
                'neg': torch.zeros(args.common_dim, requires_grad=False).to(args.device),
            }
        }

        # new labels
        self.label_map = {
            'fusion': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'text': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'audio': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'vision': torch.zeros(args.train_samples, requires_grad=False).to(args.device)
        }

        self.name_map = {
            'M': 'fusion',
            'T': 'text',
            'A': 'audio',
            'V': 'vision'
        }

    def do_train(self, model, dataloader):
        bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_params = list(model.Model.text_model.named_parameters())
        bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
        bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
        spt_params = list(model.Model.spe.named_parameters())
        spt_params = [p for n, p in spt_params]
        model_params_other = [p for n, p in list(model.Model.named_parameters()) if 'text_model' not in n and \
                              'spe' not in n]
        optimizer_grouped_parameters = [
            {'params': bert_params_decay, 'weight_decay': self.args.weight_decay_bert,
             'lr': self.args.learning_rate_bert},
            {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': self.args.learning_rate_bert},
            {'params': spt_params, 'weight_decay': self.args.weight_decay_text,
             'lr': 5e-4},
            {'params': model_params_other, 'weight_decay': self.args.weight_decay_other,
             'lr': self.args.learning_rate_other}
        ]
        optimizer = optim.Adam(optimizer_grouped_parameters)

        saved_labels = {}
        # init labels
        logger.info("Init labels...")
        with tqdm(dataloader['train']) as td:
            for batch_data in td:
                labels_m = batch_data['labels']['M'].view(-1).to(self.args.device)
                indexes = batch_data['index'].view(-1)
                self.init_labels(indexes, labels_m)
                self.save_index()

        # initilize results
        logger.info("Start training...")
        self.epochs, best_epoch = 0, 0
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        # loop util earlystop
        while True: 
            self.epochs += 1
            # train
            y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
            y_true = {'M': [], 'T': [], 'A': [], 'V': []}
            model.train()
            train_loss = 0.0
            left_epochs = self.args.update_epochs
            ids = []
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1

                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    indexes = batch_data['index'].view(-1)
                    cur_id = batch_data['id']
                    ids.extend(cur_id)

                    if not self.args.need_data_aligned:
                        audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                        vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                    else:
                        audio_lengths, vision_lengths = 0, 0

                    # forward
                    outputs = model(text, (audio, audio_lengths), (vision, vision_lengths))

                    # store results
                    for m in self.args.tasks:
                        y_pred[m].append(outputs[m].cpu())
                        y_true[m].append(self.label_map[self.name_map[m]][indexes].cpu())
                    # compute loss
                    loss = 0.0
                    for m in self.args.tasks:
                        loss += self.weighted_loss(outputs[m], self.label_map[self.name_map[m]][indexes], \
                                                    indexes=indexes, mode=self.name_map[m])

                    # backward
                    loss.backward()
                    train_loss += loss.item()
                    # update features
                    f_fusion = outputs['Feature_f'].detach()
                    f_text = outputs['Feature_t'].detach()
                    f_audio = outputs['Feature_a'].detach()
                    f_vision = outputs['Feature_v'].detach()

                    if self.epochs > 1:
                        self.update_labels(f_fusion, f_text, f_audio, f_vision, self.epochs, indexes, outputs)

                    self.update_features(f_fusion, indexes)
                    self.update_centers()
                    
                    # update parameters
                    if not left_epochs:
                        # update
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    # update
                    optimizer.step()
            train_loss = train_loss / len(dataloader['train'])
            logger.info("TRAIN-(%s) (%d/%d/%d)>> loss: %.4f " % (self.args.modelName, \
                        self.epochs-best_epoch, self.epochs, self.args.cur_time, train_loss))

            for m in self.args.tasks:
                pred, true = torch.cat(y_pred[m]), torch.cat(y_true[m])
                train_results = self.metrics(pred, true)
                logger.info('%s: >> ' %(m) + dict_to_str(train_results))

            # validation
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            cur_valid = val_results[self.args.KeyEval]
            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, self.epochs
                # save model
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)
            # save labels
            if self.args.save_labels:
                tmp_save = {k: v.cpu().numpy() for k, v in self.label_map.items()}
                tmp_save['ids'] = ids
                saved_labels[self.epochs] = tmp_save
            # early stop
            if self.epochs - best_epoch >= self.args.early_stop:
                if self.args.save_labels:
                    with open(os.path.join(self.args.res_save_dir, f'{self.args.modelName}-{self.args.datasetName}-labels.pkl'), 'wb') as df:
                        plk.dump(saved_labels, df, protocol=4)
                return

    def do_test(self, model, dataloader, mode="VAL"):
        model.eval()
        y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
        y_true = {'M': [], 'T': [], 'A': [], 'V': []}
        eval_loss = 0.0
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    if not self.args.need_data_aligned:
                        audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                        vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                    else:
                        audio_lengths, vision_lengths = 0, 0

                    labels_m = batch_data['labels']['M'].to(self.args.device).view(-1)
                    outputs = model(text, (audio, audio_lengths), (vision, vision_lengths))
                    loss = self.weighted_loss(outputs['M'], labels_m)
                    eval_loss += loss.item()
                    y_pred['M'].append(outputs['M'].cpu())
                    y_true['M'].append(labels_m.cpu())

        eval_loss = eval_loss / len(dataloader)
        logger.info(mode+"-(%s)" % self.args.modelName + " >> loss: %.4f " % eval_loss)
        pred, true = torch.cat(y_pred['M']), torch.cat(y_true['M'])
        eval_results = self.metrics(pred, true)
        logger.info('M: >> ' + dict_to_str(eval_results))
        eval_results['Loss'] = eval_loss

        return eval_results

    def weighted_loss(self, y_pred, y_true, indexes=None, mode='fusion'):
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        if mode == 'fusion':
            weighted = torch.ones_like(y_pred)
        else:
            weighted = torch.tanh(torch.abs(self.label_map[mode][indexes] - self.label_map['fusion'][indexes]))
        loss = torch.mean(weighted * torch.abs(y_pred - y_true))
        return loss

    def update_features(self, f_fusion, indexes):
        self.feature_map['fusion'][indexes] = f_fusion

    def update_centers(self):
        def update_single_center(mode):
            neg_indexes = self.label_map[mode] < 0
            if self.args.excludeZero:
                pos_indexes = self.label_map[mode] > 0
            else:
                pos_indexes = self.label_map[mode] >= 0
            self.center_map[mode]['pos'] = torch.mean(self.feature_map[mode][pos_indexes], dim=0)
            self.center_map[mode]['neg'] = torch.mean(self.feature_map[mode][neg_indexes], dim=0)

        update_single_center(mode='fusion')
    
    def init_labels(self, indexes, m_labels):
        self.label_map['fusion'][indexes] = m_labels
        self.label_map['text'][indexes] = m_labels
        self.label_map['audio'][indexes] = m_labels
        self.label_map['vision'][indexes] = m_labels

    def feat_diff(self):
        feat_with_label_p3 = []
        feat_with_label_n3 = []
        for p in self.feat_p3:
            feat_with_label_p3.append(self.feature_map['fusion'][p])
        for n in self.feat_n3:
            feat_with_label_n3.append(self.feature_map['fusion'][n])
        feat_with_label_p3 = torch.stack(feat_with_label_p3, 0)
        feat_with_label_n3 = torch.stack(feat_with_label_n3, 0)
        feat_with_label_p3_mean = torch.mean(feat_with_label_p3, 0)
        feat_with_label_n3_mean = torch.mean(feat_with_label_n3, 0)
        max_diff = torch.norm(feat_with_label_p3_mean - feat_with_label_n3_mean)
        return max_diff

    def save_index(self):
        for i, l in enumerate(self.label_map['fusion']):
            if l == 3.0:
                self.feat_p3.append(i)
            elif l == -3.0:
                self.feat_n3.append(i)
    
    def update_labels(self, f_fusion, f_text, f_audio, f_vision, cur_epoches, indexes, outputs):

        def update_single_label(f_single, mode):
            feat_diff_s = torch.norm(f_fusion-f_single, dim=-1)
            feat_diff_s = torch.clamp(feat_diff_s, max=max_diff)
            magnitude = 6*feat_diff_s/max_diff
            d_sp = torch.norm(f_single - self.center_map['fusion']['pos'], dim=-1)
            d_sn = torch.norm(f_single - self.center_map['fusion']['neg'], dim=-1)
            s_ratio = d_sp/d_sn
            for i, s_r in enumerate(s_ratio):
                if s_r > m_ratio[i]:      # should shift to negative direction
                    new_labels = self.label_map['fusion'][indexes[i]] - magnitude[i]
                    new_labels = torch.clamp(new_labels, min=-self.args.H, max=self.args.H)
                elif s_r < m_ratio[i]:      # should shift to positive direction
                    new_labels = self.label_map['fusion'][indexes[i]] + magnitude[i]
                    new_labels = torch.clamp(new_labels, min=-self.args.H, max=self.args.H)
                else:
                    new_labels = self.label_map['fusion'][indexes[i]]

                n = cur_epoches
                self.label_map[mode][indexes[i]] = (n - 1) / (n + 1) * self.label_map[mode][indexes[i]] + 2 / (n + 1) * new_labels

        max_diff = self.feat_diff()
        d_mp = torch.norm(f_fusion - self.center_map['fusion']['pos'], dim=-1)
        d_mn = torch.norm(f_fusion - self.center_map['fusion']['neg'], dim=-1)
        m_ratio = d_mp/d_mn

        update_single_label(f_text, mode='text')
        update_single_label(f_audio, mode='audio')
        update_single_label(f_vision, mode='vision')