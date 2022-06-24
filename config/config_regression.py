import os
import argparse

from utils.functions import Storage

class ConfigRegression():
    def __init__(self, args):
        # hyper parameters for models
        HYPER_MODEL_MAP = {
            'sugrm': self.__SUGRM
        }
        # hyper parameters for datasets
        HYPER_DATASET_MAP = self.__datasetCommonParams()

        # normalize
        model_name = str.lower(args.modelName)
        dataset_name = str.lower(args.datasetName)
        # load params
        commonArgs = HYPER_MODEL_MAP[model_name]()['commonParas']
        dataArgs = HYPER_DATASET_MAP[dataset_name]
        dataArgs = dataArgs['aligned'] if (commonArgs['need_data_aligned'] and 'aligned' in dataArgs) else dataArgs['unaligned']
        # integrate all parameters
        self.args = Storage(dict(vars(args),
                            **dataArgs,
                            **commonArgs,
                            **HYPER_MODEL_MAP[model_name]()['datasetParas'][dataset_name],
                            ))
    
    def __datasetCommonParams(self):
        root_dataset_dir = os.getcwd()
        tmp = {
            'mosi':{
                'aligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSI/aligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (768, 5, 20),
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss' 
                },
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSI/unaligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (768, 5, 20),
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss' 
                }
            },
            'mosei':{
                'aligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSEI/aligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (768, 74, 35),
                    'train_samples': 16326,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss'
                },
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSEI/unaligned_50.pkl'),
                    'seq_lens': (50, 500, 375),
                    # (text, audio, video)
                    'feature_dims': (768, 74, 35),
                    'train_samples': 16326,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss'
                }
            }
        }
        return tmp

    def __SUGRM(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': False,  # False,
                'need_model_aligned': False,
                'need_normalized': False,
                'use_bert': True,
                'use_finetune': True,
                'save_labels': False,
                'early_stop': 8,
                'update_epochs': 4
            },
            # dataset
            'datasetParas':{
                'mosi':{
                    # the batch_size of each epoch is update_epochs * batch_size
                    'batch_size': 32,
                    'learning_rate_bert': 5e-5,
                    'learning_rate_other': 1e-2,
                    'weight_decay_bert': 0.001,
                    'weight_decay_text': 0.001,
                    'weight_decay_other': 0.001,
                    'common_dim': 32,
                    # res
                    'H': 3.0,
                    # sparse phased transformer
                    'd_model': 32,
                    'orig_d_l': 768,
                    'orig_d_a': 5,
                    'orig_d_v': 20,
                    'num_heads': 8,
                    'layers': 4,
                    'attn_dropout': 0.3,
                    'relu_dropout': 0.3,
                    'res_dropout': 0.3,
                    'embed_dropout': 0.3,
                    'S': 5,
                    'r': [8,4,3],
                    'shift_mode': dict(I=['S,P,R'],X=['S'],S=['S'],C=[1,0.25,0.05]),
                    'use_fast': False,
                    'use_dense': False,
                    'combined_dim': 32,
                    'out_dropout': 0.3  # 0.1
                },
                'mosei':{
                    # the batch_size of each epoch is update_epochs * batch_size
                    'batch_size': 32,
                    'learning_rate_bert': 5e-5,
                    'learning_rate_other': 1e-3,
                    'weight_decay_bert': 0.001,
                    'weight_decay_text': 0.001,
                    'weight_decay_other': 0.01,
                    'common_dim': 32,
                    # res
                    'H': 3.0,
                    # sparse phased transformer
                    'd_model': 32,
                    'orig_d_l': 768,
                    'orig_d_a': 74,
                    'orig_d_v': 35,
                    'num_heads': 4,
                    'layers': 4,
                    'attn_dropout': 0.1,
                    'relu_dropout': 0.1,
                    'res_dropout': 0.1,
                    'embed_dropout': 0.2,
                    'S': 5,
                    'r': [8, 4, 3],
                    'shift_mode': dict(I=['S,P,R'], X=['S'], S=['S'], C=[1, 0.25, 0.05]),
                    'use_fast': False,
                    'use_dense': False,
                    'combined_dim': 32,
                    'out_dropout': 0.1

                }
            }
        }
        return tmp

    def get_config(self):
        return self.args