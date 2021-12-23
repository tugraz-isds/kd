import os
from pathlib import Path
import json
import sys
import shutil

import torch
from torch.utils.data import DataLoader
import datasets
from transformers import pipeline

import bertconve.lm.feature_extractor as fe
from bertconve.preprocess.conve_preprocess_utils import ConvEPipeline
from bertconve.lm import utils# import mean_by_label
from bertconve.helper import log
from bertconve.helper.paths import paths_all

logger = log.get_logger(__name__)
paths = paths_all()

class KGFeatureExtractor:
    def __init__(self, datanm_conve, avg_tag, fe_strategy, modelnm = 'bert-base-cased', batch_size = 2500):
        # avg_tag: avg, sample
        # fe_strategy: from_triple, from_5nb
        # modelnm = 'bert-base-uncased'
        known_strategies = ['from_triple', 'from_5nb']

        assert avg_tag in ['avg', 'sample'], f'input unknown avg_tag = {avg_tag}'
        assert fe_strategy in known_strategies, f'input unknown fe_strategy = {fe_strategy}. (all known strategies: {known_strategies})'

        self.avg_tag = avg_tag
        self.fe_strategy = fe_strategy
        self.batch_size = batch_size    
        self.use_test_hr_or_unk = 'hr'
        self.datanm_conve = datanm_conve
        self.datanm = f'{datanm_conve}-neighbor'
        self._pathdata = os.path.join(paths.data, 'processed', self.datanm)

        self.splits = ['train', 'valid', 'test']
        
        if self.avg_tag == 'sample':
            self.random_state = 225322
        else:
            self.random_state = None

        if modelnm in ['bert-base-uncased', 'bert-base-cased', 'bert-large-cased']:
            path_model = modelnm
        else:# dir name where models are stored
            path_model = os.path.join(paths.models, modelnm)
        self.path_model = path_model
        self.modelnm  = modelnm

    @staticmethod
    def cat_dict_trpl_to_sent(dict_trpl, keys_to_cat = None):
        if keys_to_cat is None:
            keys_to_cat = dict_trpl.keys()
        dict_trpl['sent'] = ' '.join([dict_trpl[k] for k in keys_to_cat])
        return dict_trpl
    
    def load_prep_triple_data(self, n_triple_sparse = None):
        fn_dict = {'validation' if split == 'valid' else split:os.path.join(self._pathdata, f'{split}2txt.json') for split in self.splits}
        if n_triple_sparse is not None:
            split = 'train'
            fn_dict[split] = os.path.join(self._pathdata, f'{split}2txt_{n_triple_sparse}.json')
        dataset = datasets.load_dataset('json', data_files=fn_dict)
        logger.info(f'all split loaded dataset = {dataset}')

        datasplit_lst = []
        for split in fn_dict.keys():
            if split == 'train':
                keys_to_cat = ['h', 'r', 't']
            else:
                keys_to_cat = ['h', 'r']
            logger.info(f'processing split {split}')
            # datasplit_lst.append(dataset[split].map(lambda x: self.cat_dict_trpl_to_sent(x, keys_to_cat)).remove_columns(['r', 't']))
            datasplit_lst.append(dataset[split].map(lambda x: self.cat_dict_trpl_to_sent(x, keys_to_cat), 
                                                    remove_columns = ['r', 't']))
        dataset_sent = datasets.concatenate_datasets(datasplit_lst)
        logger.info(f'after concatenating all splits, shape: {dataset_sent.shape}')
        assert dataset_sent.shape[0] == sum([dataset[split].shape[0] for split in fn_dict.keys()]), 'sum of size of all splits not equal to the concatenated dataset size'

        return dataset_sent
    
    def load_prep_n_neighbor_data(self, max_nneighbor = 5, n_triple_sparse = None):
        fn_dict = {'validation' if split == 'valid' else split:os.path.join(self._pathdata, f'{split}2txt.json') for split in ['valid', 'test']}
        # fn_dict = {}
        split = 'train'
        if n_triple_sparse is not None:
            fn_train = os.path.join(self._pathdata, f'{split}_{max_nneighbor}neighbor_sent_{n_triple_sparse}.json')
        else:
            fn_train = os.path.join(self._pathdata, f'{split}_{max_nneighbor}neighbor_sent.json')

        dataset = datasets.load_dataset('json', data_files=fn_dict)
        logger.info(f'split valid and test loaded. dataset = {dataset}')
        # return dataset

        datasplit_lst = []
        for split in ['train']+list(fn_dict.keys()):
            print('-'*60, flush=True)
            logger.info(f'processing split {split}')
            if split == 'train':
                dataset_train = datasets.load_dataset('json', data_files=fn_train)['train']
                logger.info(f'split train loaded. dataset = {dataset_train}')
                datasplit_lst.append(dataset_train)
            else:
                keys_to_cat = ['h', 'r']
                # datasplit_lst.append(dataset[split].map(lambda x: self.cat_dict_trpl_to_sent(x, keys_to_cat)).remove_columns(['r', 't']))
                datasplit_lst.append(dataset[split].map(lambda x: self.cat_dict_trpl_to_sent(x, keys_to_cat), 
                                                        remove_columns = ['r', 't']))
        dataset_sent = datasets.concatenate_datasets(datasplit_lst)
        logger.info(f'after concatenating all splits, shape: {dataset_sent.shape}')
        assert dataset_sent.shape[0] == (dataset_train.shape[0]
                            +sum([dataset[split].shape[0] for split in fn_dict.keys()])), 'sum of size of all splits not equal to the concatenated dataset size'
        logger.info('data before and after concatenate size check OK')
        return dataset_sent
    
    @staticmethod
    def _join_ent_txt_textlong(dict_row):
        dict_row['sent'] = ' '.join([dict_row['ent_txt'], dict_row['textlong']])
        return dict_row

    

    def load_prep_data_for_feature_extraction(self):
        if self.fe_strategy == 'from_triple':
            dataset_sent = self.load_prep_triple_data(n_triple_sparse=None)
        elif self.fe_strategy == 'from_5nb':
            dataset_sent = self.load_prep_n_neighbor_data(max_nneighbor=5, n_triple_sparse=None)
        else:
            raise ValueError(f'unknown fe_strategy {self.fe_strategy}')

        logger.info(f'{self.fe_strategy} data prepared: {dataset_sent}')

        if self.avg_tag == 'sample':
            logger.info(f'avg_tag = {self.avg_tag}. shuffle data and then drop duplicates with pandas.')
            dataset_sent_u = self._drop_duplicates_from_dataset_sent(dataset_sent)
            return dataset_sent_u
        else:
            return dataset_sent

    def _drop_duplicates_from_dataset_sent(self, dataset):
        drop_duplicates_keep = 'first'
        df = dataset.to_pandas()

        logger.info(f'shuffling data with random_state {self.random_state}')
        df = df.sample(frac=1, random_state=self.random_state)
        logger.info(f'dropping duplicates using keep = {drop_duplicates_keep}')
        df = df.drop_duplicates('h', keep = drop_duplicates_keep)

        return datasets.Dataset.from_pandas(df)

    def _make_output_file_names(self, out_or_oi = 'oi'):
        if self.random_state is None:
            parnms_for_fn = f'{self.fe_strategy}_{self.avg_tag}_{self.use_test_hr_or_unk}'
        else:
            parnms_for_fn = f'{self.fe_strategy}_{self.avg_tag}{self.random_state}_{self.use_test_hr_or_unk}'

        datanm_for_fn = self.datanm_conve.replace('-', '').lower()
        output_fn_pattern = self.modelnm if datanm_for_fn in self.modelnm or self.datanm_conve in self.modelnm else f'{self.modelnm}_{datanm_for_fn}'

        path_emb_raw = os.path.join(paths.oi, f'{output_fn_pattern}_{parnms_for_fn}_emb_raw')
        if out_or_oi == 'out':
            fn_emb_head = os.path.join(paths.out, f'{output_fn_pattern}_{parnms_for_fn}_emb_head.pt')
            fn_emb_sum_all = os.path.join(path_emb_raw, 'emb_sum_all_batches.pt')
            return path_emb_raw, fn_emb_head, fn_emb_sum_all
        else: # 'oi
            fn_emb_raw_all = os.path.join(path_emb_raw, 'emb_raw_all_batches.pt')
            return path_emb_raw, fn_emb_raw_all



    def extract_emb(self, dataset):
        logger.info(f'----- loading model {self.modelnm} -----')
        if torch.cuda.is_available():
            nlp = pipeline('feature-extraction', model = self.path_model, device = torch.cuda.current_device())
        else:
            nlp = pipeline('feature-extraction', model = self.path_model)

        # logger.info('----- building dataset -----')
        dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=1)

        logger.info(f'{"="*20} extracting features {"="*20}')
        path_emb_raw, fn_emb_raw_all = self._make_output_file_names(out_or_oi = 'oi')

        if os.path.exists(path_emb_raw) and os.path.isdir(path_emb_raw):
            logger.warning(f'redo extracting embedding. removing all prev. extractions from {path_emb_raw}')
            shutil.rmtree(path_emb_raw)
        os.makedirs(path_emb_raw)

        ibatch = 0
        emb_raw = []
        h = []
        for batch_data in dl:
            fn_emb_raw = os.path.join(path_emb_raw, f'batch_size{self.batch_size}_ibatch{ibatch:05}.pt') 

            bfe = fe.BertFeatureExtractor(nlp)
            e = bfe.get_head_embeddings_from_triple(batch_data['sent'], batch_data['h'])

            emb_raw.append(e.cpu())
            h.extend(batch_data['h'])

            torch.save(dict(emb_raw=e, heads = batch_data['h']), fn_emb_raw)
            if ibatch % 10 == 0:
                logger.info(f'| features extracted for batch nr {ibatch}')
                logger.info(f'| raw embeddings saved to file {fn_emb_raw}')

            ibatch += 1

        try:
            emb = torch.cat(emb_raw, dim=0)
            logger.info(f'head embeddings extracted from each input sequence. total dim {emb.shape}. total head {len(h)}')

            logger.info(f'saving output embeddings to {fn_emb_raw_all}')
            if torch.cuda.is_available():
                torch.save(dict(heads = h,
                                emb_raw = emb.cuda()), fn_emb_raw_all)
            else:
                torch.save(dict(heads = h,
                                emb_raw = emb), fn_emb_raw_all)

        except:
            logger.warning(f'failed to concatenate and save all raw embeddings. {sys.exc_info()}')


        del emb_raw, nlp, emb
        torch.cuda.empty_cache()

    def get_conve_token2idx(self):
        logger.info('loading vocab with spodernet')
        p = ConvEPipeline(self.datanm_conve)
        vocab = p.load_vocab(do_preprocess=False)
        logger.info(f'vocab loaded from data path {self.datanm_conve}')
        return vocab['e1'].token2idx

    @staticmethod
    def map_bert_htxt_to_conve_hid(htxt, token2idx):
        return [token2idx.get(hi.replace(' ', '_').strip().lower()) for hi in htxt]

    def aggregate_emb(self):
        # aggregating raw embeddings extracted via extract_emb, such that 
        # there's only one embedding per entity, in the same order as ConvE preprocessor
        path_emb_raw, fn_emb_head, fn_emb_sum_all = self._make_output_file_names(out_or_oi = 'out')
        print('-'*60, flush = True)
        token2idx = self.get_conve_token2idx()
        n_class = max(token2idx.values())+1

        fns = sorted(Path(path_emb_raw).glob('batch_size*.pt'), key=os.path.getmtime)

        print('\n' +'-'*60, flush = True)
        logger.info('aggregating embeddings per batch')
        head_ids_all = []

        for ibatch, fn_emb_raw_batch in enumerate(fns):
            dict_emb = torch.load(fn_emb_raw_batch)

            h_ids = self.map_bert_htxt_to_conve_hid(dict_emb['heads'], token2idx)
            assert all([hid is not None for hid in h_ids]), 'some hids are not found'

            if ibatch % 20 == 0 or ibatch == len(fns)-1:
                logger.info(f'| batch {ibatch}................')
                sample_size = min(10, len(h_ids))
                logger.info(f'| h_ids examples {h_ids[:sample_size]}')
                logger.info(f'| token2id keys examples {list(token2idx.keys())[:sample_size]}')
                logger.info(f'| dict_emb examples {dict_emb["heads"][:sample_size]}')

            head_ids_all.extend(h_ids)
            emb_sum_batch, w = utils.sum_by_label_scatter_add(dict_emb['emb_raw'].to('cpu'), h_ids, n_class)
            if ibatch == 0:
                weights = w
                emb_sum = emb_sum_batch
            else:
                weights = weights + w
                emb_sum = emb_sum + emb_sum_batch

            if ibatch % 20 == 0 or ibatch == len(fns)-1:
                logger.info(f'|{"-"*20} processing batch {ibatch} {"-"*20}')
                logger.info(f'|raw embeddings loaded from {fn_emb_raw_batch}')
                logger.info(f'|raw embeddings loaded. keys: {dict_emb.keys()}\n\
                    shape heads: {len(dict_emb["heads"])}\n\
                    shape emb: {dict_emb["emb_raw"].shape}')
                logger.info(f'|total {len(h_ids)} ({len(set(h_ids))} unique) heads found in data (out of total {n_class} heads).')

                logger.info(f'|batch mean embedding has shape {emb_sum.shape}')

                logger.info(f'|sanity check: unique embeddings collected: {emb_sum.unique(dim=0).shape} \n\
                    unique head_ids saved: {len(set(head_ids_all))}.')
        
        # the sum is saved still to output_intermediate as it's intermediate step
        logger.info(f'emb aggration per batch done. saving output embeddings to {fn_emb_sum_all}')
        torch.save(dict(emb_sum = emb_sum, 
                        head_ids = head_ids_all,
                        counts = weights), fn_emb_sum_all)

        try:
            logger.info('calculating overall average')
            emb_mean = emb_sum/weights.view(-1,1)
            emb_mean[torch.isnan(emb_mean)] = 0
            # (m1*w1.view(-1, 1) + m2*w2.view(-1, 1))/(w1+w2).view(-1,1)
            logger.info(f'mean embedding shape {emb_mean.shape}, weights shape {weights.shape}')

            logger.info(f'saving output embeddings to {fn_emb_head}')
            torch.save(dict(emb_mean = emb_mean, 
                            head_ids = head_ids_all,
                            counts = weights), fn_emb_head)

            del emb_mean, weights
            torch.cuda.empty_cache()

            emb = torch.load(fn_emb_head)
            logger.info(f'saved mean embedding shape {emb["emb_mean"].shape}, weights shape {emb["counts"].shape}')
            logger.info(f'sanity check overall average: unique embeddings collected: \n\
                {emb["emb_mean"].unique(dim=0).shape} \n\
                unique head_ids saved: {len(set(emb["head_ids"]))}.')

        except:
            logger.warning(f'failed to calculate mean embeddings and save. {sys.exc_info()}')


    def agg_fix(self):
        # path to read
        path_emb_raw, fn_emb_raw_all = self._make_output_file_names(out_or_oi = 'oi')
        # path to write
        fn_emb_mean = os.path.split(path_emb_raw)[-1].replace('_emb_raw', '_emb_mean')
        logger.info(f'agg_fix file name: {fn_emb_mean}.pt')
        fn_emb_mean_full = os.path.join(paths.out, f'{fn_emb_mean}.pt')
        
        print('-'*60, flush = True)
        token2idx = self.get_conve_token2idx()
        n_class = max(token2idx.values())+1
        print('\n' +'-'*60, flush = True)

        raw = torch.load(fn_emb_raw_all)

        h_ids = self.map_bert_htxt_to_conve_hid(raw['heads'], token2idx)
        emb_sum, w = utils.sum_by_label_scatter_add(raw['emb_raw'].to('cpu'), h_ids, n_class)
        nemb = emb_sum.unique(dim=0).shape[0]
        logger.info(f'after sum by label, found nemb = {nemb}')
        assert nemb > (n_class-2)

        emb_mean = emb_sum/w.view(-1,1)
        emb_mean[torch.isnan(emb_mean)] = 0
        nemb = emb_mean.unique(dim=0).shape[0]
        logger.info(f'after calc mean, found nemb = {nemb}')
        assert nemb > (n_class-2)

        logger.info(f'emb_mean saved to {fn_emb_mean_full}')
        torch.save(dict(emb_mean = emb_mean, 
                        head_ids = h_ids,
                        counts = w), fn_emb_mean_full)


        emb = torch.load(fn_emb_mean_full)
        logger.info(f'saved mean embedding shape {emb["emb_mean"].shape}, weights shape {emb["counts"].shape}')
        logger.info(f'sanity check overall average: unique embeddings collected: \n\
            {emb["emb_mean"].unique(dim=0).shape} \n\
            unique head_ids saved: {len(set(emb["head_ids"]))}.')
        assert len(set(emb["head_ids"]))+1 == emb["emb_mean"].unique(dim=0).shape[0], 'expect nunique embeddings to be equal to unique-head_ids +1 (unk " " with all zero embeddings)'

