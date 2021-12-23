#!/project/.venv/bert_env/bin/python

import os
import json
import csv
from collections import namedtuple
import random
from ast import literal_eval

import pandas as pd
import numpy as np
from datasets import load_dataset

from bertconve.preprocess.triple_preprocessor import TriplePreprocessor
from bertconve.helper.various import file_head
from bertconve.helper import log
from bertconve.helper.paths import paths_all


paths = paths_all()
logger = log.get_logger(__name__)


class KGIdPreprocessor(TriplePreprocessor):
    def __init__(self, datanm, n_rel_expect, n_triple_sparse = None, **kwargs):
        super().__init__(**kwargs)
        self.n_triple_sparse = n_triple_sparse
        self.datanm = datanm
        self.column_nms = list(self.triple_names)
        self.n_rel_expect = n_rel_expect#*(self.do_add_reverse_edge+1)
        self.splits = ['train', 'valid', 'test']
        self.get_paths_fns()

    def get_paths_fns(self):
        PathMapping = namedtuple('PathMapping', ['conve', 'bertconve'])

        path_rel_repo = 'nlp_repo'

        self.datanm_mapping = PathMapping(conve=self.datanm, bertconve = f'{self.datanm}-id')
        self.path_conve = os.path.join(paths.prj_home, path_rel_repo, 'ConvE', 'data', self.datanm_mapping.conve)
        self.path_id = os.path.join(paths.data, 'processed', self.datanm_mapping.bertconve)

        logger.info(f'creating output path {self.path_id} if not yet existing')
        os.makedirs(self.path_id, exist_ok=True)

        if self.n_triple_sparse is not None:
            self.datanm_conve_sparse = f'{self.datanm}-textsparse{self.n_triple_sparse}'
            self.path_conve_sparse = os.path.join(paths.prj_home, path_rel_repo, 'ConvE', 'data', self.datanm_conve_sparse)
            logger.info(f'creating output path {self.path_conve_sparse} if not yet existing')
            os.makedirs(self.path_conve_sparse, exist_ok=True)


    def make_dict_ent_rel(self, df_trpl_all, do_add_reverse_edge = True):
        # make files:
        # - entity2id.txt # all entities
        # - relation2id.txt # all relations
        fn_entid = os.path.join(self.path_id, 'entity2id.txt')
        fn_relid = os.path.join(self.path_id, 'relation2id.txt')
        
        # encode entities and relations
        logger.info('generating node encoding dictionaries')
        id_start_ent = 1 # start from 1 since 0 is reserved for unk
        dict_ent = self.generate_ent_dict(df_trpl_all, id_start_ent)

        logger.info('generating edge encoding dictionaries')
        id_start_rel = max(dict_ent.values())+1
        edges = df_trpl_all.loc[:, self.rel_nm].to_list()
        if do_add_reverse_edge:
            edges = self.add_reverse_edge_list(edges, rev_suffix = self.rev_suffix, rel_type_symmetric=[])
        dict_rel = self.generate_rel_dict(edges, id_start_rel)


        logger.info('update attributes dict_ent2id and dict_rel2id')
        self.update_dict_ent_rel(dict_ent=dict_ent, dict_rel=dict_rel)

        logger.info('check if number of relation is consistent')
        assert len(self.dict_rel2id.keys()) == self.n_rel_expect*(do_add_reverse_edge+1) + 1
        logger.info(f'number of relation as expected. OK')

        logger.info('check if entities contail space')
        n_word_per_ent = [len(ent.split()) == 1 for ent in dict_ent.keys()]
        assert all(n_word_per_ent), 'some entities contain space'
        logger.info(f'entity space check OK (none has space)')

        logger.info('saving dictionaries')
        self.write_cat_id(self.dict_ent2id, fn_entid)
        self.write_cat_id(self.dict_rel2id, fn_relid)

    def make_relid_reverse_mapping(self):
        fn_relid = os.path.join(self.path_id, 'relation2id.txt')
        fn_relid_mapping = os.path.join(self.path_id, 'relation2id_rev_mapping.txt')

        logger.info(f'making mapping between relation-id and reverse-relation-id')

        dfrelid = self.load_data(fn_relid, ['rel', 'rel_id'])
        dfrelid = dfrelid.loc[dfrelid.rel_id != self.unk_id]
        dfrelid.loc[:,'rel_rev'] = dfrelid.rel.apply(lambda x: f'{x}{self.rev_suffix}')

        df = dfrelid.drop('rel_rev', axis = 1).merge(dfrelid.drop('rel', axis = 1), left_on = 'rel', right_on = 'rel_rev', how = 'left')
        df = df.loc[~df.rel_id_y.isna()].astype({'rel_id_y':int})
        df.loc[:, ['rel_id_y', 'rel_id_x']].to_csv(fn_relid_mapping, sep = self.sep, index = False, header = False)
        logger.info(f'relation id mapping to reverse saved to {fn_relid_mapping}')
        file_head(fn_relid_mapping)


    def load_split(self, split, func_clean_triple = None):
        fn_trpl = os.path.join(self.path_conve, f'{split}.txt')
        df = self.load_data(fn_trpl, self.column_nms)
        if func_clean_triple is None:
            return df
        else:
            return func_clean_triple(df)

    def triple_txt2id(self, df_trpl, do_add_reverse_edge):
        logger.info(f'input df_trpl has shape {df_trpl.shape} (unique rows {df_trpl.drop_duplicates().shape})')
        if do_add_reverse_edge:
            df_trpl = self.add_reverse_edge(df_trpl, rev_suffix = self.rev_suffix, rel_type_symmetric=[])
            logger.info(f'after adding reverse edges, df_trpl has shape {df_trpl.shape}, with unique rows {df_trpl.drop_duplicates().shape}')

        df_id = self.triple_encode(df_trpl, self.dict_ent2id, self.dict_rel2id)
        logger.info(f'after encoding, df_id has shape {df_id.shape}')
        assert len(df_id) == len(df_trpl), 'after decoding the shape is not the same'
        n_unk = (df_id.stack()==self.unk_id).sum()
        logger.info(f'{n_unk} occurrences of unknown entity or relation')
        assert n_unk == 0, 'expect no unk entities or relations'
        return df_id.loc[:, self.column_nms]

    def triple_preprocess(self, func_clean_triple = None, dict_rev_split = None):
        # make files:
        # test2id.txt
        # train2id.txt
        # valid2id.txt
        # triples_all2id.txt

        # which split to add reverse edge
        if dict_rev_split is None:
            dict_rev_split = {split:True for split in self.splits}
            dict_rev_split.update({'triples_all': True})


        # make triple_all
        logger.info('make triples all')
        df_trpl_all = pd.DataFrame()
        d_df = dict()
        for split in self.splits:
            d_df[split] = self.load_split(split, func_clean_triple)
            df_trpl_all = pd.concat([df_trpl_all, d_df[split]])
        logger.info(f'triples all have shape {df_trpl_all.shape}')
        # d_df['all'] = df_trpl_all

        print('='*60, flush=True)
        self.make_dict_ent_rel(df_trpl_all, do_add_reverse_edge=True)
        print('='*60, flush=True)
        self.get_unknown_entites(d_df['train'])

        print('='*60, flush=True)
        # for split in ['train', 'valid', 'test']:
        for split, do_add_reverse_edge in dict_rev_split.items():
            print('-'*60, flush=True)
            fn_id = os.path.join(self.path_id, f'{split}2id.txt')
            fn_neighbor_id = os.path.join(self.path_id, f'{split}_neighbor2id.txt')

            logger.info(f'encode {split} triples')
            if split == 'triples_all':
                df_id = self.triple_txt2id(df_trpl_all, do_add_reverse_edge)
            else:
                df_id = self.triple_txt2id(d_df[split], do_add_reverse_edge)

            logger.info(f'saving ids of {split} triples')
            self.write_triples(df_id.drop_duplicates(), fn_id)


    def get_unknown_entites(self, dftrain):
        # make files:
        # - entity_unknown2id.txt
        fn_entid_unknown = os.path.join(self.path_id, 'entity_unknown2id.txt')
        logger.info(f'making entity_unknown2id...')

        ent_train = dftrain.loc[:, [self.head_nm, self.tail_nm]].stack().unique().tolist()
        logger.info(f'training data contains {len(ent_train)} unique entities')
        logger.info(f'there are {len(self.dict_ent2id)} unique entities in total')
        ent_unknown = set(list(self.dict_ent2id.keys()))-set(ent_train)
        d_ent_id_unknown = {k:self.dict_ent2id[k] for k in ent_unknown}

        logger.info(f'{len(ent_unknown)} unknown entities found. to dict {len(d_ent_id_unknown)} entities.')
        self.write_cat_id(d_ent_id_unknown, fn_entid_unknown)

    def make_conve_sparse(self):
        fn_entid = os.path.join(self.path_id, 'entity2id.txt')
        fn_relid = os.path.join(self.path_id, 'relation2id.txt')

        fn_train_txt_conve = os.path.join(self.path_conve_sparse, 'train.txt')

        fn_train_id_no_rev = os.path.join(self.path_id, f'train2id_{self.n_triple_sparse}_no_rev.txt')
        df_id = self.load_data(fn_train_id_no_rev, self.column_nms, remove_duplicates=True)

        dftxt = self.triple_decode_pandas(df_id, fn_entid, fn_relid, drop_duplicates = True)
        self.write_triples(dftxt.loc[:, self.column_nms], fn_train_txt_conve)

        return dftxt 




class KGNeighborPreprocessor(TriplePreprocessor):
    # make the following files:
    # train_1neighbor_sent.txt
    # valid_1neighbor_sent.txt
    # test_1neighbor_sent.txt

    # entity_unknown.json
    # train_1neighbor_sent.json
    # # test_1neighbor_sent.json
    # # valid_1neighbor_sent.json

    def __init__(self, datanm_conve, add_period = False, **kwargs):
        super().__init__(**kwargs)
        self.datanm = f'{datanm_conve}-neighbor'
        self.column_nms = list(self.triple_names)
        self.splits = ['train', 'valid', 'test']
        self.path_id = os.path.join(paths.data, 'processed', f'{datanm_conve}-id')
        self.path_nb = os.path.join(paths.data, 'processed', self.datanm)
        self.add_period = add_period
        self.max_nneighbor = 5 

        logger.info(f'creating output path {self.path_nb} if not yet existing')
        os.makedirs(self.path_nb, exist_ok=True)

    def update_df_ent2id_rel2id(self, clean_ent, clean_rel):
        fn_entid = os.path.join(self.path_id, 'entity2id.txt')
        fn_relid = os.path.join(self.path_id, 'relation2id.txt')

        self.dfent2id = self.load_clean_txt_id(fn_entid, 'ent', 'txt2id', clean_ent, 'pandas')
        self.dfrel2id = self.load_clean_txt_id(fn_relid, 'rel', 'txt2id', clean_rel, 'pandas')

        assert set(self.dfent2id.ent_id.unique()) & set(self.dfrel2id.rel_id.unique())== set([self.unk_id]),  f'\
            expecting the only overlapping id for rel and ent to be the id for {self.unk_txt} ({self.unk_id})'

    def _make_all_known_entities(self):
        fn_entity_known = os.path.join(self.path_nb, 'entities_all.txt')
        logger.info(f'saving all known entities to file {fn_entity_known}')
        logger.info(f'input known entites shape {self.dfent2id.shape}')
        self.write_pd_series_to_text(self.dfent2id.loc[self.dfent2id.ent_txt!=self.unk_txt].ent_txt, fn_entity_known)


    def load_clean_txt_id(self, fn, colnm_pattern, colnm_id_txt_seq = 'txt2id', func_clean = None, return_pandas_or_dict = 'pandas'):
        colnm_id, colnm_txt = f'{colnm_pattern}_id', f'{colnm_pattern}_txt'
        if colnm_id_txt_seq == 'txt2id':
            colnms = [colnm_txt, colnm_id]
        elif colnm_id_txt_seq == 'id2txt':
            colnms = [colnm_id, colnm_txt]

        df = self.load_data(fn, colnms)
        if func_clean is not None:
            df.loc[:, colnm_txt] = df.loc[:, colnm_txt].apply(func_clean)
            logger.info(f'text column "{colnm_txt}" cleaned')

        if return_pandas_or_dict == 'dict':
            return df.set_index(colnm_id).to_dict(orient = 'dict')[colnm_txt]
        else:
            return df

    def _make_json_and_sent_txt_from_triple_id_split(self, split, n_triple_sparse = None):
        # n_triple_sparse: number of triples in sparse data. None if original data (not sparsified)
        if n_triple_sparse is None:
            fn_split = os.path.join(self.path_id, f'{split}2id.txt')
            fn_json_trpl = os.path.join(self.path_nb, f'{split}2txt.json')
            fn_txt = os.path.join(self.path_nb, f'{split}_1neighbor_sent.txt')
        else:
            fn_split = os.path.join(self.path_id, f'{split}2id_{n_triple_sparse}.txt')
            fn_json_trpl = os.path.join(self.path_nb, f'{split}2txt_{n_triple_sparse}.json')
            fn_txt = os.path.join(self.path_nb, f'{split}_1neighbor_sent_{n_triple_sparse}.txt')

        logger.info(f'making 1neighbor file {fn_split} for {split}')
        df_id = self.load_data(fn_split, self.column_nms, remove_duplicates=True)
        dftxt = self.triple_decode_pandas(df_id, self.dfent2id, self.dfrel2id, drop_duplicates = True)
        self.df_or_lst_write_json(dftxt, fn_json_trpl)

        dftxt.loc[:, self.sent_nm] = dftxt.apply(lambda x: ' '.join(x.to_list())+'.'*self.add_period, axis = 1)
        logger.info(f'example sent and triple: {dftxt.sample(1).to_dict(orient = "records")}')

        self.write_pd_series_to_text(dftxt.loc[:,self.sent_nm], fn_txt)

    def make_1neighbor(self):
        for split in self.splits:
            print('='*60, flush=True)
            self._make_json_and_sent_txt_from_triple_id_split(split, n_triple_sparse= None)

    
    def load_neighbors(self, fn_neighbor):
        logger.info(f'loading neighbors file {fn_neighbor}')
        dfnb = pd.read_csv(fn_neighbor)
        dfnb.loc[:,self.rt_nm]=dfnb.loc[:, self.rt_nm].apply(literal_eval)

        logger.info(f'neighbor file loaded. shape {dfnb.shape}, \ndtypes {dfnb.dtypes.to_dict()} \
            \nfirst row {dfnb.iloc[0].to_dict()}, \ndatatype of column rt: {type(dfnb.iloc[0].rt)}')
        return dfnb

    def _sample_list(self, list_of_neighbors):
        n_nb = len(list_of_neighbors)
        if n_nb <= self.max_nneighbor:
            return [list_of_neighbors]

        lst_of_n_nb = []
        for _ in range(n_nb):
            lst_of_n_nb.append(random.sample(list_of_neighbors, self.max_nneighbor))
        return lst_of_n_nb

    def sample_neighbors(self, dfnb):
        logger.info(f'sampling neighbors with max nb {self.max_nneighbor} ')
        dfnb.loc[:, self.sent_nm] = dfnb.loc[:,self.rt_nm].apply(self._sample_list)
        dfid = dfnb.drop(self.rt_nm, axis = 1).explode(self.sent_nm)

        logger.info(f'checking size consistency before and after exploding')
        nnb = dfnb[self.rt_nm].apply(len)
        logger.info(f'total number of neighbors {nnb.sum()}')
        nnb.loc[nnb<=self.max_nneighbor] = 1
        logger.info(f'expected number of rows after exploading {nnb.sum()}. actual size after exploading {dfid.shape}')
        assert nnb.sum() == len(dfid), 'after exploding the size doesnt match'
        logger.info('assert ok')
        return dfid

    @staticmethod
    def _neighbor_id_list_to_sent(id_list, dict_id2txt):
        return ' '.join([dict_id2txt.get(id) for rt in id_list for id in rt])

    def decode_neighbors(self, dfnb_id):
        logger.info('making dict_id2txt')
        txt_ids = np.concatenate((self.dfent2id.to_numpy(), self.dfrel2id.to_numpy()), axis = 0)
        dict_id2txt = {txt_id[1]:txt_id[0] for txt_id in txt_ids}
        assert len(dict_id2txt) == len(txt_ids)-1, 'after converting to dict, size doesnt match after concatenate'

        logger.info('converting id to text')
        dftxt = dfnb_id.copy()
        dftxt.loc[:, self.head_nm] = dfnb_id.loc[:, self.head_nm].apply(dict_id2txt.get)
        dftxt.loc[:, self.sent_nm] = dfnb_id.loc[:, self.sent_nm].apply(lambda x: self._neighbor_id_list_to_sent(x, dict_id2txt))
        dftxt.loc[:, self.sent_nm] = dftxt.apply(lambda x: ' '.join([x.loc[self.head_nm], x.loc[self.sent_nm]]), axis = 1)
        logger.info(f'shape after converting to texts {dftxt.shape}')

        dftxt = dftxt.drop_duplicates()
        logger.info(f'shape after dropping duplicates {dftxt.shape}')

        seed = random.randint(1, 1e4)
        logger.info(f'example before decode {dfnb_id.sample(1, random_state = seed).to_dict(orient = "records")}')
        logger.info(f'example after decode {dftxt.sample(1, random_state = seed).to_dict(orient = "records")}')
        return dftxt


    def _make_json_and_sent_txt_from_n_neighbor_id_split(self, split, n_triple_sparse = None):
        # n_triple_sparse: number of triples in sparse data. None if original data (not sparsified)
        if n_triple_sparse is not None and split == 'train':
            fn_split = os.path.join(self.path_id, f'{split}_neighbor2id_{n_triple_sparse}.txt')
            fn_json_sent = os.path.join(self.path_nb, f'{split}_{self.max_nneighbor}neighbor_sent_{n_triple_sparse}.json')
            fn_txt = os.path.join(self.path_nb, f'{split}_{self.max_nneighbor}neighbor_sent_{n_triple_sparse}.txt')
        
        else:# n_triple_sparse is None:
            fn_split = os.path.join(self.path_id, f'{split}_neighbor2id.txt')
            fn_json_sent = os.path.join(self.path_nb, f'{split}_{self.max_nneighbor}neighbor_sent.json')
            fn_txt = os.path.join(self.path_nb, f'{split}_{self.max_nneighbor}neighbor_sent.txt')

        logger.info(f'making n-neighbor file from {fn_split} for {split}')
        print('-'*60, flush = True)
        dfnb = self.load_neighbors(fn_split)

        print('-'*60, flush = True)
        dfid = self.sample_neighbors(dfnb)

        print('-'*60, flush = True)
        logger.info('decoding ids to texts')
        dftxt = self.decode_neighbors(dfid)

        print('-'*60, flush = True)
        logger.info('saving results to files..')
        # write _sent.json file only for training data (since these are used for feature extraction only)
        fn_json_sent = fn_json_sent if split == 'train' else None
        if fn_txt is not None or fn_json_sent is not None:
            self.write_sents(dftxt, fn_txt, fn_json_sent)
        return dftxt
    
    def make_n_neighbor_files(self, n_triple_sparse = None):
        if n_triple_sparse is not None:
            logger.info(f'sparse data {n_triple_sparse}. make nb file only for training split')
            print('='*60, flush=True)
            self._make_json_and_sent_txt_from_n_neighbor_id_split('train', n_triple_sparse= n_triple_sparse)

        else:
            logger.info('original data. make nb file for all splits')
            for split in self.splits:
                print('='*60, flush=True)
                self._make_json_and_sent_txt_from_n_neighbor_id_split(split, n_triple_sparse= n_triple_sparse)

