import os, json, random
from typing import Union, List

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from bertconve.helper.various import file_head
from bertconve.helper import log
from bertconve.helper.paths import paths_all

logger = log.get_logger(__name__)
paths = paths_all()


class TriplePreprocessor:
    def __init__(self):
        self.sep = '\t'
        self.unk_txt = '<unk>'
        self.unk_id = 0
        self.head_nm = 'h'
        self.tail_nm = 't'
        self.rel_nm = 'r'
        self.triple_names = (self.head_nm, self.rel_nm, self.tail_nm)
        self.rev_suffix = '_reverse'
        self.rt_nm = 'rt' # colnm for [rel_id, tail_id]
        self.sent_nm = 'sent'

    def load_data(self, fn, colnm, remove_duplicates = True):
        logger.info(f'loading data from {fn}, delimiter {self.sep}, columns: {colnm}')
        df = pd.read_csv(fn, 
                         sep = self.sep, 
                         header = None, 
                         names = colnm)#, 
                        #  nrows = 100)
        logger.info(f'{len(df)} rows loaded from {fn}')
        if remove_duplicates:
            df = df.drop_duplicates()
            logger.info(f'{len(df)} unique rows')

        logger.info(f'example row {df.sample(1).to_dict(orient = "records")}')
        return df

    def write_triples(self, trpl_lst_or_df, fn):
        logger.info(f'writing triples to file {fn}')
        if isinstance(trpl_lst_or_df, pd.DataFrame):
            df = trpl_lst_or_df
        else: # isinstance(trpl_lst_or_df, list):
            df = pd.DataFrame(trpl_lst_or_df)
        df.to_csv(fn, sep = self.sep, index = False, header = False, quoting=3) # quoting = csv.QUOTE_NONE

        logger.info(f'{len(df)} triples of ids written to file {fn}')
        file_head(fn)
    
    def write_pd_series_to_text(self, pd_series, fn):
        logger.info(f'input data size {len(pd_series)}')
        
        with open(fn, 'w') as f:
            f.write('\n'.join(pd_series.to_list()))
        file_head(fn)
    
    def df_or_lst_write_json(self, df_or_lst, fn_json):
        logger.info(f'saving input {type(df_or_lst)} to {fn_json}')
        if isinstance(df_or_lst, list):
            logger.info(f'input list of dict, keys {df_or_lst[0].keys()}')
            lst = df_or_lst
        else: #isinstance(df_or_lst, pd.DataFrame)
            logger.info(f'input pandas dataframe, columns {df_or_lst.columns.values}')
            logger.info(f'first row of input df {df_or_lst.iloc[0]}')
            lst = df_or_lst.to_dict(orient='records')

        logger.info(f'first element of records {lst[0]}')
        with open(fn_json, 'w') as f:
            f.writelines([json.dumps(j)+'\n' for j in lst])
        file_head(fn_json)

    def write_sents(self, dftxt, fn_txt, fn_json):
        if fn_txt is not None:
            logger.info(f'saving sentences to {fn_txt}')
            self.write_pd_series_to_text(dftxt.loc[:,self.sent_nm], fn_txt)
        if fn_json is not None:
            logger.info(f'saving head/sentences pairs to {fn_json}')
            self.df_or_lst_write_json(dftxt.loc[:, [self.head_nm, self.sent_nm]], fn_json)

    def load_vocab_dict(self, fn, dtype = int):
        logger.info(f'loading vocab dictionary from {fn}')
        with open(fn, 'r') as f:
            t = f.read()
        lst = t.strip().split('\n')
        d = {row.split(self.sep)[0]: dtype(row.split(self.sep)[1]) for row in lst}

        logger.info(f'vocab dictionary loaded from {fn}. size {len(lst)}')
        return d

    def write_cat_id(self, dict_cat_id, fn):
        logger.info(f'writing categories and ids to file {fn}')

        cat_ids = '\n'.join([f'{k}{self.sep}{v}' for k,v in dict_cat_id.items()])
        with open(fn, 'w') as f:
            f.write(cat_ids)

        logger.info(f'{len(dict_cat_id.keys())} category - id pairs written to file {fn}')

    def label_encode_unk(self, categories: List[str], id_start = 1):

        le = LabelEncoder()
        le.fit(categories)

        d_cat = dict(zip(le.classes_, le.transform(le.classes_)+id_start))

        assert self.unk_id not in d_cat.values(), f'{self.unk_id} is already in encode results'

        d_cat.update({self.unk_txt:self.unk_id})
        return d_cat

    def generate_ent_dict(self, 
                          df_triple_or_categories: Union[pd.DataFrame, List[str]], 
                          id_start = 1):
        logger.info('encode nodes...')

        if isinstance(df_triple_or_categories, list):
            categories = df_triple_or_categories
        else:
            categories = df_triple_or_categories.loc[:, [self.head_nm, self.tail_nm]].stack().to_list()
        
        dict_ent = self.label_encode_unk(categories, id_start=id_start)
        logger.info(f'{len(dict_ent.keys())} categories in nodes')

        return dict_ent

    def generate_rel_dict(self, 
                          df_triple_or_categories: Union[pd.DataFrame, List[str]],
                          id_start = 1):
        logger.info('encode edges...')

        if isinstance(df_triple_or_categories, list):
            categories = df_triple_or_categories
        else:
            categories = df_triple_or_categories[self.rel_nm].to_list()

        dict_rel = self.label_encode_unk(categories, id_start=id_start)
        logger.info(f'{len(dict_rel.keys())} categories in edges')

        return dict_rel

    def update_dict_ent_rel(self, dict_ent: dict, dict_rel:dict):
        assert isinstance(dict_ent, dict), 'input dict_ent must be a dictionary'
        assert isinstance(dict_rel, dict), 'input dict_rel must be a dictionary'
        self.test_unk(dict_ent, dict_rel)

        self.dict_ent2id = dict_ent
        self.dict_rel2id = dict_rel

    def test_unk(self, dict_ent2id, dict_rel2id):
        logger.info('testing unknown token consistency')
        assert self.unk_txt in dict_ent2id.keys(), f'{self.unk_txt} not in ent_dict'
        assert self.unk_txt in dict_rel2id.keys(), f'{self.unk_txt} not in rel_dict'
        assert dict_ent2id.get(self.unk_txt) == self.unk_id, f'dict_ent2id.get({self.unk_txt}) != {self.unk_id}'
        assert dict_rel2id.get(self.unk_txt) == self.unk_id, f'dict_rel2id.get({self.unk_txt}) != {self.unk_id}'

    def triple_encode(self, df_trpl, dict_ent2id, dict_rel2id):
        logger.info('encode triples to ids...')
        logger.info(f'input df_trpl has {len(df_trpl)} rows ({len(df_trpl.drop_duplicates())} unique rows)')

        # use map iso replace because it's faster
        # see https://stackoverflow.com/questions/41985566/pandas-replace-dictionary-slowness
        self.test_unk(dict_ent2id, dict_rel2id)

        df_id = pd.DataFrame({
                self.head_nm: df_trpl[self.head_nm].map(lambda x: dict_ent2id.get(x, dict_ent2id[self.unk_txt])), 
                self.tail_nm: df_trpl[self.tail_nm].map(lambda x: dict_ent2id.get(x, dict_ent2id[self.unk_txt])), 
                self.rel_nm: df_trpl[self.rel_nm].map(lambda x: dict_rel2id.get(x, dict_rel2id[self.unk_txt])), 
                })

        logger.info(f'triples converted to ids. {len(df_id)} triples after conversion')
        return df_id.loc[:, df_trpl.columns]

    def triple_decode(self, df_trpl_id, dict_id2ent, dict_id2rel):
        logger.info('decode triples back to text...')
        logger.info(f'input df_trpl_id has {len(df_trpl_id)} triples ({len(df_trpl_id.drop_duplicates())} unique rows).')

        # use map iso replace because it's faster
        # see https://stackoverflow.com/questions/41985566/pandas-replace-dictionary-slowness
        df_txt = pd.DataFrame({
                self.head_nm: df_trpl_id[self.head_nm].map(dict_id2ent.get), 
                self.tail_nm: df_trpl_id[self.tail_nm].map(dict_id2ent.get), 
                self.rel_nm: df_trpl_id[self.rel_nm].map(dict_id2rel.get)
                })

        logger.info(f'ids of triples converted back to texts. {len(df_txt)} rows after conversion')
        self.inspect_decode_results(df_trpl_id, df_txt)
        return df_txt

    def triple_decode_pandas(self, df_trpl_id, dfent2id_or_fn, dfrel2id_or_fn = None, drop_duplicates = True):
        logger.info('decode triples back to text...')

        if isinstance(dfent2id_or_fn, str):
            dfent2id = self.load_data(dfent2id_or_fn, colnm = ['ent_txt', 'ent_id'])
        else:
            dfent2id = dfent2id_or_fn

        if isinstance(dfrel2id_or_fn, str):
            dfrel2id = self.load_data(dfrel2id_or_fn, colnm = ['rel_txt', 'rel_id'])
        else:
            dfrel2id = dfrel2id_or_fn


        n_unique_rows = len(df_trpl_id.drop_duplicates())
        logger.info(f'input df_trpl_id has {len(df_trpl_id)} triples ({n_unique_rows} unique rows).')
        if drop_duplicates and (len(df_trpl_id) != n_unique_rows):
            df_trpl_id = df_trpl_id.drop_duplicates()

        df_trpl_htxt = df_trpl_id.merge(dfent2id, left_on = 'h', right_on = 'ent_id', how = 'left').rename(columns={'ent_txt':'h_txt'}).drop('ent_id', axis = 1)
        logger.info(f'after merging with head-texts, data shape {df_trpl_htxt.shape}')
        # print(df_trpl_htxt.shape)
        df_trpl_httxt = df_trpl_htxt.merge(dfent2id, left_on = 't', right_on = 'ent_id', how = 'left').rename(columns={'ent_txt':'t_txt'}).drop('ent_id', axis = 1)
        logger.info(f'after merging with tail-texts, data shape {df_trpl_httxt.shape}')

        if dfrel2id is not None:
            df_trpl_txt = df_trpl_httxt.merge(dfrel2id, left_on = 'r', right_on = 'rel_id', how = 'left').drop('rel_id', axis = 1)
            logger.info(f'after merging with rel-texts, data shape {df_trpl_txt.shape}')
        else:
            df_trpl_txt = df_trpl_httxt.rename(columns={'r': 'rel_txt'})

        assert (len(df_trpl_id) == len(df_trpl_htxt)) & (len(df_trpl_id) == len(df_trpl_httxt)) & (len(df_trpl_id) == len(df_trpl_txt)), 'changed length after merging'

        df_trpl_txt = df_trpl_txt.loc[:, ['h_txt', 'rel_txt','t_txt']]
        df_trpl_txt.columns = list(self.triple_names)
        if drop_duplicates:
            df_trpl_txt_unique = df_trpl_txt.drop_duplicates()
            if len(df_trpl_txt) != len(df_trpl_txt_unique):
                logger.warning( f'after converting to texts, there are non unique rows -- shape after drop duplicates {df_trpl_txt_unique.shape} \
                \nduplicated head(10) = {df_trpl_txt.loc[df_trpl_txt.duplicated(keep = False)].head(10)}')
            assert len(df_trpl_txt) == len(df_trpl_txt_unique), f'after converting to texts, there are non unique rows.'
            df_trpl_txt = df_trpl_txt_unique
        
        logger.info(f'ids of triples converted back to texts. {len(df_trpl_txt)} rows after conversion')
        self.inspect_decode_results(df_trpl_id, df_trpl_txt)
        return df_trpl_txt
    
    @staticmethod
    def inspect_decode_results(df_id, df_txt):
        seed = random.randint(1, 1e4)
        logger.info(f'example before decode {df_id.sample(1, random_state = seed).to_dict(orient = "records")}')
        logger.info(f'example after decode {df_txt.sample(1, random_state = seed).to_dict(orient = "records")}')


    @staticmethod
    def _add_rev_suffix(text, rev_suffix, rel_type_symmetric = []):
        if text.lower() in [r.lower() for r in rel_type_symmetric]:
            return text
        else: 
            return f'{text}{rev_suffix}'

    def add_reverse_edge(self, df_trpl, rev_suffix = None, rel_type_symmetric = []):
        rev_suffix = self.rev_suffix if rev_suffix is None else rev_suffix

        logger.info(f'adding reverse edges with suffix {rev_suffix}')
        logger.info(f'input df_trpl has shape {df_trpl.shape} (unique rows {df_trpl.drop_duplicates().shape})')

        df_trpl_rev = pd.DataFrame({
            self.rel_nm: df_trpl[self.rel_nm].apply(lambda x: self._add_rev_suffix(x, rev_suffix, rel_type_symmetric)),
            self.head_nm: df_trpl[self.tail_nm],
            self.tail_nm: df_trpl[self.head_nm]
        })

        df =  pd.concat([df_trpl, df_trpl_rev], axis = 0, sort = True).reset_index(drop = True)
        logger.info(f'after adding reverse edges, df_trpl has shape {df.shape}, with unique rows {df.drop_duplicates().shape}')
        return df

    def add_reverse_edge_list(self, rel_list, rev_suffix = None, rel_type_symmetric = []):
        rev_suffix = self.rev_suffix if rev_suffix is None else rev_suffix

        rel_list.extend([self._add_rev_suffix(r, rev_suffix, rel_type_symmetric) for r in rel_list])
        return rel_list

    def agg_all_tails_per_head(self, df_trpl_id, fn_neighbor_id, rt_nm = 'rt'):
        # for making files {split}_neighbor2id.txt

        logger.info(f'aggregate all tails per head')
        # df_trpl_id.loc[:, 'rt'] = df_trpl_id.apply(lambda x: f'{x[self.rel_nm]}{self.sep_walk}{x[self.tail_nm]}', axis = 1)
        df_trpl_id.loc[:, rt_nm] = df_trpl_id.apply(lambda x: [x[self.rel_nm], x[self.tail_nm]], axis = 1)
        pds_nb = df_trpl_id.groupby(self.head_nm)[rt_nm].apply(list)#.to_frame()#.reset_index()
        # fn_csv = os.path.join(path_data, 'valid_neighbor2id.txt')
        logger.info(f'tails for {len(pds_nb)} heads found. descriptive stats on nr neighbors:')
        print(pds_nb.apply(len).describe(), flush = True)
        if fn_neighbor_id is not None:
            logger.info(f'saving aggregated results to {fn_neighbor_id}')
            pds_nb.to_csv(fn_neighbor_id)

        return pds_nb.to_frame()

