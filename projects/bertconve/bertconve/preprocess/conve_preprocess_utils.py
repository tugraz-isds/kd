import os
from spodernet.preprocessing.pipeline import Pipeline#, DatasetStreamer

from bertconve.helper import log

logger = log.get_logger(__name__)


class ConvEPipeline:
    def __init__(self, data_name):
        self.data_name = data_name
        self.sep = '\t'

    def load_vocab(self, do_preprocess = False):
        logger.info(f'loading vocab for dataset {self.data_name}')
        input_keys = ['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2']
        p = Pipeline(self.data_name, keys=input_keys)
        loaded = p.load_vocabs()
        if not loaded and do_preprocess:
            logger.info(f'cannot load vocab, trying to redo preprocessing')
            self.preprocess()
            loaded = p.load_vocabs()
        
        if not loaded:
            raise Exception('load vocab failed')
        
        vocab = p.state['vocab']

        return vocab

    def preprocess(self):
        from main import preprocess
        preprocess(self.data_name, delete_data=True)

    def save_vocab(self, vocab, path_to_save):
        token2idx = vocab['e1'].token2idx
        idx2token = vocab['e1'].idx2token

        fn_token2idx = os.path.join(path_to_save, 'token2idx.txt')
        fn_idx2token = os.path.join(path_to_save, 'idx2token.txt')

        logger.info('writing token2idx..')
        write_dict_as_txt(token2idx, fn_token2idx)
        logger.info('writing idx2token..')
        write_dict_as_txt(idx2token, fn_idx2token)

    def read_vocab_txt_as_dict(self, path_to_data):
        fn_token2idx = os.path.join(path_to_data, 'token2idx.txt')
        fn_idx2token = os.path.join(path_to_data, 'idx2token.txt')

        logger.info(f'loading vocab file {fn_token2idx}')
        with open(fn_token2idx, 'r') as f:
            txt = f.read()
        self.token2idx = {k:int(v) for ln in txt.split('\n') for k,v in ln.split(self.sep)}

        logger.info(f'loading vocab file {fn_idx2token}')
        with open(fn_idx2token, 'r') as f:
            txt = f.read()
        self.idx2token = {int(k):v for ln in txt.split('\n') for k,v in ln.split(self.sep)}
        # return token2idx, idx2token

def write_dict_as_txt(dict_to_write, fn_to_write, sep = '\t'):
    logger.info(f'writing dict to file {fn_to_write}')
    txt = '\n'.join([f'{k}{sep}{v}' for k,v in dict_to_write.items()])
    with open(fn_to_write, 'w') as f:
        f.write(txt)
    logger.info(f'dict written to file {fn_to_write}')


