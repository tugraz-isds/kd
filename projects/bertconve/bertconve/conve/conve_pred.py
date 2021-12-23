import os

import torch 
import pandas as pd
import numpy as np

from model import BertConvE, ConvE_channel #ConvE, BertConvE#, DistMult, Complex, BertConvE
from main import get_bert_embedding
# from evaluation import ranking_and_hits
from spodernet.preprocessing.batching import StreamBatcher

from bertconve.preprocess.conve_preprocess_utils import ConvEPipeline
from bertconve.helper import log
from bertconve.helper.paths import paths_all

paths = paths_all()
logger = log.get_logger(__name__)

def check_hidden_size_lookup(d):
    for k, v in d.items():
        emb_size, emb_dim1, n_channel, k = k

        emb_dim2 = emb_size/emb_dim1
        hidden_size = (emb_dim1*2 - k + 1)*(emb_dim2 -k + 1)*n_channel
        assert hidden_size == v

def hidden_size_lookup(emb_size, emb_dim1, n_channel, k):
    d = {(200, 20, 32, 3): 9728,
         (768, 32, 4, 3): 5456,
         (768, 32, 8, 3): 10912,
         (768, 32, 32, 3): 43648,
         (768, 32, 256, 5): 307200}
    check_hidden_size_lookup(d)
    return d[(emb_size, emb_dim1, n_channel, k)]

def filter_pred(pred1, pred2, e1, e2, e2_multi1, e2_multi2):
    for i in range(e1.shape[0]):
        # these filters contain ALL labels
        filter1 = e2_multi1[i].long()
        filter2 = e2_multi2[i].long()


        # save the prediction that is relevant
        target_value1 = pred1[i,e2[i, 0].item()].item()
        target_value2 = pred2[i,e1[i, 0].item()].item()
        # zero all known cases (this are not interesting)
        # this corresponds to the filtered setting
        pred1[i][filter1] = 0.0
        pred2[i][filter2] = 0.0
        # write base the saved values
        pred1[i][e2[i]] = target_value1
        pred2[i][e1[i]] = target_value2

    return pred1, pred2

def pred_one_batch(model, str2var, vocab, name, embeddings = None, do_filter = True, emb_dict_or_fn = None):
    # for i, str2var in enumerate(dev_rank_batcher):
    e1 = str2var['e1']
    e2 = str2var['e2']
    rel = str2var['rel']
    rel_reverse = str2var['rel_eval']
    e2_multi1 = str2var['e2_multi1'].float()
    e2_multi2 = str2var['e2_multi2'].float()
    if embeddings is not None:
        pred1 = model.forward(embeddings, e1, rel)
        pred2 = model.forward(embeddings, e2, rel_reverse)
    else:
        pred1 = model.forward(e1, rel)
        pred2 = model.forward(e2, rel_reverse)
    pred1, pred2 = pred1.data, pred2.data
    e1, e2 = e1.data, e2.data
    e2_multi1, e2_multi2 = e2_multi1.data, e2_multi2.data
    if do_filter:
        pred1, pred2 = filter_pred(pred1, pred2, e1, e2, e2_multi1, e2_multi2)

    sim = calc_sim(emb_dict_or_fn, e1, e2)
    return pred1, pred2, e1, e2, rel, rel_reverse, sim

def calc_rank(pred1, pred2, e1, e2):
    # sort and rank
    max_values, argsort1 = torch.sort(pred1, 1, descending=True)
    max_values, argsort2 = torch.sort(pred2, 1, descending=True)

    ranks = []
    ranks_left = []
    ranks_right = []
    argsort1 = argsort1.cpu().numpy()
    argsort2 = argsort2.cpu().numpy()
    for i in range(e1.shape[0]):
    # for i in range(batch_size):
        # find the rank of the target entities
        rank1 = np.where(argsort1[i]==e2[i, 0].item())[0][0]
        rank2 = np.where(argsort2[i]==e1[i, 0].item())[0][0]
        # rank+1, since the lowest rank is rank 1 not rank 0
        ranks.append(rank1+1)
        ranks_left.append(rank1+1)
        ranks.append(rank2+1)
        ranks_right.append(rank2+1)
    return ranks_left, ranks_right

def calc_sim(emb_dict_or_fn, e1, e2):
    if emb_dict_or_fn is None:
        logger.info('does not compute cos similarity')
        return torch.zeros(e1.shape).view(-1)

    logger.info('computing cos similarity')
    if isinstance(emb_dict_or_fn, str):
        emb_dict = torch.load(emb_dict_or_fn)
        emb = emb_dict['emb_mean']
    elif isinstance(emb_dict_or_fn, dict):
        emb = emb_dict_or_fn['emb_mean']
    else:
        raise TypeError(f'unknown input type for input emb_dict_or_fn {type(emb_dict_or_fn)}. expecting str (filename) or dict.')

    cos1 = torch.nn.CosineSimilarity(dim=1)
    sim = cos1(emb[e1.view(-1)], emb[e2.view(-1)])
    # e1, e2 = 4512,804

    logger.info(f'cos sim computed. dims: emb={emb.shape}, e1={e1.shape}, e2={e2.shape}, sim={sim.shape}')
    
    return sim

def process_batch(model, str2var, vocab, name, embeddings=None, do_filter = True, emb_dict_or_fn = None):
    pred1, pred2, e1, e2, rel, rel_reverse, sim = pred_one_batch(model, str2var, vocab, name, 
                                                    embeddings = embeddings, do_filter = do_filter,
                                                    emb_dict_or_fn=emb_dict_or_fn)
    ranks2, ranks1 = calc_rank(pred1, pred2, e1, e2)
    return zip(e1.view(-1).tolist(), rel.view(-1).tolist(), 
                e2.view(-1).tolist(), rel_reverse.view(-1).tolist(),
                ranks2, ranks1, sim.view(-1).tolist())


class ConvEPredictor:
    def __init__(self, args):
        if args.rand_or_pretrained == 'rand':
            args.resume = False
            self.model = 'conve_channel'
            logger.warning(f'randomly init a new model. args.model set to conve_channel.')
        elif args.rand_or_pretrained == 'pretrained':
            args.resume = True
        else:
            raise ValueError(f'unknown input args.rand_or_pretrained {args.rand_or_pretrained}')

        if args.model == 'bertconve':
            args.fn_bert_embedding = os.path.join(paths.out, f'{args.lm_name}_emb_head.pt')
            args.embedding_dim = 768
            args.embedding_shape1 = 32
            self.modelnm = f'{args.data}_{args.lm_name}_{args.model}_c{args.conv_channel}_k{args.kernel_size}_lr{args.lr:.4f}'
        elif args.model == 'conve_channel':
            args.lm_name = 'rand'
            args.fn_bert_embedding = ''
            args.embedding_dim = 200
            args.embedding_shape1 = 20
            self.modelnm = f'{args.data}_{args.lm_name}_{args.model}_c{args.conv_channel}_k{args.kernel_size}_lr{args.lr:.4f}'
        
        args.hidden_drop = 0.3
        args.input_drop = 0.2
        args.feat_drop = 0.2
        args.use_bias = False
        args.hidden_size = hidden_size_lookup(args.embedding_dim, args.embedding_shape1, args.conv_channel, args.kernel_size)
        self.args = args
        
        logger.info(f'modelnm, fn_bert_embedding = {self.modelnm}, {args.fn_bert_embedding}')
        logger.info(f'all arguments: {args}')


    def get_model(self, vocab):
        logger.info(f'initializing model {self.args.model}')
        if self.args.model == 'conve_channel':
            model = ConvE_channel(self.args, vocab['e1'].num_token, vocab['rel'].num_token)
            bert_embedding = None
            emb_dict = None
        elif self.args.model == 'bertconve':
            model = BertConvE(self.args, vocab['e1'].num_token, vocab['rel'].num_token)
            bert_embedding = get_bert_embedding(self.args, rand_init_id=[0,1])
            emb_dict = torch.load(self.args.fn_bert_embedding)
            logger.info(f'embedding:  {bert_embedding}, device: {bert_embedding.weight.device}.')


        if torch.cuda.is_available():
            logger.info('move model to cuda')
            model.cuda()

        if self.args.resume:
            fn_model = os.path.join(paths.prj_home, 'nlp_repo', 'ConvE', 'saved_models', f'{self.modelnm}.model')
            logger.info(f'loading model from {fn_model}')
            model_params = torch.load(fn_model)
            model.load_state_dict(model_params)
        else:
            logger.info('randomly initialize model')
            model.init()
        return model, bert_embedding, emb_dict

    def do_pred(self, split):
        # split = 'test'        
        logger.info('loading vocab...')
        cep = ConvEPipeline(self.args.data)
        vocab = cep.load_vocab()

        model, bert_embedding, emb_dict = self.get_model(vocab)

        model.eval()
        input_keys = ['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2']
        logger.info(f'processing split {split}')
        if split == 'test':
            rank_batcher = StreamBatcher(self.args.data, f'{split}_ranking', self.args.test_batch_size, randomize=False, loader_threads=1, keys=input_keys)
        elif split == 'valid':
            rank_batcher = StreamBatcher(self.args.data, f'dev_ranking', self.args.test_batch_size, randomize=False, loader_threads=1, keys=input_keys)
        fn_ranks = os.path.join(paths.oi, f'ranks_per_node_{split}_best_{self.modelnm}_{self.args.rand_or_pretrained}.csv')

        for i, str2var in enumerate(rank_batcher):
            # e1, rel, e2, rel_reverse, ranks2, ranks1 = process_batch(model, str2var, vocab, 'test_evaluation', embeddings=None, do_filter = True)
            t = process_batch(model, str2var, vocab, f'{split}_evaluation', embeddings=bert_embedding, do_filter = True, emb_dict_or_fn=emb_dict)
            df = pd.DataFrame(t, columns = ['e1', 'rel', 'e2', 'rel_reverse', 'ranks2', 'ranks1', 'sim'])
            if i == 0:
                mode = 'w'
                header = True
            else:
                mode = 'a'
                header = False

            df.to_csv(fn_ranks, mode = mode, header=header, index=False)
            # if i % 5 == 0: #and epoch > 0:
            logger.info(f'| batch {i} done, {len(df)} rows saved')

        logger.info(f'ranks saved to file {fn_ranks}.')

        df = pd.read_csv(fn_ranks)
        # df.ranks1.to_numpy()
        logger.info(f'{len(df)} rows')

        logger.info(f'Mean reciprocal rank left: {np.mean(1./df.ranks2.to_numpy()):.6f}')
        logger.info(f'Mean reciprocal rank right: {np.mean(1./df.ranks1.to_numpy()):.6f}')


