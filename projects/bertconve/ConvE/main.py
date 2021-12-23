import json
import torch
import pickle
import numpy as np
import argparse
import sys
import os
import math

from os.path import join
import torch.backends.cudnn as cudnn

from evaluation import ranking_and_hits, ranking_and_hits_bertconve
from model import ConvE, DistMult, Complex, BertConvE, ConvE_channel

from spodernet.preprocessing.pipeline import Pipeline, DatasetStreamer
from spodernet.preprocessing.processors import JsonLoaderProcessors, Tokenizer, AddToVocab, SaveLengthsToState, StreamToHDF5, SaveMaxLengthsToState, CustomTokenizer
from spodernet.preprocessing.processors import ConvertTokenToIdx, ApplyFunction, ToLower, DictKey2ListMapper, ApplyFunction, StreamToBatch
from spodernet.utils.global_config import Config, Backends
from spodernet.utils.logger import Logger, LogLevel
from spodernet.preprocessing.batching import StreamBatcher
from spodernet.preprocessing.pipeline import Pipeline
from spodernet.preprocessing.processors import TargetIdx2MultiTarget
from spodernet.hooks import LossHook, ETAHook
from spodernet.utils.util import Timer
from spodernet.preprocessing.processors import TargetIdx2MultiTarget
import argparse


np.set_printoptions(precision=3)

cudnn.benchmark = True

import logging
def get_logger(module_name):
    # copied from https://realpython.com/python-logging/

    # Create a custom logger
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)

    # Create handlers
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)

    return logger
        

logger = get_logger(__name__)

''' Preprocess knowledge graph using spodernet. '''
def preprocess(dataset_name, delete_data=False):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path , flush=True)
    logger.info(f'preprocessing... current path = {dir_path}')
    full_path = 'data/{0}/e1rel_to_e2_full.json'.format(dataset_name)
    train_path = 'data/{0}/e1rel_to_e2_train.json'.format(dataset_name)
    dev_ranking_path = 'data/{0}/e1rel_to_e2_ranking_dev.json'.format(dataset_name)
    test_ranking_path = 'data/{0}/e1rel_to_e2_ranking_test.json'.format(dataset_name)

    full_path=os.path.join(dir_path, full_path)
    train_path=os.path.join(dir_path, train_path)
    dev_ranking_path=os.path.join(dir_path, dev_ranking_path)
    test_ranking_path=os.path.join(dir_path, test_ranking_path)

    keys2keys = {}
    keys2keys['e1'] = 'e1' # entities
    keys2keys['rel'] = 'rel' # relations
    keys2keys['rel_eval'] = 'rel' # relations
    keys2keys['e2'] = 'e1' # entities
    keys2keys['e2_multi1'] = 'e1' # entity
    keys2keys['e2_multi2'] = 'e1' # entity
    input_keys = ['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2']
    d = DatasetStreamer(input_keys)
    d.add_stream_processor(JsonLoaderProcessors())
    d.add_stream_processor(DictKey2ListMapper(input_keys))

    # process full vocabulary and save it to disk
    logger.info(f'processing full vocab from {full_path}')
    d.set_path(full_path)
    p = Pipeline(dataset_name, delete_data, keys=input_keys, skip_transformation=True)
    p.add_sent_processor(ToLower())
    p.add_sent_processor(CustomTokenizer(lambda x: x.split(' ')),keys=['e2_multi1', 'e2_multi2'])
    p.add_token_processor(AddToVocab())
    p.execute(d)
    p.save_vocabs()


    # process train, dev and test sets and save them to hdf5
    p.skip_transformation = False
    for path, name in zip([train_path, dev_ranking_path, test_ranking_path], ['train', 'dev_ranking', 'test_ranking']):
        logger.info(f'processing {name} from {path}')
        d.set_path(path)
        p.clear_processors()
        p.add_sent_processor(ToLower())
        p.add_sent_processor(CustomTokenizer(lambda x: x.split(' ')),keys=['e2_multi1', 'e2_multi2'])
        p.add_post_processor(ConvertTokenToIdx(keys2keys=keys2keys), keys=['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2'])
        p.add_post_processor(StreamToHDF5(name, samples_per_file=1000, keys=input_keys))
        p.execute(d)

def get_bert_embedding(args, rand_init_id = [0,1]):
    logger.info('retrieving bert embeddings')
    emb_dict = torch.load(args.fn_bert_embedding)
    # emb_bert = torch.nn.Embedding.from_pretrained(emb_dict['emb_mean']) # default will freeze
    emb_bert = emb_dict['emb_mean']
    if torch.cuda.is_available():
        emb_bert = emb_bert.cuda()

    if rand_init_id is not None:
        # randomly init embeddings of a few entities (0,1 are UNK and ?)
        emb_rand = torch.nn.Embedding(len(rand_init_id), emb_bert.shape[1], padding_idx=0)
        torch.nn.init.xavier_normal_(emb_rand.weight.data)

        emb_bert[rand_init_id, :] = emb_rand.weight.clone().detach().requires_grad_(False).to(emb_bert.device)
    return torch.nn.Embedding.from_pretrained(emb_bert) # default will freeze

def main(args, model_path):
    if args.preprocess: preprocess(args.data, delete_data=True)
    logger.info(f'loading preprocessed data')
    input_keys = ['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2']
    p = Pipeline(args.data, keys=input_keys)
    p.load_vocabs()
    vocab = p.state['vocab']

    num_entities = vocab['e1'].num_token
    logger.info(f'vocab loaded. {num_entities} entities.')
    # return

    train_batcher = StreamBatcher(args.data, 'train', args.batch_size, randomize=True, keys=input_keys, loader_threads=args.loader_threads)
    dev_rank_batcher = StreamBatcher(args.data, 'dev_ranking', args.test_batch_size, randomize=False, loader_threads=args.loader_threads, keys=input_keys)
    test_rank_batcher = StreamBatcher(args.data, 'test_ranking', args.test_batch_size, randomize=False, loader_threads=args.loader_threads, keys=input_keys)


    if args.model is None:
        model = ConvE(args, vocab['e1'].num_token, vocab['rel'].num_token)
    elif args.model == 'conve':
        model = ConvE(args, vocab['e1'].num_token, vocab['rel'].num_token)
    elif args.model == 'conve_channel':
        model = ConvE_channel(args, vocab['e1'].num_token, vocab['rel'].num_token)
    elif args.model == 'distmult':
        model = DistMult(args, vocab['e1'].num_token, vocab['rel'].num_token)
    elif args.model == 'complex':
        model = Complex(args, vocab['e1'].num_token, vocab['rel'].num_token)
    else:
        log.info('Unknown model: {0}', args.model)
        raise Exception("Unknown model!")

    train_batcher.at_batch_prepared_observers.insert(1,TargetIdx2MultiTarget(num_entities, 'e2_multi1', 'e2_multi1_binary'))


    eta = ETAHook('train', print_every_x_batches=args.log_interval)
    train_batcher.subscribe_to_events(eta)
    train_batcher.subscribe_to_start_of_epoch_event(eta)
    train_batcher.subscribe_to_events(LossHook('train', print_every_x_batches=args.log_interval))

    model.cuda()
    if args.resume:
        model_params = torch.load(model_path)
        print(model)
        total_param_size = []
        params = [(key, value.size(), value.numel()) for key, value in model_params.items()]
        for key, size, count in params:
            total_param_size.append(count)
            print(key, size, count)
        print(np.array(total_param_size).sum())
        model.load_state_dict(model_params)
        model.eval()
        ranking_and_hits(model, test_rank_batcher, vocab, 'test_evaluation')
        ranking_and_hits(model, dev_rank_batcher, vocab, 'dev_evaluation')
    else:
        model.init()

    total_param_size = []
    params = [value.numel() for value in model.parameters()]
    print(params)
    print(np.sum(params))

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    for epoch in range(args.epochs):
        model.train()
        for i, str2var in enumerate(train_batcher):
            opt.zero_grad()
            e1 = str2var['e1']
            rel = str2var['rel']
            e2_multi = str2var['e2_multi1_binary'].float()
            # label smoothing
            e2_multi = ((1.0-args.label_smoothing)*e2_multi) + (1.0/e2_multi.size(1))

            pred = model.forward(e1, rel)
            loss = model.loss(pred, e2_multi)
            loss.backward()
            opt.step()

            train_batcher.state.loss = loss.cpu()


        print('saving to {0}'.format(model_path))
        torch.save(model.state_dict(), model_path)

        model.eval()
        with torch.no_grad():
            if epoch % 5 == 0 and epoch > 0:
                print('ranking and hits on dev', flush=True)
                ranking_and_hits(model, dev_rank_batcher, vocab, 'dev_evaluation')
            if epoch % 5 == 0:
                if epoch > 0:
                    print('ranking and hits on test', flush=True)
                    ranking_and_hits(model, test_rank_batcher, vocab, 'test_evaluation')
    print('all done', flush = True)

def main_bertconve(args, model_path):
    assert args.model == 'bertconve', 'this function is only for training model bertconve'
    if args.preprocess: preprocess(args.data, delete_data=True)

    logger.info(f'loading preprocessed data')
    input_keys = ['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2']
    p = Pipeline(args.data, keys=input_keys)
    p.load_vocabs()
    vocab = p.state['vocab']

    num_entities = vocab['e1'].num_token
    logger.info(f'vocab loaded. {num_entities} entities.')

    train_batcher = StreamBatcher(args.data, 'train', args.batch_size, randomize=True, keys=input_keys, loader_threads=args.loader_threads)
    dev_rank_batcher = StreamBatcher(args.data, 'dev_ranking', args.test_batch_size, randomize=False, loader_threads=args.loader_threads, keys=input_keys)
    test_rank_batcher = StreamBatcher(args.data, 'test_ranking', args.test_batch_size, randomize=False, loader_threads=args.loader_threads, keys=input_keys)


    model = BertConvE(args, vocab['e1'].num_token, vocab['rel'].num_token)
    # if args.model is None:
    #     model = ConvE(args, vocab['e1'].num_token, vocab['rel'].num_token)
    # elif args.model == 'bertconve':
    #     model = BertConvE(args, vocab['e1'].num_token, vocab['rel'].num_token)
    # elif args.model == 'conve':
    #     model = ConvE(args, vocab['e1'].num_token, vocab['rel'].num_token)
    # elif args.model == 'distmult':
    #     model = DistMult(args, vocab['e1'].num_token, vocab['rel'].num_token)
    # elif args.model == 'complex':
    #     model = Complex(args, vocab['e1'].num_token, vocab['rel'].num_token)
    # else:
    #     log.info('Unknown model: {0}', args.model)
    #     raise Exception("Unknown model!")

    train_batcher.at_batch_prepared_observers.insert(1,TargetIdx2MultiTarget(num_entities, 'e2_multi1', 'e2_multi1_binary'))


    eta = ETAHook('train', print_every_x_batches=args.log_interval)
    train_batcher.subscribe_to_events(eta)
    train_batcher.subscribe_to_start_of_epoch_event(eta)
    train_batcher.subscribe_to_events(LossHook('train', print_every_x_batches=args.log_interval))

    # if args.model == 'bertconve':
    bert_embedding = get_bert_embedding(args, rand_init_id=[0,1])
    print(f'use bert_embedding = {bert_embedding}', flush=True)
    # else:
    #     bert_embedding = None
        
    model.cuda()
    if args.resume:
        print('resuming model')
        model_params = torch.load(model_path)
        print('model:', flush=True)
        # print(model, flush=True)
        total_param_size = []
        params = [(key, value.size(), value.numel()) for key, value in model_params.items()]
        for key, size, count in params:
            total_param_size.append(count)
            print('key, size, count', flush=True)
            print(key, size, count, flush=True)
        print('total number of parameters', flush=True)
        print(np.array(total_param_size).sum(), flush=True)
        model.load_state_dict(model_params)
        model.eval()
        ranking_and_hits_bertconve(model, dev_rank_batcher, vocab, 'dev_evaluation', embeddings=bert_embedding)
        ranking_and_hits_bertconve(model, test_rank_batcher, vocab, 'test_evaluation', embeddings=bert_embedding)
    else:
        logger.info(f'does not resume model. init again.')
        model.init()

    total_param_size = []
    params = [value.numel() for value in model.parameters()]
    print(f'params: {params}', flush=True)
    print(f'params total: {np.sum(params)}', flush=True)


    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    for epoch in range(args.epochs):
        model.train()
        for i, str2var in enumerate(train_batcher):
            opt.zero_grad()
            e1 = str2var['e1']
            rel = str2var['rel']
            e2_multi = str2var['e2_multi1_binary'].float()
            # label smoothing
            e2_multi = ((1.0-args.label_smoothing)*e2_multi) + (1.0/e2_multi.size(1))

            # if args.model == 'bertconve':
            pred = model.forward(bert_embedding, e1, rel)
            # else:
            #     pred = model.forward(e1, rel)
            loss = model.loss(pred, e2_multi)
            loss.backward()
            opt.step()

            train_batcher.state.loss = loss.cpu()


        print('saving to {0}'.format(model_path))
        torch.save(model.state_dict(), model_path)

        model.eval()
        with torch.no_grad():
            if epoch % 5 == 0: #and epoch > 0:
                ranking_and_hits_bertconve(model, dev_rank_batcher, vocab, 'dev_evaluation', embeddings=bert_embedding)
            if epoch % 5 == 0:
                # if epoch > 0:
                ranking_and_hits_bertconve(model, test_rank_batcher, vocab, 'test_evaluation', embeddings=bert_embedding)
    print('all done', flush = True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Link prediction for knowledge graphs')
    parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=128, help='input batch size for testing/validation (default: 128)')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate (default: 0.003)')
    parser.add_argument('--seed', type=int, default=17, metavar='S', help='random seed (default: 17)')
    parser.add_argument('--log-interval', type=int, default=100, help='how many batches to wait before logging training status')
    parser.add_argument('--data', type=str, default='FB15k-237', help='Dataset to use: {FB15k-237, YAGO3-10, WN18RR, umls, nations, kinship}, default: FB15k-237')
    parser.add_argument('--l2', type=float, default=0.0, help='Weight decay value to use in the optimizer. Default: 0.0')
    parser.add_argument('--model', type=str, default='conve', help='Choose from: {conve, distmult, complex}')
    parser.add_argument('--embedding-dim', type=int, default=200, help='The embedding dimension (1D). Default: 200')
    parser.add_argument('--embedding-shape1', type=int, default=20, help='The first dimension of the reshaped 2D embedding. The second dimension is infered. Default: 20')
    parser.add_argument('--conv-channel', type=int, default=32, help='Number of channels for conv filter. Default: 32')
    parser.add_argument('--kernel-size', type=int, default=3, help='Kernel size for conv filter. Default: 3')
    parser.add_argument('--hidden-drop', type=float, default=0.3, help='Dropout for the hidden layer. Default: 0.3.')
    parser.add_argument('--input-drop', type=float, default=0.2, help='Dropout for the input embeddings. Default: 0.2.')
    parser.add_argument('--feat-drop', type=float, default=0.2, help='Dropout for the convolutional features. Default: 0.2.')
    parser.add_argument('--lr-decay', type=float, default=0.995, help='Decay the learning rate by this factor every epoch. Default: 0.995')
    parser.add_argument('--loader-threads', type=int, default=4, help='How many loader threads to use for the batch loaders. Default: 4')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess the dataset. Needs to be executed only once. Default: 4')
    parser.add_argument('--resume', action='store_true', help='Resume a model.')
    parser.add_argument('--use-bias', action='store_true', help='Use a bias in the convolutional layer. Default: True')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing value to use. Default: 0.1')
    parser.add_argument('--hidden-size', type=int, default=9728, help='The side of the hidden layer. The required size changes with the size of the embeddings. Default: 9728 (embedding size 200).')
    parser.add_argument('--fn-bert-embedding', type=str, default='', help='filename to load bert embddings from (for BertConvE only)')
    parser.add_argument('--model-desc', type=str, default='', help='extra descriptive text for identifying model')

    args = parser.parse_args()
    print(f'resume model: {args.resume}')


    # parse console parameters and set global variables
    Config.backend = 'pytorch'
    Config.cuda = True
    Config.embedding_dim = args.embedding_dim
    #Logger.GLOBAL_LOG_LEVEL = LogLevel.DEBUG


    # model_name = '{2}_{0}_{1}'.format(args.input_drop, args.hidden_drop, args.model)
    # model_path = 'saved_models/{0}_{1}.model'.format(args.data, model_name)
    model_name = f'{args.data}_{args.model_desc}_{args.model}_c{args.conv_channel}_k{args.kernel_size}_lr{args.lr:.4f}'
    model_path = f'saved_models/{model_name}.model'

    torch.manual_seed(args.seed)
    if args.model == 'bertconve':
        main_bertconve(args, model_path)
    else:
        main(args, model_path)

    print('all done.', flush= True)