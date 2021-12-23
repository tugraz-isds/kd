import os
import sys


from bertconve.lm import kg_extract_lm_feature as ke

from bertconve.helper import log
from bertconve.helper.paths import paths_all

logger = log.get_logger(__name__)
paths = paths_all()

modelnm = sys.argv[1]
avg_tag = sys.argv[2]
fe_strategy = sys.argv[3]
datanm_conve = sys.argv[4]

do_extract_emb = True

if 'long' in fe_strategy:
    batch_size = 400
elif fe_strategy == 'from_5nb':
    batch_size = 1000
else:
    batch_size = 2500

logger.info(f'feature extraction with arguments: \
    \n modelnm = {modelnm}, \
    \n avg_tag = {avg_tag}, \
    \n fe_strategy = {fe_strategy}, \
    \n datanm = {datanm_conve}, \
    \n do_extract_emb = {do_extract_emb}, \
    \n batch_size = {batch_size}')


fe = ke.KGFeatureExtractor(datanm_conve, avg_tag, fe_strategy, modelnm, batch_size = batch_size)

logger.info(f'feature extractor initialized, with attributes {fe.__dict__}')

if do_extract_emb:
    print('#'*60, flush=True)
    logger.info(f'loading data')
    dataset_sent = fe.load_prep_data_for_feature_extraction()
    logger.info(f'dataset in total has shape {dataset_sent.shape}, expect {dataset_sent.shape[0]/batch_size} batches')
    
    logger.info(f'extract embeddings')
    fe.extract_emb(dataset_sent)
    logger.info(f'extraction done')

print('#'*60, flush=True)
logger.info(f'agg embeddings')
fe.aggregate_emb()

logger.info('all done')


