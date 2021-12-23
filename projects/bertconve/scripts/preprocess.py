import sys
from bertconve.preprocess import kg_preprocessors as p

from bertconve.helper import log
logger = log.get_logger(__name__)

# datanm = 'atomic'
datanm = sys.argv[1]

def clean_ent(ent):
    return ent.replace('_', ' ').strip().lower()

def clean_rel(rel):
    return rel

dict_n_rel_per_data = {'atomic': 9,
                       'conceptnet100k': 34,
                       'FB15k-237': 237}

print('#'*60, flush=True)
logger.info(f'making {datanm}-id files...')
apid = p.KGIdPreprocessor(datanm=datanm, n_rel_expect=dict_n_rel_per_data[datanm], n_triple_sparse=None)
apid.triple_preprocess()

print('#'*60, flush=True)
logger.info(f'making {datanm}-neighbor files...')
apnb = p.KGNeighborPreprocessor(datanm_conve=datanm)
apnb.update_df_ent2id_rel2id(clean_ent, clean_rel)
apnb._make_all_known_entities()
apnb.make_1neighbor()
apnb.make_n_neighbor_files()


logger.info('all done')



