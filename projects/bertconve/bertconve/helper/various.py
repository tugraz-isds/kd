import os, platform
from collections import Counter
from itertools import chain

from . import log
from .paths import paths_all

logger = log.get_logger(__name__)
paths = paths_all()

def file_head(fn):
    with open(fn, 'r') as f:
        logger.info(f'first line of file {fn}: {f.readline()}')

