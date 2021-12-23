import os

from collections import namedtuple

def paths_all(prj_home = '/project'):
    prj_nm = 'bertconve'

    prj = os.path.join(prj_home, prj_nm)
    path_dict = dict(prj_home = prj_home,
        prj = prj,
        data = os.path.join(prj, 'data'),
        oi = os.path.join(prj, 'output_intermediate'),
        out = os.path.join(prj, 'output'),
        resources = os.path.join(prj, 'resources'),
        models = os.path.join(prj, 'models'),
        logs = os.path.join(prj, 'logs')
        )

    return namedtuple('Paths', path_dict.keys())(**path_dict)

