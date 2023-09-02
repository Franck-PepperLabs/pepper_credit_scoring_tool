import os
import pandas as pd
from pepper.utils import bold, green, red
from pepper.env import get_project_dir, get_dataset_dir


def print_path_info(path):
    project_dir = get_project_dir()
    print(
        path.replace(project_dir, '[project_dir]'),
        'exists' if os.path.exists(path) else 'doesn\'t exist'
    )


def create_subdir(project_path, rel_path='', verbose=True):
    path = os.path.join(project_path, rel_path)
    if verbose:
        print_path_info(path)
    if not os.path.exists(path):
        os.makedirs(path)
        project_dir = get_project_dir()
        print(bold('✔ ' + path.replace(project_dir, '[project_dir]')), 'created.')
    return path


status = lambda s, o: bold(green('✔ ' + o) if s else red('✘ ' + o))
def commented_return(s, o, a, *args): # ='✔'
    print(status(s, o), a)
    return args


# Multi-indexing utils
def _load_struct(no_comment=True):
    dataset_dir = get_dataset_dir()
    data = pd.read_json(os.path.join(dataset_dir, 'struct.json'), typ='frame', orient='index')
    # print(bold('✔ struct'), 'loaded')
    return data if no_comment else commented_return(True, 'struct', 'loaded', data)
    #return data

_struct = _load_struct()

# get element by id and label
_get_element = lambda id, label: _struct.loc[_struct.id == id, label].values[0]
group = lambda id: _get_element(id, 'group')
subgroup = lambda id: _get_element(id, 'subgroup')
domain = lambda id: _get_element(id, 'domain')
format = lambda id: _get_element(id, 'format')
unity = lambda id: _get_element(id, 'unity')
astype = lambda id: _get_element(id, 'astype')
nan_code = lambda id: _get_element(id, 'nan_code')
nap_code = lambda id: _get_element(id, 'nap_code')

# get columns labels from ancestor
_get_labels = lambda k, v: _struct.name[_struct[k] == v].values
get_group_labels = lambda gp_label: _get_labels('group', gp_label)


def new_multi_index(levels=None):
    if levels is None:
        levels = ['group']
    return pd.MultiIndex.from_frame(_struct[levels + ['name']], names=levels+['var'])
