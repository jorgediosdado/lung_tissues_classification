import os
import glob
import numpy as np
import pandas as pd
from sklearn.utils import class_weight
from tensorflow.keras.utils import plot_model as tf_plot
from sklearn.model_selection import GroupShuffleSplit

def get_files(base_dir, resolution, exclude_pd=False):
    ext = ['jpg', 'jpeg']
    ret_files = []

    for f in glob.glob(f'{base_dir}/**', recursive=True):
        if not any([f.endswith(e) for e in ext]):
            continue
        if  (resolution is not None) and f'_{resolution}' not in f:
            continue
        if (exclude_pd) and ('_pd' in f):
            continue
        ret_files.append(f)
        
    return ret_files

def get_dataframe(dataset_csv, resolution=None, exclude_pd=False):
    da = pd.read_csv(dataset_csv)
    da = da[da.hc != 0]
    da['image_path'] = ''
    for idx, row in da.iterrows():
        css = row['superclass']
        css = css if pd.isnull(row['subclass']) else css+'_'+row['subclass']
        pth = css + '_' + row['resolution'] + '_' + row['image_id']
        da.loc[idx, 'image_path'] = f'data/dataset_w_HC/{css}/' + pth+'.jpg'
    
    if resolution is not None:
        da = da[da.resolution==resolution]
    if exclude_pd:
        da = da[da.subclass != 'pd']
    return da

def get_classes_labels(root_directory, image_paths, class_type, exclude_pd=False):
    if class_type == 'micro':
        class_names = sorted([f for f in os.listdir(root_directory) if not f.startswith('.')])
    else:
        class_names = sorted(list(set([f if '_' not in f else f.split('_')[0] for f in os.listdir(root_directory) if not f.startswith('.')])))
        
    class_names = class_names if not exclude_pd else [c for c in class_names if '_pd' not in c]

    class2int = dict(zip(class_names, range(len(class_names))))
    labels = list(map(lambda im: class2int[im.split(root_directory)[1].split('/')[0]] if class_type=='micro' else class2int[im.split(root_directory)[1].split('/')[0].split('_')[0]], image_paths))
    
    return class_names, labels


def train_test_split(df, test_size=0.5, random_state=7):
    groups = df.groupby('targetclass')
    all_train = []
    all_test = []
    for group_id, group in groups:
        group = group[~group['hc'].isin(all_train+all_test)]
        if group.shape[0] == 0:
            continue
        train_inds, test_inds = next(GroupShuffleSplit(
            test_size=test_size, n_splits=2, random_state=random_state).split(group, groups=group['hc']))

        all_train += group.iloc[train_inds]['hc'].tolist()
        all_test += group.iloc[test_inds]['hc'].tolist()

    df_train= df[df['hc'].isin(all_train)]
    df_test= df[df['hc'].isin(all_test)]
    
    return df_train, df_test


def compute_weights(my_generator):
    labels = np.concatenate([l.argmax(1) for _, l in my_generator])
    class_weights = class_weight.compute_class_weight('balanced',
                                                         classes=sorted(np.unique(labels)),
                                                         y=list(labels))
    class_weights = dict(enumerate(class_weights))
    return class_weights


def plot_model(model):
    return tf_plot(model, rankdir='LR', show_shapes=True)
    
    
def dataset_description(dataset_csv='data/dataset_HC.csv'):
    df = get_dataframe(dataset_csv=dataset_csv)
    table = []
    for sp, scs in [('aca', ('bd', 'md', 'pd')), 
                    ('nor', None),
                    ('scc', ('bd', 'md', 'pd')),
                   ]:
        if scs is not None:
            for sc in scs:
                tmp = []
                dfc = df.copy()
                for res in ['20x', '40x']:
                    cnt = len(dfc[(dfc.superclass==sp) & (dfc.subclass==sc) & (dfc.resolution==res)])
                    tmp.append(cnt)
                table.append(tmp)
        else:
            tmp = []
            dfc = df.copy()
            for res in ['20x', '40x']:
                cnt = len(dfc[(dfc.superclass==sp) & (dfc.resolution==res)])
                tmp.append(cnt)
            table.append(tmp)
    summary = pd.DataFrame(table, 
             columns=['20x', '40x'],
            index=['aca_bd', 'aca_md', 'aca_pd', 'nor', 'scc_bd', 'scc_md', 'scc_pd']
            )
    summary.loc['Total'] = summary.sum()
    summary.loc[:, 'Total'] = summary.sum(1)
    
    print('Total images', summary.iloc[-1,-1])
    print('Total patients', len(df.hc.unique()))
    
    return summary