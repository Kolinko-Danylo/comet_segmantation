import fasttext
import pandas as pd
import string


def get_embedder(type='fasttext', emb_size=150):
    if type == 'fasttext':
        emb_model = fasttext.load_model('cc.en.300.bin')
        fasttext.util.reduce_model(emb_model, emb_size)
    return emb_model

def get_common_names(dblp_path):
    '''

    :param dblp_path: DBLP dataset path
    :return: DBLP's homonymic last names
    '''
    names_df = pd.read_csv(dblp_path, header=None)
    names = list(set(names_df[names_df[0].str.endswith(tuple(string.digits))][0].str.split().str[-2]))
    return names