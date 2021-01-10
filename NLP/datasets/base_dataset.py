# TODO: fix nans in pdfs.

import re
import numpy as np
import pandas as pd
import torch
import json
import os

class BaseDataset:
    VAL = None
    TRAIN = None

    def __init__(self, article_wsize=None, embed_model=None, names=None):
        self.WORD = re.compile(r'\w+')
        self.common_names = names
        self.article_wsize = article_wsize
        self.ft = embed_model
        df = pd.read_csv("metadata.csv")
        self.df_author_article = df[df['authors'].notna()][df['pdf_json_files'].notna()]
        self.df_author_article = pd.concat([pd.Series(row['pdf_json_files'], row['authors'].split(';'))
                                            for _, row in self.df_author_article.iterrows()]).reset_index()
        self.df_author_article.columns = ['author', 'pdf']
        self.df_author_article.author = self.df_author_article.author.str.strip('. ')
        self.df_author_article['author_lastname'] = self.df_author_article.author.str.split(',').str[0].str.strip()

        self.df_author_article.dropna(inplace=True)

        self.filter_common_names()
        self.df_author_article['unique_id'] = pd.factorize(self.df_author_article['author_lastname'].tolist())[0]
        self.df_author_article = self.df_author_article.reset_index(drop=True)

        self.author_ids = self.df_author_article.unique_id.unique()

        np.random.shuffle(self.author_ids)
        ln = int(0.8 * len(self.author_ids))
        self.train, self.val = self.author_ids[:ln], self.author_ids[ln:]
        self.train_inds = self.df_author_article.loc[
            self.df_author_article['unique_id'].isin(self.train.tolist())].index
        self.val_inds = self.df_author_article.loc[self.df_author_article['unique_id'].isin(self.val.tolist())].index


    def filter_common_names(self):
        self.df_author_article = self.df_author_article[~self.df_author_article.author_lastname.isin(self.common_names)]

    def get_article(self, path):
        path = path.split()[0].strip(';')  # TODO: fix
        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except os.FileNotFoundError:

            print(path)
        article = []
        for chunk in data["body_text"]:  # TODO: process json.
            article.append(chunk["text"])
        return " ".join(article)

    def article_emb(self, article):
        padded_tensor = np.zeros((self.article_wsize, self.ft.get_dimension()))
        embeddings = np.vectorize(self.ft.get_word_vector, signature='()->(n)')(self.tokenize(article))
        padded_tensor[:min(embeddings.shape[0], self.article_wsize)] = embeddings[
                                                                       :min(embeddings.shape[0], self.article_wsize)]
        return padded_tensor[:self.article_wsize // 2] + padded_tensor[self.article_wsize // 2:]

    def tokenize(self, txt):

        words = self.WORD.findall(txt)
        splitted = np.array(words)
        return splitted


class ArticleDataset(torch.utils.data.Dataset):
    def __init__(self, prep_data_instance, train=True):
        self.preped = prep_data_instance
        self.train = train
        self.inds = self.preped.train_inds if self.train else self.preped.val_inds

    def __len__(self):
        return self.inds.shape[0]

    def __getitem__(self, idx):
        cidx = self.inds[idx]

        article_row = self.preped.df_author_article.iloc[cidx]
        author_idx = article_row.unique_id
        article_pdf = str(article_row.pdf)
        article = self.preped.get_article(article_pdf)
        article_embedding = self.preped.article_emb(article)
        return article_embedding.astype(float), author_idx

    def base_ind(self):
        return self.preped.df_author_article.unique_id[self.inds].tolist()
