from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pandas as pd
import torch
import datetime


class USData(Dataset):
    def __init__(self, year=2013, type='train', market='NASDAQ', root_path='/home/kikyo/data/qt/', tickers=None,
                 seq_len=30):
        # init
        self.seq_len = seq_len
        self.root_path = root_path
        self.market = market
        self.type = type
        self.year = year
        self.tickers = tickers
        self.__read_data__()

    def __read_data__(self):
        for index, ticker in enumerate(self.tickers):
            stock_path = os.path.join(self.root_path, self.market + '_' + ticker + '_1.csv')
            stocks_pd = pd.read_csv(stock_path, index_col=0)
            stocks_pd = stocks_pd.rename(columns={'Unnamed: 0': 'date'})
            if index == 0:
                print('single EOD data shape:', stocks_pd.shape)  # days*6
                self.volumn = np.zeros([len(self.tickers), stocks_pd.shape[0]], dtype=np.float32)
                self.eod_data = np.zeros([len(self.tickers), stocks_pd.shape[0], 5],
                                         dtype=np.float32)
                self.masks = np.ones([len(self.tickers), stocks_pd.shape[0]], dtype=np.int8)
                self.close = np.zeros([len(self.tickers), stocks_pd.shape[0]], dtype=np.float32)
                self.open = np.zeros([len(self.tickers), stocks_pd.shape[0]], dtype=np.float32)
            stocks_pd = stocks_pd.values
            for row in range(stocks_pd.shape[0]):
                if abs(stocks_pd[row][-1] + 1234) < 1e-8:
                    self.masks[index][row] = 0.0
                for col in range(stocks_pd.shape[1]):
                    if abs(stocks_pd[row][col] + 1234) < 1e-8:
                        stocks_pd[row][col] = 1
            self.volumn[index] = stocks_pd[:, 4]
            self.eod_data[index, :, :] = stocks_pd
            self.close[index] = stocks_pd[:, 3]
            self.open[index] = stocks_pd[:, 0]
            # self.stocks_close[index] = stocks_pd['Close'].values + 1
            # self.stocks_open[index] = stocks_pd['Open'].values + 1
            # stocks_pd = stocks_pd.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
            # stocks_pd = stocks_pd.fillna(0)
            # self.close_norm[index] = stocks_pd['Close'].values
            # self.stocks_data[index] = stocks_pd.values

    def __getitem__(self, index):
        macro_data = 0
        mask_batch = self.masks[:, index: index + self.seq_len + 1]
        mask_batch = np.min(mask_batch, axis=1)
        stocks_data = self.eod_data[:, index:index + self.seq_len]
        stocks_close_trend = self.close[:, index:index + 30]
        stocks_close = self.close[:, index + self.seq_len]
        stocks_open = self.open[:, index + self.seq_len]
        # date = self.date[index + self.seq_len]
        # macro_data = get_macro_data(date)
        return stocks_data, stocks_close, stocks_open, stocks_close_trend, mask_batch, macro_data

    def __len__(self):
        return self.eod_data.shape[1] - 30


# news_gpt = pd.read_csv('./data/sector_days.csv', index_col=0)
def get_macro_data(date):
    path = './data/macro/'
    one_data = []
    macro_data = [path + macro_data_path for macro_data_path in os.listdir(path)]
    for macro in macro_data:
        if 'ipy' in macro:
            continue
        pd_macro = pd.read_csv(macro, index_col=0)
        cur_date = datetime.datetime.strptime(date, "%Y-%m-%d")
        for i in range(len(pd_macro)):
            try:
                macro_date = datetime.datetime.strptime(pd_macro['日期'][i], "%Y-%m-%d")
            except KeyError:
                print(macro)
            if macro_date >= cur_date:
                for column in pd_macro.columns:
                    if not pd_macro.iloc[i][column]:
                        one_data.append(0)
                    elif type(pd_macro.iloc[i][column]) != str:
                        one_data.append(pd_macro.iloc[i][column])
                break
    return one_data
