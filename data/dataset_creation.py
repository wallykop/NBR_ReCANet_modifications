import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from tqdm import tqdm
import scipy
from scipy.sparse import csr_array


# preparind dataset for torch.dataset class
class DatasetInit():
    def __init__(
            self,
            path_train,
            path_val,
            path_test,
            dataset,
            history_len,
            basket_count_min=0,
            min_item_count=0
            ):
        
        self.basket_count_min = basket_count_min
        self.min_item_count = min_item_count
        self.history_len = history_len
        
        self.train_baskets = pd.read_csv(path_train)
        self.valid_baskets = pd.read_csv(path_val)
        self.test_baskets = pd.read_csv(path_test)
        
        basket_per_user = self.train_baskets[['user_id', 'basket_id']].drop_duplicates().groupby('user_id').agg({'basket_id': 'count'}).reset_index()
        self.test_users = basket_per_user[basket_per_user['basket_id'] >= self.basket_count_min]['user_id'].tolist()
    
        print("number of test users:", len(self.test_users))
        
        self.model_name = 'data/dunnhumby_cj/' + dataset + '_recanet'
        self.dataset = dataset
        self.all_items = self.train_baskets[['item_id']].drop_duplicates()['item_id'].tolist()
        self.all_users = self.train_baskets[['user_id']].drop_duplicates()['user_id'].tolist()
        self.num_items = len(self.all_items) + 1
        self.num_users = len(self.all_users) + 1

        print("items:", self.num_items)
        item_counts = self.train_baskets.groupby(['item_id']).size().to_frame(name='item_count').reset_index()
        item_counts = item_counts[item_counts['item_count'] >= min_item_count]
        item_counts_dict = dict(zip(item_counts['item_id'], item_counts['item_count']))
        print("filtered items:", len(item_counts_dict))
        self.num_items = len(item_counts_dict) + 1
        self.item_id_mapper = {}
        self.id_item_mapper = {}
        self.user_id_mapper = {}
        self.id_user_mapper = {}

        counter = 0
        for i in range(len(self.all_items)):
            if self.all_items[i] in item_counts_dict:
                self.item_id_mapper[self.all_items[i]] = counter+1
                self.id_item_mapper[counter+1] = self.all_items[i]
                counter += 1
        for i in range(len(self.all_users)):
            self.user_id_mapper[self.all_users[i]] = i+1
            self.id_user_mapper[i+1] = self.all_users[i]

    def create_large_train_data(self):
        if os.path.isfile(self.model_name + '_' + str(self.history_len) + '_train_users.npy'):
            print('Data allready in use')
            train_users = np.load(self.model_name + '_' + str(self.history_len) + '_train_users.npy')
            train_items = np.load(self.model_name + '_' + str(self.history_len) + '_train_items.npy')

            train_history2 = scipy.sparse.load_npz(self.model_name + '_' + str(self.history_len) + '_train_history2.npz')
            train_labels = np.load(self.model_name + '_' + str(self.history_len) + '_train_labels.npy')
            return train_items, train_users, train_history2, train_labels

        basket_items = self.train_baskets.groupby(['basket_id'])['item_id'].apply(list).reset_index()
        basket_items_dict = dict(zip(basket_items['basket_id'], basket_items['item_id']))
        basket_items_dict['null'] = []

        user_baskets = self.train_baskets[['user_id', 'date', 'basket_id']].drop_duplicates().\
            sort_values(['user_id', 'date'], ascending=True).groupby(['user_id'])['basket_id'].apply(list).reset_index()

        user_baskets_dict = dict(zip(
            user_baskets['user_id'],
            user_baskets['basket_id']
            ))

        train_users = []
        train_items = []
        train_history = csr_array((0, 20))
        train_history2 = csr_array((0, 20))
        train_labels = []
        print('num users:', len(self.test_users))

        for c, user in tqdm(enumerate(self.test_users)):
            if c % 1000 == 1:
                print(c, 'user passed')

            baskets = user_baskets_dict[user]
            item_seq = {}
            for i, basket in enumerate(baskets):
                for item in basket_items_dict[basket]:
                    if item not in self.item_id_mapper:
                        continue
                    if item not in item_seq:
                        item_seq[item] = []
                    item_seq[item].append(i)

            train_history_tmp = []
            train_history2_tmp = []
            for i in range(max(0, len(baskets)-50), len(baskets)):
                # print(i)
                label_basket = baskets[i]
                all_history_baskets = baskets[:i]
                items = []
                for basket in all_history_baskets:
                    for item in basket_items_dict[basket]:
                        items.append(item)
                items = list(set(items))
                for item in items:
                    if item not in self.item_id_mapper:
                        continue
                    index = np.argmax(np.array(item_seq[item]) >= i)
                    if np.max(np.array(item_seq[item])) < i:
                        index = len(item_seq[item])
                    input_history = item_seq[item][:index].copy()
                    if len(input_history) == 0:
                        continue
                    if len(input_history) == 1 and input_history[0] == -1:
                        continue
                    while len(input_history) < self.history_len:
                        input_history.insert(0, -1)
                    real_input_history = []
                    for x in input_history:
                        if x == -1:
                            real_input_history.append(0)
                        else:
                            real_input_history.append(i-x)
                    real_input_history2 = []
                    for j, x in enumerate(input_history[:-1]):
                        if x == -1:
                            real_input_history2.append(0)
                        else:
                            real_input_history2.append(input_history[j+1]-input_history[j])
                    real_input_history2.append(i-input_history[-1])
                    train_users.append(self.user_id_mapper[user])
                    train_items.append(self.item_id_mapper[item])

                    train_history_tmp.append(real_input_history[-self.history_len:])
                    train_history2_tmp.append(real_input_history2[-self.history_len:])

                    train_labels.append(float(item in basket_items_dict[label_basket]))

            train_history = scipy.sparse.vstack([train_history, csr_array(train_history_tmp)])
            train_history2 = scipy.sparse.vstack([train_history2, csr_array(train_history2_tmp)])
            print(train_history.shape)

        train_items = np.array(train_items)
        train_users = np.array(train_users)

        train_labels = np.array(train_labels)
        random_indices = np.random.choice(range(len(train_items)), len(train_items), replace=False)
        train_items = train_items[random_indices]
        train_users = train_users[random_indices]
        train_history = train_history[random_indices]
        train_history2 = train_history2[random_indices]
        train_labels = train_labels[random_indices]

        np.save(self.model_name + '_' + str(self.history_len) + '_train_items.npy', train_items)
        np.save(self.model_name + '_' + str(self.history_len) + '_train_users.npy', train_users)

        scipy.sparse.save_npz(self.model_name + '_' + str(self.history_len) + '_train_history2.npz', train_history2)
        np.save(self.model_name + '_' + str(self.history_len) + '_train_labels.npy', train_labels)

        return train_items, train_users, train_history2, train_labels

    def create_train_data(self):
        if os.path.isfile(self.model_name + '_' + str(self.history_len) + '_train_users.npy'):
            print('Data allready in use')
            train_users = np.load(self.model_name + '_' + str(self.history_len) + '_train_users.npy')
            train_items = np.load(self.model_name + '_' + str(self.history_len) + '_train_items.npy')

            train_history2 = scipy.sparse.load_npz(self.model_name + '_' + str(self.history_len) + '_train_history2.npz')
            train_labels = np.load(self.model_name + '_' + str(self.history_len) + '_train_labels.npy')
            return train_items, train_users, train_history2, train_labels
        row_counts = 0
        basket_items = self.train_baskets.groupby(['basket_id'])['item_id'].apply(list).reset_index()
        basket_items_dict = dict(zip(basket_items['basket_id'], basket_items['item_id']))
        basket_items_dict['null'] = []

        user_baskets = self.train_baskets[['user_id', 'date', 'basket_id']].drop_duplicates().\
            sort_values(['user_id', 'date'], ascending=True).groupby(['user_id'])['basket_id'].apply(list).reset_index()

        user_baskets_dict = dict(zip(
            user_baskets['user_id'],
            user_baskets['basket_id']
            ))

        train_users = []
        train_items = []
        train_history = []
        train_history2 = []
        train_labels = []
        print('num users:', len(self.test_users))

        for c, user in tqdm(enumerate(self.test_users)):
            if c % 1000 == 1:
                print(c, 'user passed')

            baskets = user_baskets_dict[user]
            item_seq = {}
            for i, basket in enumerate(baskets):
                for item in basket_items_dict[basket]:
                    if item not in self.item_id_mapper:
                        continue
                    if item not in item_seq:
                        item_seq[item] = []
                    item_seq[item].append(i)

            for i in range(max(0, len(baskets)-50), len(baskets)):
                label_basket = baskets[i]
                all_history_baskets = baskets[:i]
                items = []
                for basket in all_history_baskets:
                    for item in basket_items_dict[basket]:
                        items.append(item)
                items = list(set(items))
                for item in items:
                    if item not in self.item_id_mapper:
                        continue
                    index = np.argmax(np.array(item_seq[item]) >= i)
                    if np.max(np.array(item_seq[item])) < i:
                        index = len(item_seq[item])
                    input_history = item_seq[item][:index].copy()
                    if len(input_history) == 0:
                        continue
                    if len(input_history) == 1 and input_history[0] == -1:
                        continue
                    while len(input_history) < self.history_len:
                        input_history.insert(0, -1)
                    real_input_history = []
                    for x in input_history:
                        if x == -1:
                            real_input_history.append(0)
                        else:
                            real_input_history.append(i-x)
                    real_input_history2 = []
                    for j, x in enumerate(input_history[:-1]):
                        if x == -1:
                            real_input_history2.append(0)
                        else:
                            real_input_history2.append(input_history[j+1]-input_history[j])
                    real_input_history2.append(i-input_history[-1])
                    train_users.append(self.user_id_mapper[user])
                    train_items.append(self.item_id_mapper[item])
                    train_history.append(real_input_history[-self.history_len:])
                    train_history2.append(real_input_history2[-self.history_len:])
                    train_labels.append(float(item in basket_items_dict[label_basket]))

                    row_counts += 1
            # print(row_counts)

        train_items = np.array(train_items)
        train_users = np.array(train_users)
        train_history = np.array(train_history)
        train_history2 = np.array(train_history2)
        train_labels = np.array(train_labels)
        random_indices = np.random.choice(range(len(train_items)), len(train_items), replace=False)
        train_items = train_items[random_indices]
        train_users = train_users[random_indices]
        train_history = train_history[random_indices]
        train_history2 = train_history2[random_indices]
        train_labels = train_labels[random_indices]

        np.save(self.model_name + '_' + str(self.history_len) + '_train_items.npy', train_items)
        np.save(self.model_name + '_' + str(self.history_len) + '_train_users.npy', train_users)
        
        np.save(self.model_name + '_' + str(self.history_len) + '_train_history2.npy', train_history2)
        np.save(self.model_name + '_' + str(self.history_len) + '_train_labels.npy', train_labels)

        return train_items, train_users, train_history2, train_labels
    
    def create_test_data(
            self,
            test_data='test'
            ):
        if os.path.isfile(self.model_name + '_' + str(self.history_len) + '_' + test_data + '_users.npy'):
            test_users = np.load(
                self.model_name + '_' + str(self.history_len) + '_' + test_data + '_users.npy'
                )
            test_items = np.load(
                self.model_name + '_' + str(self.history_len) + '_' + test_data + '_items.npy')
        
            test_history2 = np.load(
                self.model_name + '_' + str(self.history_len) + '_' + test_data + '_history2.npy')
            test_labels = np.load(
                self.model_name + '_' + str(self.history_len) + '_' + test_data + '_labels.npy')
            return test_items, test_users,  test_history2, test_labels

        train_basket_items = self.train_baskets.groupby(['basket_id'])['item_id'].apply(list).reset_index()
        train_basket_items_dict = dict(zip(train_basket_items['basket_id'], train_basket_items['item_id']))

        train_user_baskets = self.train_baskets[['user_id', 'date', 'basket_id']].drop_duplicates(). \
            sort_values(['user_id', 'date'], ascending=True).groupby(['user_id'])['basket_id'].apply(list).reset_index()
        train_user_baskets_dict = dict(zip(train_user_baskets['user_id'], train_user_baskets['basket_id']))

        train_user_items = self.train_baskets[['user_id', 'item_id']].drop_duplicates().groupby(['user_id'])['item_id'] \
            .apply(list).reset_index()
        train_user_items_dict = dict(zip(train_user_items['user_id'], train_user_items['item_id']))

        test_user_items = None
        if test_data == 'test':
            test_user_items = self.test_baskets.groupby(['user_id'])['item_id'].apply(list).reset_index()
        else:
            test_user_items = self.valid_baskets.groupby(['user_id'])['item_id'].apply(list).reset_index()
        test_user_items_dict = dict(zip(test_user_items['user_id'], test_user_items['item_id']))

        test_users = []
        test_items = []
        test_history = []
        test_history2 = []
        test_labels = []

        train_basket_items_dict['null'] = []
        for c, user in tqdm(enumerate(test_user_items_dict)):
            if user not in train_user_baskets_dict:
                continue
            if c % 100 == 1:
                print(c, 'user passed')

            baskets = train_user_baskets_dict[user]
            item_seq = {}
            for i, basket in enumerate(baskets):
                for item in train_basket_items_dict[basket]:
                    if item not in self.item_id_mapper:
                        continue
                    if item not in item_seq:
                        item_seq[item] = []
                    item_seq[item].append(i)

            label_items = test_user_items_dict[user]

            items = list(set(train_user_items_dict[user]))

            for item in items:
                if item not in self.item_id_mapper:
                    continue
                input_history = item_seq[item][-self.history_len:]
                if len(input_history) == 0:
                    continue
                if len(input_history) == 1 and input_history[0] == -1:
                    continue
                while len(input_history) < self.history_len:
                    input_history.insert(0, -1)
                real_input_history = []
                for x in input_history:
                    if x == -1:
                        real_input_history.append(0)
                    else:
                        real_input_history.append(len(baskets)-x)

                real_input_history2 = []
                for j, x in enumerate(input_history[:-1]):
                    if x == -1:
                        real_input_history2.append(0)
                    else:
                        real_input_history2.append(input_history[j+1]-input_history[j])
                real_input_history2.append(len(baskets)-input_history[-1])
                test_users.append(self.user_id_mapper[user])
                test_items.append(self.item_id_mapper[item])
                test_history.append(real_input_history)
                test_history2.append(real_input_history2)
                test_labels.append(float(item in label_items))

        test_items = np.array(test_items)
        test_users = np.array(test_users)
        test_history = np.array(test_history)
        test_history2 = np.array(test_history2)
        test_labels = np.array(test_labels)

        np.save(self.model_name + '_' + str(self.history_len) + '_' + test_data + '_items.npy', test_items)
        np.save(self.model_name + '_' + str(self.history_len) + '_' + test_data + '_users.npy', test_users)

        np.save(self.model_name + '_' + str(self.history_len) + '_' + test_data + '_history2.npy', test_history2)
        np.save(self.model_name + '_' + str(self.history_len) + '_' + test_data + '_labels.npy', test_labels)

        return test_items, test_users, test_history2, test_labels


class CustomDatasetSmall(Dataset):
    def __init__(
            self,
            dataset,
            mode
            ):

        self.mode = mode
        if self.mode == 'train':
            self.item_input, self.user_input,  self.history_input, self.target = dataset.create_train_data()
        elif self.mode == 'val':
            self.item_input, self.user_input,  self.history_input, self.target = dataset.create_test_data(
                test_data='val'
                )
        elif self.mode == 'test':
            self.item_input, self.user_input,  self.history_input, self.target = dataset.create_test_data(
                test_data='test'
                )
        else:
            print('Mode error')
        
        self.item_input = torch.tensor(self.item_input).long()
        self.user_input = torch.tensor(self.user_input).long()
        self.history_input = torch.FloatTensor(self.history_input)
        self.target = torch.FloatTensor(self.target)

    def __len__(self):
        return len(self.item_input)

    def __getitem__(
            self,
            idx
            ):
        item = (
            self.item_input[idx],
            self.user_input[idx],
            self.history_input[idx],
            self.target[idx]
            )

        return item


class CustomDatasetLarge(Dataset):
    def __init__(
            self,
            dataset,
            mode
            ):

        self.mode = mode

        if self.mode == 'train':
            self.item_input, self.user_input,  self.history_input, self.target = dataset.create_large_train_data()
        else:
            print('Mode error')

        self.item_input = torch.tensor(self.item_input).long()
        self.user_input = torch.tensor(self.user_input).long()
        self.target = torch.FloatTensor(self.target)

    def __len__(self):
        return len(self.item_input)

    def __getitem__(self, idx):
        item = (
            self.item_input[idx],
            self.user_input[idx],
            torch.FloatTensor(self.history_input[idx].toarray()).squeeze(),
            self.target[idx]
            )
        
        return item


class ToDevice():
    """Wrap a dataloader to move data to a device"""
    def __init__(
            self,
            dl,
            device
            ):
        self.dl = dl
        self.device = device

    def to_device(
            self,
            data,
            device
            ):
        if isinstance(data, (list, tuple)):
            return [self.to_device(data=x, device=device) for x in data]
        
        return data.to(device, non_blocking=True)
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield self.to_device(
                data=b,
                device=self.device
                )

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
