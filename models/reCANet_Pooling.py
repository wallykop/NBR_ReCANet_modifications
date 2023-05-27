import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ReCaNet_Pooling(nn.Module):
    def __init__(
            self,
            num_items: int,
            item_embed_size: int,
            num_users: int,
            user_embed_size: int,
            history_len: int,
            user_item_hidden_size: int,
            user_item_history_hidden_size: int,
            lstm_hidden_size: int,
            dense_1_hidden_size: int,
            dense_2_hidden_size: int,
            poolings: List[nn.Module]
            ):
        super(ReCaNet_Pooling, self).__init__()

        self.num_items = num_items
        self.item_embed_size = item_embed_size
        self.num_users = num_users
        self.user_embed_size = user_embed_size
        self.history_len = history_len

        self.user_item_hidden_size = user_item_hidden_size
        self.user_item_history_hidden_size = user_item_history_hidden_size
        self.lstm_hidden_size = lstm_hidden_size
        self.dense_1_hidden_size = dense_1_hidden_size
        self.dense_2_hidden_size = dense_2_hidden_size

        self.item_embedding = nn.Embedding(
            num_embeddings=self.num_items,
            embedding_dim=self.item_embed_size
            )
        
        self.user_embedding = nn.Embedding(
            num_embeddings=self.num_users,
            embedding_dim=self.user_embed_size
            )

        self.user_item_fc = nn.Linear(
            in_features=self.item_embed_size + self.user_embed_size,
            out_features=self.user_item_hidden_size
            )
        self.user_item_history_fc = nn.Linear(
            in_features=self.user_item_hidden_size + 1,  # added history vector
            out_features=self.user_item_history_hidden_size
            )
        
        self.LSTM = nn.LSTM(
            input_size=self.user_item_history_hidden_size,
            hidden_size=self.lstm_hidden_size,
            batch_first=True,
            num_layers=2
            )

        self.dense_1 = nn.Linear(
            in_features=self.lstm_hidden_size * len(poolings),
            out_features=self.dense_1_hidden_size
            )
        self.dense_2 = nn.Linear(
            in_features=self.dense_1_hidden_size,
            out_features=self.dense_2_hidden_size
            )
        self.dense_output = nn.Linear(
            in_features=self.dense_2_hidden_size,
            out_features=1
            )
        
        self.pooling = nn.ModuleList(poolings)

    def forward(
            self,
            item_input,
            user_input,
            history_input
            ):
        item_embed = self.item_embedding(
            item_input.to(torch.int64)).view(-1, self.item_embed_size)

        user_embed = self.user_embedding(
            user_input.to(torch.int64)).view(-1, self.user_embed_size)

        history_vector = history_input.unsqueeze(2)
        
        user_item_embed = torch.cat(
            tensors=(item_embed, user_embed),
            axis=1
            )

        user_item_embed = F.relu(
            input=self.user_item_fc(user_item_embed)
            )

        user_item_embed = user_item_embed.unsqueeze(1)
        user_item_embed = user_item_embed.repeat(1, self.history_len, 1)
        
        user_item_history_embed = torch.cat(
            tensors=(user_item_embed, history_vector),
            axis=2
            )
        
        user_item_history_embed = F.relu(
            input=self.user_item_history_fc(user_item_history_embed)
            )
        
        hidden_state, (_, _) = self.LSTM(user_item_history_embed)

        pooled_list = []
        for pool in self.pooling:
            pooled_list.append(pool(hidden_state))
        
        pooled_embedding = torch.cat(pooled_list, dim=1)

        output_vector = F.relu(
            input=self.dense_1(pooled_embedding)
            )
        output_vector = F.relu(
            input=self.dense_2(output_vector)
            )

        pred = self.dense_output(output_vector)

        output = pred.view(-1)
        output = torch.sigmoid(output)

        return output
