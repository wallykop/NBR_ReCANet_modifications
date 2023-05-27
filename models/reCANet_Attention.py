import torch
import torch.nn as nn
import torch.nn.functional as F


class ReCaNet_Attention(nn.Module):
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
            dense_2_hidden_size: int
            ):
        super(ReCaNet_Attention, self).__init__()

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
            num_layers=1,
            bidirectional=False
            )

        self.dense_1 = nn.Linear(
            in_features=self.lstm_hidden_size,
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

    def attention_net(
            self,
            lstm_output,
            final_state
            ):
        batch_size = len(lstm_output)
        hidden = final_state.view(batch_size, -1, 1)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)

        context = torch.bmm(
            lstm_output.transpose(1, 2),
            soft_attn_weights.unsqueeze(2)).squeeze(2)
        
        return context, soft_attn_weights

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
        
        output_state, (hx, _) = self.LSTM(user_item_history_embed)

        last_hidden_state, _ = self.attention_net(output_state, hx)

        output_vector = F.relu(
            input=self.dense_1(last_hidden_state)
            )
        output_vector = F.relu(
            input=self.dense_2(output_vector)
            )

        pred = self.dense_output(output_vector)

        output = pred.view(-1)
        output = torch.sigmoid(output)

        return output


class Bidir_ReCaNet_Attention(nn.Module):
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
            dense_2_hidden_size: int
            ):
        super(Bidir_ReCaNet_Attention, self).__init__()

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
            num_layers=1,
            bidirectional=True
            )

        self.dense_1 = nn.Linear(
            in_features=self.lstm_hidden_size*2,
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

    def attention_net(
            self,
            lstm_output,
            final_state
            ):
        batch_size = len(lstm_output)
        hidden = final_state.view(batch_size, -1, 1)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)

        context = torch.bmm(
            lstm_output.transpose(1, 2),
            soft_attn_weights.unsqueeze(2)).squeeze(2)
        
        return context, soft_attn_weights

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
        
        output_state, (hx, _) = self.LSTM(user_item_history_embed)

        last_hidden_state, _ = self.attention_net(output_state, hx)

        output_vector = F.relu(
            input=self.dense_1(last_hidden_state)
            )
        output_vector = F.relu(
            input=self.dense_2(output_vector)
            )

        pred = self.dense_output(output_vector)

        output = pred.view(-1)
        output = torch.sigmoid(output)

        return output
