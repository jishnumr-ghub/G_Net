
import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        return output, attn

class GESTURE_NET(nn.Module):
    def __init__(self, input_dim=2048, num_gestures=15, num_classes=2, window_size=5, gru_hidden_size=64, attn_dim=64, gesture_embed_dim=32):
        super().__init__()
        self.conv1d = nn.Conv1d(input_dim, 128, kernel_size=window_size, padding=window_size//2)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(128, gru_hidden_size, batch_first=True)
        self.attention = ScaledDotProductAttention(d_k=attn_dim)
        self.gesture_embedding = nn.Embedding(num_gestures, gesture_embed_dim)
        self.agnostic_fc = nn.Conv1d(gru_hidden_size, 1, kernel_size=1)
        self.fusion = nn.Linear(gru_hidden_size + gesture_embed_dim, gru_hidden_size)
        self.aware_fc = nn.Conv1d(gru_hidden_size, 1, kernel_size=1)
        self.weighting = nn.Linear(gru_hidden_size, 2)

    def forward(self, x, gesture):
        x = x.transpose(1, 2)
        x = self.relu(self.conv1d(x))
        x = x.transpose(1, 2)
        gru_out, _ = self.gru(x)
        Q = K = V = gru_out
        attn_out, _ = self.attention(Q, K, V)

        # Gesture-agnostic path
        agnostic_logits = self.agnostic_fc(attn_out.transpose(1, 2)).squeeze(1)

        # Gesture-aware path
        embed = self.gesture_embedding(gesture)
        fusion_input = torch.cat([attn_out, embed], dim=-1)
        fusion_out = self.relu(self.fusion(fusion_input))
        aware_logits = self.aware_fc(fusion_out.transpose(1, 2)).squeeze(1)

        # Dynamic fusion
        weights = torch.softmax(self.weighting(attn_out), dim=-1)
        final = weights[..., 0] * agnostic_logits + weights[..., 1] * aware_logits
        return final
