import torch.nn as nn


class FiLM(nn.Module):
    def __init__(self, input_channels, embedding_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.a = nn.Linear(embedding_channels, input_channels)
        self.b = nn.Linear(embedding_channels, input_channels)

    def forward(self, x, emb):
        emb = emb.transpose(1, 3)
        
        a = self.a(emb).transpose(1, 3)
        b = self.b(emb).transpose(1, 3)
        return x * a + b