import torch
import torch.nn as nn
from src.models.common.batched_lstm import BatchedLSTM


class GridNetBlock(nn.Module):
    def __init__(
        self,
        latent_dim,
        n_freqs,
        hidden_channels = 32,
        bidirectional=False,
    ):
        super().__init__()
        self.time_domain_bidirectional = bidirectional # Causal
        
        self.H = hidden_channels
        self.n_freqs = n_freqs
        
        # Intra-frame processing
        self.intra_norm = nn.LayerNorm(latent_dim)
        self.intra_seq2seq = nn.LSTM(
            latent_dim, hidden_channels, 1, batch_first=True, bidirectional=True
        )
        self.intra_linear = nn.Linear(
            2 * hidden_channels, latent_dim,
        )

        # Time-domain LSTM
        self.inter_norm = nn.LayerNorm(latent_dim)
        self.inter_rnn = nn.LSTM( 
            latent_dim, hidden_channels, 1, batch_first=True,
            bidirectional=self.time_domain_bidirectional,
        )
        self.inter_linear = nn.Linear(
            hidden_channels*(self.time_domain_bidirectional + 1), latent_dim
        )

        # Edge mode
        self.edge = False
        
    def init_buffers(self, batch_size, device):
        ctx_buf = {}
        First_dimension_hidden_states = 1
        if self.time_domain_bidirectional:
            First_dimension_hidden_states = 2
        if not self.edge:
            c0 = torch.zeros((First_dimension_hidden_states,
                            batch_size * self.n_freqs,
                            self.H), device=device)
            h0 = torch.zeros((First_dimension_hidden_states,
                            batch_size * self.n_freqs,
                            self.H), device=device)
        else:
            c0 = torch.zeros((First_dimension_hidden_states,
                            self.H,
                            batch_size * self.n_freqs), device=device)
            h0 = torch.zeros((First_dimension_hidden_states,
                            self.H,
                            batch_size * self.n_freqs), device=device)
            
        ctx_buf['c0'] = c0
        ctx_buf['h0'] = h0

        return ctx_buf

    def forward(self, x, init_state = None):
        """GridNetBlock Forward.

        Args:
            x: [B, T, Q, C]
            out: [B, T, Q, C]
        """
        
        if init_state is None:
            init_state = self.init_buffers(x.shape[0], Q.device)

        B, T, Q, C = x.shape
        # Store input for residual connection
        input_ = x

        intra_rnn = x.reshape(B * T, Q, C)  # [B * T, Q, C]

        # Intra-frame processing
        intra_rnn = self.intra_norm(intra_rnn) # LayerNorm
        intra_rnn, _ = self.intra_seq2seq(intra_rnn)  # [BT, *, H]
        intra_rnn = self.intra_linear(intra_rnn)  # [BT, *, C]
        
        intra_rnn = intra_rnn.view(B, T, Q, C) # [B, T, Q, C]
        intra_rnn = intra_rnn + input_  # [B, T, Q, C]
        out = intra_rnn

        # Inter-frame processing
        input_ = intra_rnn # [B, T, Q, C]
        
        inter_rnn = input_  # [B, T, Q, C]
        inter_rnn = self.inter_norm(inter_rnn) # LayerNorm
        
        h0 = init_state['h0']
        c0 = init_state['c0']

        # Inter frame processing with state updates
        if not self.edge:
            inter_rnn = inter_rnn.transpose(1, 2).reshape(B * Q, T, C)  # [BQ, T, C]
            
            self.inter_rnn.flatten_parameters()
            inter_rnn, (h0, c0) = self.inter_rnn(inter_rnn, (h0, c0))  # [BQ, -1, H]
            if self.time_domain_bidirectional:
                inter_rnn = inter_rnn.view([B, Q, T, 2*self.H]).transpose(1, 2) # [B, T, Q, 2H]
            else:
                inter_rnn = inter_rnn.view([B, Q, T, self.H]).transpose(1, 2) # [B, T, Q, H]
        else:
            assert T == 1, f"In edge mode, there must be only 1 frame. Found {T}"
            inter_rnn = inter_rnn.squeeze(1) # [B, Q, H]
            inter_rnn, (h0, c0) = self.inter_rnn(inter_rnn, (h0, c0))  # [B, Q, H]
            inter_rnn = inter_rnn.unsqueeze(1) # [B, T, Q, H]
       
        init_state['h0'] = h0
        init_state['c0'] = c0
        
        inter_rnn = self.inter_linear(inter_rnn)  # [*, C]
        
        inter_rnn = inter_rnn + input_  # [B, T, Q, C]
        
        out = inter_rnn

        return out, init_state

    def edge_mode(self):
        state_dict = self.inter_rnn.state_dict()
        self.inter_rnn = BatchedLSTM(self.inter_rnn.input_size, self.inter_rnn.hidden_size)
        self.inter_rnn.set_weights(state_dict)
        
        self.edge = True
