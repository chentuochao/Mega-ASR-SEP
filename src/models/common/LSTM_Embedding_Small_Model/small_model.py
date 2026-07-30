import torch
import torch.nn as nn
import torch.nn.functional as F

def count_parameters(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return trainable, total


class FiLM(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.gamma_layer = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU()
            )
        self.beta_layer = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU()
            )

    def forward(self, x1, cond):
        """
        x:     (B, T, D)  — base input
        cond:  (B, T, D)  — modulator input
        """
        #x1 = x1.clone()
        gamma = self.gamma_layer(cond)
        beta =  self.beta_layer(cond)
        return gamma * x1 + beta                    # FiLM modulation


class Combine_Big_and_Speaker_Embed(nn.Module):
    def __init__(self, speaker_dim, embedding_dim):
        super().__init__()
        self.gamma_layer = nn.Sequential(
            nn.Linear(speaker_dim, embedding_dim)
            )
        self.beta_layer = nn.Sequential(
            nn.Linear(speaker_dim, embedding_dim)
            )

    def forward(self, x1, cond):
        #print("cond:", cond.shape)
        cond = cond[:, None, None, :].repeat(1, x1.shape[1], x1.shape[2], 1)
        #print("cond2:", cond.shape)
        gamma = self.gamma_layer(cond)
        beta =  self.beta_layer(cond)
        return gamma * x1 + beta                    # FiLM modulation

class StateSpaceBlock(nn.Module):
    def __init__(self, 
                 small_num_microphones, channels, 
                 big_frequency_layer_dimension, small_frequency_layer_dimension, 
                 apply_embedding=True):
        super().__init__()
        
        self.small_frequency_layer_dimension = small_frequency_layer_dimension
        self.big_frequency_layer_dimension = big_frequency_layer_dimension

        self.encoder_layer1 = nn.Linear(in_features=2*small_num_microphones, out_features=channels)
        self.encoder_layer2 = nn.Conv1d(in_channels=small_frequency_layer_dimension, out_channels=big_frequency_layer_dimension, kernel_size=1)
        
        if apply_embedding:
            self.film_layer = FiLM(embedding_dim=channels)
        
        self.middle_layer1 = nn.Sequential(
            nn.Linear(in_features=channels, out_features=2),
            nn.ReLU()
        )

        self.middle_layer2 = nn.Sequential(
            nn.Conv1d(in_channels=big_frequency_layer_dimension, out_channels=small_frequency_layer_dimension, kernel_size=1),
            nn.ReLU()
        )

        self.time_layer = nn.LSTM(input_size=2*small_frequency_layer_dimension, hidden_size=2*small_frequency_layer_dimension, num_layers=1,
            bias=True, batch_first=True, dropout=0.0, bidirectional=False)
        self.activation = nn.ReLU()

        self.dense_layer = nn.Sequential(
            nn.Linear(in_features=2*small_frequency_layer_dimension, out_features=2*small_frequency_layer_dimension*small_num_microphones)
        )
        
    def forward(self, x, big_model_embedding, h00, c00):

        # x = inputs['mixture'] # [B, T, F_S, 2*M]
        # big_model_embedding # [B, T, F_B, C]
        # output = [B, T, F_S, 2*M]
        
        BB, TT, F_S, _ = x.shape
        x = x.reshape(BB*TT, self.small_frequency_layer_dimension, -1)
        
        x = self.encoder_layer1(x)                              # [B*T, F_S, C]
        x = self.encoder_layer2(x)                              # [B*T, F_B, C]
        
        if big_model_embedding is not None:
            big_model_embedding = big_model_embedding.reshape(BB*TT, self.big_frequency_layer_dimension, -1)
            x = self.film_layer(x, big_model_embedding)             # [B*T, F_B, C]
        
        x = self.middle_layer1(x)                               # [B*T, F_B, 2]
        x = self.middle_layer2(x)                               # [B*T, F_S, 2]
        x = x.reshape(BB*TT, -1)                                # [B*T, F_S*2]
        x = x.reshape(BB, TT, -1)                               # [B, T, F_S*2]
        ##########################
        x, (h00, c00) = self.time_layer(x, (h00, c00))          # [B, T, F_S*2]
        #x = self.ln(x)
        x = self.activation(x)
        ##########################
        x = self.dense_layer(x)                                 # [B, T, F_S*2*M]
        x = x.reshape(x.shape[0], x.shape[1], self.small_frequency_layer_dimension, -1)            # [B, T, F_S, 2*M]
        
        return x, h00, c00

class SmallModel(nn.Module):
    def __init__(self, 
                 small_num_microphones, channels, n_layers, n_srcs,
                 big_frequency_layer_dimension, n_fft, apply_embedding=True):
        super().__init__()
        print(f"Initalizing small model {n_layers} layers ...")

        self.edge = False
        self.small_frequency_layer_dimension = int((n_fft)/2) + 1
        self.big_frequency_layer_dimension = big_frequency_layer_dimension
        self.small_num_microphones = small_num_microphones
        self.channels = channels
        self.n_layers = n_layers
        self.n_srcs = n_srcs

        self.StateSpaceBlock = nn.ModuleList([])
        for _ in range(n_layers):
            self.StateSpaceBlock.append(StateSpaceBlock(small_num_microphones, channels, 
                                    self.big_frequency_layer_dimension, self.small_frequency_layer_dimension, apply_embedding))

        self.encoder_to_reduce = nn.Linear(in_features=2*small_num_microphones, out_features=2)
        
    def init_buffers(self, batch_size, device):
        state_buffer = {}
        channel_dim = 2*self.small_frequency_layer_dimension

        for ii in range(self.n_layers):
            state_buffer[f'h{ii}'] = torch.zeros((1, batch_size, channel_dim), device=device)
            state_buffer[f'c{ii}'] = torch.zeros((1, batch_size, channel_dim), device=device)

        return state_buffer
    
    def edge_mode(self):
        self.edge = True
    
    def forward(self, x, big_model_embedding=None, input_state=None):
        """
        # x = inputs['mixture']    # real real ... Imag Imag      # [B, 2*M, T, F_S]
        # big_model_embedding.                                    # [B, T, F_B, C]
        # output                                                  # [B, 2, T, F_S]

        """                                                     
        
        B, R, T, F_S = x.shape
        x = x.permute(0,2,3,1)                                   # [B, T, F_S, 2*M]
        
        

        if input_state is None:
            input_state = self.init_buffers(B, x.device)

        # [B, T, F_S, 2*Mic]
        for ii in range(self.n_layers):
            y, input_state[f"h{ii}"], input_state[f"c{ii}"] = self.StateSpaceBlock[ii](x, big_model_embedding, input_state[f"h{ii}"], input_state[f"c{ii}"])      # [B, T, F_S, 2*M]
            x = x + y
        
        x = self.encoder_to_reduce(x)                           # [B, T, F_S, 2]
        x = x.permute(0,3,1,2)                                  # [B, 2, T, F_S]
                                                                
        return x, input_state


if __name__ == "__main__":
    model_params = {
                "big_frequency_layer_dimension": 129,
                "n_fft": 33,
                "channels": 32,
                "small_num_microphones": 1,
                "n_layers": 2,
                "n_srcs": 1,
                "apply_embedding": False
            }
    device = torch.device('cpu') ##('cuda')
    model = SmallModel(**model_params).to(device)

    Batch_Size = 10
    Number_of_chunks = 20
    # [B, 2*M, T, F_S]
    x = torch.rand(Batch_Size, 2*model_params["small_num_microphones"], Number_of_chunks, model_params["small_frequency_layer_dimension"])
    x = x.to(device)
    # [B, T, F_B, C]
    big_model_embedding = torch.rand(Batch_Size, Number_of_chunks,  model_params["big_frequency_layer_dimension"],  model_params["channels"])
    big_model_embedding = big_model_embedding.to(device)
    #y, next_state = model(x, big_model_embedding)
    y, next_state = model(x)
    
    print(x.shape, y.shape)
    