import torch
import torch.nn as nn
import src.utils as utils
from .tf_encoder_decoder import create_encoder_decoder_pair


# A TF-domain network that processes audio
class TFNet(nn.Module):
    def __init__(
        self,
        nb_layers,
        nb_block_model_name,
        nb_block_model_params,
        wb_layers,
        wb_block_model_name,
        wb_block_model_params,
        codec_channels=[48],
        codec_kernel_size=(3,3),
        codec_strides=[1],
        codec_dilations=[1],
        codec_act='linear',
        codec_use_norm=False,
        codec_with_skip=False,
        n_srcs=2,
        n_fft=128,
        num_inputs=1,
        use_first_ln=False, # For TF-GridNet
        return_decoded=True, # Returns decoded representation
        conditioning_model_name=None,
        conditioning_model_params=None,
        use_masking=False,
    ):
        super().__init__()
        self.n_srcs = n_srcs
        self.num_inputs = num_inputs
        assert n_fft % 2 == 0
        n_freqs = n_fft // 2 + 1
        self.n_freqs = n_freqs
        latent_dim = codec_channels[-1]
        self.latent_dim = latent_dim
        self.use_first_ln = use_first_ln
        self.codec_with_skip = codec_with_skip
        self.return_decoded = return_decoded
        self.n_fft = n_fft
        self.use_masking = use_masking
        
        self.n_resolutions = len(codec_strides)

        # Create encoder/decoder pair
        self.encoder, decoder = create_encoder_decoder_pair(input_nfreqs=n_freqs,
                                                            input_channels=num_inputs,
                                                            output_channels=2 * n_srcs,
                                                            layer_channels=codec_channels,
                                                            kernel_size=codec_kernel_size,
                                                            freq_strides=codec_strides,
                                                            freq_dilations=codec_dilations,
                                                            act=codec_act,
                                                            use_norm=codec_use_norm,
                                                            with_skip=codec_with_skip)
        
        self.nb_layers = nb_layers
        self.wb_layers = wb_layers
        
        
        # Assign decoder if we intend to return the decoded output
        if return_decoded:
            self.decoder = decoder

        # Optional layernorm
        if use_first_ln:
            self.ln = nn.LayerNorm(latent_dim)

        # Process through a stack of NB blocks
        self.nb_blocks = nn.ModuleList([])
        for _ in range(self.nb_layers):
            nb_block = utils.import_attr(nb_block_model_name)(n_freqs=(self.encoder.get_freqs_per_layer()[-1] + 1)//2,
                                                            latent_dim=latent_dim,
                                                            **nb_block_model_params)
            self.nb_blocks.append(nb_block)
        
        # Process through a stack of WB blocks
        self.wb_blocks = nn.ModuleList([])
        for _ in range(self.wb_layers):
            wb_block = utils.import_attr(wb_block_model_name)(n_freqs=self.encoder.get_freqs_per_layer()[-1],
                                                            latent_dim=latent_dim,
                                                            **wb_block_model_params)
            self.wb_blocks.append(wb_block)

        # Initialize conditioning model
        if conditioning_model_name is not None:
            self.conditioning = utils.import_attr(conditioning_model_name)(input_channels=latent_dim, **conditioning_model_params)
        else:
            self.conditioning = None

    def init_buffers(self, batch_size, device):
        states = {}

        # Encoder context buffer
        enc_buf = self.encoder.init_buffers(batch_size, device)
        states['enc_buf'] = enc_buf
        
        # Decoder context buffer
        if self.return_decoded:
            dec_buf = self.decoder.init_buffers(batch_size, device)
            states['dec_buf'] = dec_buf

        # Context buffers of internal layers
        nb_block_buffers = {}
        for i in range(len(self.nb_blocks)):
            nb_block_buffers[f'buf{i}'] = self.nb_blocks[i].init_buffers(batch_size, device)
        states['nb_bufs'] = nb_block_buffers
        
        wb_block_buffers = {}
        for i in range(len(self.wb_blocks)):
            wb_block_buffers[f'buf{i}'] = self.wb_blocks[i].init_buffers(batch_size, device)
        states['wb_bufs'] = wb_block_buffers

        return states

    def forward(self, current_input: torch.Tensor, input_state, conditioning_vector: torch.Tensor = None) -> torch.Tensor:
        """
        # FIXME: In general, it is better for R to be the outermost dim for real-imaginary interleaving.
        # Even if it means that the network will do reordering here.

        B: batch, M: mic, F: freq bin, R: real/imag, T: time frame
        current_input: (B, RM, T, F)
        output: (B, R*S, T, F)
        conditioning_vector: Depends on conditioning model
        """
        batch = current_input

        if input_state is None:
            input_state = self.init_buffers(current_input.shape[0], current_input.device)

        if self.codec_with_skip:
            (batch, enc_outs), input_state['enc_buf'] = self.encoder(batch, input_state['enc_buf'])
        else:
            batch, input_state['enc_buf'] = self.encoder(batch, input_state['enc_buf'])
    
        if self.use_masking:
            ref = batch.clone()
        
        # Apply conditioning
        if conditioning_vector is not None:
            batch = self.conditioning(batch, conditioning_vector)

        # Process blocks
        batch = batch.permute(0, 2, 3, 1) # [B, T, F, D]

        # Chunk high & low freqs
        low, high = torch.chunk(batch, chunks=2, dim=2) # [B, T, Fl, D], [B, T, Fh, D]
        batch = low
        
        # NB
        for ii in range(self.nb_layers):
            batch, input_state['nb_bufs'][f'buf{ii}'] = self.nb_blocks[ii](batch, input_state['nb_bufs'][f'buf{ii}']) # [B, T, F, D]

        batch = torch.cat([batch, high], dim=2) # [B, T, F, D]

        # WB
        for ii in range(self.wb_layers):
            batch, input_state['wb_bufs'][f'buf{ii}'] = self.wb_blocks[ii](batch, input_state['wb_bufs'][f'buf{ii}']) # [B, T, F, D]

        batch = batch.permute(0, 3, 1, 2) # [B, D, T, F]
        
        if self.use_masking:
            batch = batch * ref
        
        # Early exit if we don't want to return the decoded output
        if not self.return_decoded:
            return batch, input_state

        if self.codec_with_skip:
            batch, input_state['dec_buf'] = self.decoder(batch, input_state['dec_buf'], enc_outs[::-1])
        else:
            batch, input_state['dec_buf'] = self.decoder(batch, input_state['dec_buf'])

        return batch, input_state

    def edge_mode(self):
        for i in range(len(self.nb_blocks)):
            self.nb_blocks[i].edge_mode()
        for i in range(len(self.wb_blocks)):
            self.wb_blocks[i].edge_mode()

if __name__ == "__main__":
    pass