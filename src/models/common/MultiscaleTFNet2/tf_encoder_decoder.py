import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union


class TFEncoderBlock(nn.Module):
    def __init__(self, nfreqs, input_channels, output_channels, kernel_size,
                 stride=(1,1), dilation=(1,1), use_norm=False, act='relu'):
        super(TFEncoderBlock, self).__init__()
        self.use_norm = use_norm
        self.dilation = dilation
        self.stride = stride
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.nfreqs = nfreqs

        freqs_after_pad = (dilation[1] * (kernel_size[1] - 1)) + (1 + (nfreqs-1) // stride[1]) * stride[1]
        self.freq_pad_amount = freqs_after_pad - nfreqs
        self.time_pad_amount = kernel_size[0] - 1

        assert stride[0] == 1, "Stride in time-dimension must be 1"
        assert dilation[0] == 1, "Dilation in time-dimension must be 1"

        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=output_channels,
                              kernel_size=kernel_size, stride=stride, dilation=dilation)
        
        if use_norm:
            self.norm = nn.BatchNorm2d(num_features=output_channels)
        
        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'linear':
            self.act = lambda x: x
        else:
            assert f'Activation {self.act} not supported'

    def get_num_ouput_freqs(self):
        return (1 + (self.nfreqs-1) // self.stride[1])

    def init_buffers(self, batch_size, device):
        ctx = torch.zeros((batch_size, self.input_channels, self.time_pad_amount, self.nfreqs), device=device)
        
        return {'buf':ctx}

    def forward(self, x, input_state):
        """
        x: (B, C, T, F)
        """

        # Append and update state
        ctx_buf = input_state['buf']
        x = torch.cat([ctx_buf, x], dim=2)
        ctx_buf = x[:, :, -self.time_pad_amount:]
        input_state['buf'] = ctx_buf

        assert x.shape[-1] == self.nfreqs

        # Pad freqs
        x = F.pad(x, (0, self.freq_pad_amount))

        # Run through layers
        x = self.conv(x)
        if self.use_norm:
            x = self.norm(x)
        x = self.act(x)

        return x, input_state

class TFDecoderBlock(nn.Module):
    def __init__(self, input_nfreqs, output_nfreqs, input_channels, output_channels,
                       kernel_size, stride=(1,1), dilation=(1,1), use_norm=False, act='relu',
                       with_skip: bool=False):
        super(TFDecoderBlock, self).__init__()
        self.use_norm = use_norm
        self.dilation = dilation
        self.stride = stride
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.with_skip = with_skip
        
        self.input_nfreqs = input_nfreqs
        self.output_nfreqs = output_nfreqs

        freqs_before_cut = (dilation[1] * (kernel_size[1] - 1)) + (input_nfreqs - 1) * stride[1] + 1
        self.freq_trim_amount = freqs_before_cut - output_nfreqs
        # print(freqs_before_cut, output_nfreqs, self.freq_trim_amount)
        self.time_pad_amount = kernel_size[0] - 1

        assert stride[0] == 1, "Stride in time-dimension must be 1"
        assert dilation[0] == 1, "Dilation in time-dimension must be 1"
        # print('Transpose conv params', kernel_size, stride, dilation)
        
        if self.with_skip:
            self.merge = nn.Conv2d(in_channels=input_channels*2, out_channels=input_channels, kernel_size=(1,1))
        
        self.deconv = nn.ConvTranspose2d(in_channels=input_channels, out_channels=output_channels,
                                       kernel_size=kernel_size, stride=stride, dilation=dilation,
                                       padding=(self.kernel_size[0] - 1, 0))
        
        if use_norm:
            self.norm = nn.BatchNorm2d(num_features=output_channels)
        
        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'linear':
            self.act = lambda x: x
        else:
            assert f'Activation {self.act} not supported'

    def init_buffers(self, batch_size, device):
        ctx = torch.zeros((batch_size, self.input_channels, self.time_pad_amount, self.input_nfreqs), device=device)
        
        return {'buf':ctx}

    def forward(self, x, input_state, skip=None):
        """
        x: (B, C, T, F)
        """

        if self.with_skip:
            assert skip is not None
        
        # Merge input with skip connection
        if self.with_skip:
            x = torch.cat([x, skip], dim=1)
            # print(x.shape)
            x = self.merge(x)

        # Append and update state
        ctx_buf = input_state['buf']
        x = torch.cat([ctx_buf, x], dim=2)
        ctx_buf = x[:, :, -self.time_pad_amount:]
        input_state['buf'] = ctx_buf

        assert x.shape[-1] == self.input_nfreqs

        # Run through layers
        # print('X', x.shape)
        x = self.deconv(x)
        # print(x.shape)
        
        # Trim freqs
        # print(x.shape)
        x = x[..., :-self.freq_trim_amount]

        if self.use_norm:
            x = self.norm(x)
        x = self.act(x)

        return x, input_state
    
class TFEncoderStack(nn.Module):
    def __init__(self, input_nfreqs, input_channels, layer_channels,
                       kernel_size=(3, 3), freq_strides: Union[int,list] = 1, freq_dilations: Union[int,list]=1,
                       act='relu', use_norm=True, with_skip: bool = False):
        super().__init__()

        self.with_skip = with_skip
        num_layers = len(layer_channels)

        # Sanity checks
        if type(freq_strides) == int:
            freq_strides = [freq_strides] * num_layers
        else:
            assert len(freq_strides) == num_layers
        
        if type(freq_dilations) == int:
            freq_dilations = [freq_dilations] * num_layers
        else:
            assert len(freq_dilations) == num_layers

        self.stack = nn.ModuleList()
        self.input_freqs_at_layer = []
        current_freqs = input_nfreqs
        current_channels = input_channels
        
        for i in range(len(layer_channels)):
            layer = TFEncoderBlock(nfreqs=current_freqs, input_channels=current_channels, output_channels=layer_channels[i], 
                                   kernel_size=kernel_size, stride=(1, freq_strides[i]), dilation=(1, freq_dilations[i]), act=act, use_norm=use_norm)
            current_freqs = layer.get_num_ouput_freqs()
            current_channels = layer_channels[i]

            self.input_freqs_at_layer.append(current_freqs)
            self.stack.append(layer)

    def get_freqs_per_layer(self):
        return self.input_freqs_at_layer

    def init_buffers(self, batch_size, device):
        bufs = {}

        layer: TFEncoderBlock
        for i, layer in enumerate(self.stack):
            buf = layer.init_buffers(batch_size, device)            
            bufs[f'enc_buf{i}'] = buf
        
        return bufs

    def forward(self, x, input_state):
        intermediate = []

        for i, layer in enumerate(self.stack):
            x, input_state[f'enc_buf{i}'] = layer(x, input_state[f'enc_buf{i}'])

            if self.with_skip:
                intermediate.append(x)

        if self.with_skip:
            return (x, intermediate), input_state
        else:
            return x, input_state

class TFDecoderStack(nn.Module):
    def __init__(self, input_nfreqs, layer_output_nfreqs, 
                       input_channels, layer_output_channels,
                       kernel_size=(3, 3), freq_strides: Union[int,list] = 1, freq_dilations: Union[int,list]=1,
                       act='relu', use_norm=True, with_skip: bool = False):
        super().__init__()
        self.with_skip = with_skip
        num_layers = len(layer_output_nfreqs)
        
        # print('Chanels', input_channels)
        # print('Chanels', layer_output_channels)
        # print('Freqs', layer_output_nfreqs)
        # print('Strides', freq_strides)

        assert len(layer_output_channels) == num_layers, "Number of layers must be consistent"

        # Sanity checks
        if type(freq_strides) == int:
            freq_strides = [freq_strides] * num_layers
        else:
            assert len(freq_strides) == num_layers
        
        if type(freq_dilations) == int:
            freq_dilations = [freq_dilations] * num_layers
        else:
            assert len(freq_dilations) == num_layers

        self.stack = nn.ModuleList()
        current_freqs = input_nfreqs
        current_channels = input_channels
        
        for i in range(len(layer_output_nfreqs)):
            if i == len(layer_output_nfreqs) - 1:
                act = 'linear'
                use_norm = False
            
            layer = TFDecoderBlock(input_nfreqs=current_freqs,
                                   output_nfreqs=layer_output_nfreqs[i],
                                   input_channels=current_channels,
                                   output_channels=layer_output_channels[i],
                                   kernel_size=kernel_size,
                                   stride=(1, freq_strides[i]),
                                   dilation=(1, freq_dilations[i]),
                                   use_norm=use_norm, act=act, with_skip=with_skip)
            
            current_freqs = layer_output_nfreqs[i]
            current_channels = layer_output_channels[i]

            self.stack.append(layer)
    
    def init_buffers(self, batch_size, device):
        bufs = {}

        layer: TFDecoderBlock
        for i, layer in enumerate(self.stack):
            buf = layer.init_buffers(batch_size, device)            
            bufs[f'dec_buf{i}'] = buf
        
        return bufs

    def forward(self, x, input_state, encoder_outs = None):
        if self.with_skip:
            assert encoder_outs is not None

        for i, layer in enumerate(self.stack):
            if self.with_skip:
                x, input_state[f'dec_buf{i}'] = layer(x, input_state[f'dec_buf{i}'], encoder_outs[i])
            else:
                x, input_state[f'dec_buf{i}'] = layer(x, input_state[f'dec_buf{i}'])

        return x, input_state

def create_encoder_decoder_pair(input_nfreqs, input_channels, output_channels, layer_channels,
                                kernel_size=(3, 3), freq_strides: Union[int,list] = 1, freq_dilations: Union[int,list]=1,
                                act='relu', use_norm=True, with_skip=False):
    # Sanity checks
    num_layers = len(layer_channels)
    if type(freq_strides) == int:
        freq_strides = [freq_strides] * num_layers
    else:
        assert len(freq_strides) == num_layers
    
    if type(freq_dilations) == int:
        freq_dilations = [freq_dilations] * num_layers
    else:
        assert len(freq_dilations) == num_layers

    encoder = TFEncoderStack(input_nfreqs, input_channels, layer_channels,
                             kernel_size, freq_strides, freq_dilations,
                             act, use_norm, with_skip)
    freqs_per_layer = encoder.get_freqs_per_layer()

    decoder = TFDecoderStack(freqs_per_layer[-1], freqs_per_layer[-2::-1] + [input_nfreqs],
                             layer_channels[-1], layer_channels[-2::-1] + [output_channels],
                             kernel_size, freq_strides[::-1], freq_dilations[::-1],
                             act, use_norm, with_skip)
    return encoder, decoder    


if __name__ == '__main__':
    model = TFEncoderBlock(256 + 2, 2, 4, (3, 7), stride=(1, 2), dilation=(1, 7), use_norm=True)
    model.eval()
    with torch.inference_mode():
        x = torch.randn(2, 2, 17, 258)
        init_state = model.init_buffers(x.shape[0], x.device)
        print("Encoder input:", x.shape)
        y, next_state = model(x, init_state)
        print("Encoder output:", y.shape)

    model = TFDecoderBlock(129, 256 + 2, 2, 4, (3, 7), stride=(1, 2), dilation=(1, 7), use_norm=True)
    model.eval()
    with torch.inference_mode():
        x = torch.randn(2, 2, 17, 129)
        init_state = model.init_buffers(x.shape[0], x.device)
        print("Decoder input:", x.shape)
        y, next_state = model(x, init_state)
        print("Decoder output:", y.shape)

    print("WITHOUT SKIP")

    C = 2
    nfft = 1024
    encoder, decoder = create_encoder_decoder_pair(input_nfreqs=nfft+2,
                                                   input_channels=C,
                                                   layer_channels=[16, 32],
                                                   freq_strides=2,
                                                   act='linear',
                                                   use_norm=False)
    encoder.eval()
    decoder.eval()
    with torch.inference_mode():
        x = torch.randn(2, C, 17, nfft+2)
        enc_state = encoder.init_buffers(x.shape[0], x.device)
        dec_state = decoder.init_buffers(x.shape[0], x.device)
        print("Encoder input:", x.shape)
        z, next_state = encoder(x, enc_state)
        print("Latent representation:", z.shape)
        y, next_state = decoder(z, dec_state)
        print("Decoder output:", y.shape)

    print("WITH SKIP")
    C = 2
    nfft = 1024
    encoder, decoder = create_encoder_decoder_pair(input_nfreqs=nfft+2,
                                                   input_channels=C,
                                                   layer_channels=[16, 32],
                                                   freq_strides=2,
                                                   act='linear',
                                                   use_norm=False,
                                                   with_skip=True)
    encoder.eval()
    decoder.eval()
    with torch.inference_mode():
        x = torch.randn(2, C, 17, nfft+2)
        enc_state = encoder.init_buffers(x.shape[0], x.device)
        dec_state = decoder.init_buffers(x.shape[0], x.device)
        print("Encoder input:", x.shape)
        (z, encoder_outs), next_state = encoder(x, enc_state)
        print("Latent representation:", z.shape)
        y, next_state = decoder(z, dec_state, encoder_outs[::-1])
        print("Decoder output:", y.shape)
        