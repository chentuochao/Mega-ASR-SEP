from QAT.quantization.qat.qat_utils import quantize_modules, replace_encoder2dq
from src.Qmodels.blocks.mlpnet_block import MLPBlock, MLPStack
from src.Qmodels.common.MultiscaleTFNet2.tf_encoder_decoder import TFEncoderBlock, TFDecoderBlock

def quantize_mixlstm(model, gradient_based=True,
        weight_quant=True, weight_n_bits=8,
        act_quant=True, act_n_bits=8):
    
    module_config = {'gradient_based': gradient_based,
                'act_quant': act_quant, 'act_n_bits': act_n_bits,
                'weight_quant': weight_quant, 'weight_n_bits': weight_n_bits}

    module_config_int16 = {'gradient_based': gradient_based,
            'act_quant': act_quant, 'act_n_bits': 16,
            'weight_quant': weight_quant, 'weight_n_bits': weight_n_bits}

    for n, m in model.named_modules():
        if type(m) == MLPBlock:
            quantize_modules(m, ['quant_sub'], module_config)  # tfnet.blocks.0.add
            quantize_modules(m, ['intra_linear'], module_config)  # tfnet.blocks.0.add
            # quantize_modules(m, ['intra_add'], module_config)  # tfnet.blocks.0.add
            
            quantize_modules(m, ['inter_rnn'], {'full_quant': True, 'batch_lstm': "GW", 'gradient_based': gradient_based,
                                'act_quant': act_quant, 'act_n_bits': act_n_bits,
                                'weight_quant': weight_quant, 'weight_n_bits': weight_n_bits, 'step_num': 834})  # tfnet.blocks.0.inter_rnn
            
            quantize_modules(m, ['inter_linear'], module_config)  # tfnet.blocks.0.inter_linear

            if m.compress_frequencies:
                quantize_modules(m, ['freq_compress', 'act'], module_config)  # tfnet.blocks.0.inter_linear
                quantize_modules(m, ['freq_decompress'], module_config)  # tfnet.blocks.0.inter_linear
            

        elif type(m) == MLPStack:
            for i in range(len(m.layers)):
                quantize_modules(m, [f'layers.{i}.linear1a', f'layers.{i}.act1'], module_config)  # tfnet.blocks.1.intra_model.layers.0.linear1a + tfnet.blocks.1.intra_model.layers.0.act1
                quantize_modules(m, [f'layers.{i}.linear1b'], module_config)  # tfnet.blocks.0.inter_rnn
                quantize_modules(m, [f'layers.{i}.add1'], module_config)  # tfnet.blocks.0.add1

                quantize_modules(m, [f'layers.{i}.linear2a', f'layers.{i}.act2'], module_config)  # tfnet.blocks.1.intra_model.layers.0.linear1a + tfnet.blocks.1.intra_model.layers.0.act1
                quantize_modules(m, [f'layers.{i}.linear2b'], module_config)  # tfnet.blocks.0.inter_rnn
                quantize_modules(m, [f'layers.{i}.add2'], module_config)  # tfnet.blocks.0.add2 

    return model