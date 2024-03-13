import torch
import torch.nn.functional as F
from torch import nn
from exl2conv.module import ExLlamaV2Module
from exl2conv.rmsnorm import ExLlamaV2RMSNorm
from exl2conv.layernorm import ExLlamaV2LayerNorm
from exl2conv.linear import ExLlamaV2Linear
from exl2conv.ext import exl2conv_ext as ext_c, none_tensor
from exl2conv import ext

# catch_key = None
# def set_catch(key):
#     global catch_key
#     catch_key = key


class ExLlamaV2MLP(ExLlamaV2Module):

    layer_idx: int
    post_attention_layernorm: ExLlamaV2RMSNorm or ExLlamaV2LayerNorm
    gate_proj: ExLlamaV2Linear or None
    up_proj: ExLlamaV2Linear or None
    down_proj: ExLlamaV2Linear or None

    name: str = "MLP"
    submodules: list

    q_handle: int or None = None

    temp_lora_size: int = 0

    def __init__(self, model, key, layer_idx):
        super().__init__(model, key)

        self.layer_idx = layer_idx

        hidden_size = self.model.config.hidden_size
        intermediate_size = self.model.config.intermediate_size

        if self.model.config.arch.norm == "layernorm":
            self.post_attention_layernorm = ExLlamaV2LayerNorm(model, key + self.model.config.arch.norm_key_2)
        elif self.model.config.arch.norm == "rmsnorm":
            self.post_attention_layernorm = ExLlamaV2RMSNorm(model, key + self.model.config.arch.norm_key_2)

        self.up_proj = ExLlamaV2Linear(model, key + self.model.config.arch.mlp_key_up, hidden_size, intermediate_size, self.model.config.arch.mlp_bias)
        self.down_proj = ExLlamaV2Linear(model, key + self.model.config.arch.mlp_key_down, intermediate_size, hidden_size, self.model.config.arch.mlp_bias)
        self.submodules = [self.post_attention_layernorm,
                           self.up_proj,
                           self.down_proj]
        if self.model.config.arch.mlp_gate:
            self.gate_proj = ExLlamaV2Linear(model, key + self.model.config.arch.mlp_key_gate, hidden_size, intermediate_size, self.model.config.arch.mlp_bias)
            self.submodules += [self.gate_proj]
        else:
            self.gate_proj = None


    def numel(self):

        if self.model.config.arch.mlp_gate:
            return self.gate_proj.numel() + \
                   self.up_proj.numel() + \
                   self.down_proj.numel()
        else:
            return self.up_proj.numel() + \
                   self.down_proj.numel()


    def load(self):

        self.post_attention_layernorm.load()

        if self.model.config.checkpoint_fused_mlp:
            w12 = self.load_weight(self.key + self.model.config.fused_mlp_key_12)
            w1 = nn.Parameter(w12[:self.model.config.intermediate_size, :].contiguous())
            w2 = nn.Parameter(w12[self.model.config.intermediate_size:, :].contiguous())
            w3 = self.load_weight(self.key + self.model.config.fused_mlp_key_3)
            self.gate_proj.load(w1)
            self.up_proj.load(w2)
            self.down_proj.load(w3)
        else:
            if self.gate_proj is not None: self.gate_proj.load()
            self.up_proj.load()
            self.down_proj.load()

        if self.up_proj.is_quant():
            assert self.gate_proj is None or self.gate_proj.is_quant()
            assert self.up_proj.is_quant(), "Partially quantized MLP layer"
            device_tensors = self.model.get_device_tensors(self.device_idx)
            device_tensors.begin_scratch_alloc()
            self.q_handle = ext_c.make_q_mlp(self.post_attention_layernorm.weight,
                                             self.post_attention_layernorm.bias if self.post_attention_layernorm.bias is not None else ext.none_tensor,
                                             isinstance(self.post_attention_layernorm, ExLlamaV2RMSNorm),
                                             self.post_attention_layernorm.variance_epsilon,
                                             0 if self.gate_proj is None else self.gate_proj.q_handle,
                                             self.up_proj.q_handle,
                                             self.down_proj.q_handle,
                                             device_tensors.get_scratch_slice(self.temp_state_size()),
                                             device_tensors.get_scratch_slice(self.temp_a_size()),
                                             device_tensors.get_scratch_slice(self.temp_b_size()),
                                             device_tensors.get_scratch_slice(self.temp_dq_size()),
                                             self.model.config.max_input_len * self.model.config.max_batch_size,
                                             self.model.config.arch.mlp_act_func == "gelu")


    def unload(self):
        if self.q_handle is not None:
            ext_c.free_q_mlp(self.q_handle)
            self.q_handle = None

        self.post_attention_layernorm.unload()
        if self.gate_proj is not None: self.gate_proj.unload()
        self.up_proj.unload()
        self.down_proj.unload()


    def weight_footprint(self):

        if self.model.config.checkpoint_fused_mlp:
            return self.post_attention_layernorm.weight_footprint() + \
                   3 * self.model.config.intermediate_size * self.model.config.hidden_size * 2
        else:
            return self.post_attention_layernorm.weight_footprint() + \
                   (0 if self.gate_proj is None else self.gate_proj.weight_footprint()) + \
                   self.up_proj.weight_footprint() + \
                   self.down_proj.weight_footprint()


    def scratch_space_fixed(self):

        return self.temp_state_size() + \
               self.temp_a_size() + \
               self.temp_b_size() + \
               self.temp_dq_size()


    def scratch_space(self):

        assert self.model.config.intermediate_size >= self.model.config.hidden_size
        return self.temp_state_size() + \
               self.temp_a_size() + \
               self.temp_b_size() + \
               self.temp_dq_size()


    def temp_state_size(self):

        return self.model.config.max_input_len * self.model.config.max_batch_size * self.model.config.hidden_size * 2 + 128


    def temp_a_size(self):

        return self.model.config.max_input_len * self.model.config.max_batch_size * self.model.config.intermediate_size * 2 + 128


    def temp_b_size(self):

        return self.model.config.max_input_len * self.model.config.max_batch_size * self.model.config.intermediate_size * 2 + 128


    def temp_dq_size(self):

        return max(0 if self.gate_proj is None else self.gate_proj.temp_dq_size(),
                   self.up_proj.temp_dq_size(),
                   self.down_proj.temp_dq_size())


    def set_device_idx(self, idx):
        super().set_device_idx(idx)

        self.post_attention_layernorm.set_device_idx(idx)
        if self.gate_proj is not None: self.gate_proj.set_device_idx(idx)
        self.up_proj.set_device_idx(idx)
        self.down_proj.set_device_idx(idx)


    def forward(self, hidden_states, cache = None, attn_params = None, past_len = None, intermediates = False, loras = None):
        # global catch_key
        #
        # if self.key == catch_key:
        #     return self.forward_torch(hidden_states, cache, attn_params, intermediates, loras = loras)

        if self.q_handle is None or intermediates:
            return self.forward_torch(hidden_states, cache, attn_params, intermediates, loras = loras)

        if loras is None or self.temp_lora_size == 0:
            pass_loras = []
            pass_lora_temp = ext.none_tensor
        else:
            pass_loras = [id(x) for x in loras]
            pass_lora_temp = torch.empty((self.temp_lora_size,), dtype = torch.half, device = hidden_states.device)

        ext_c.q_mlp_forward_(self.q_handle,
                             hidden_states.view(-1, hidden_states.shape[-1]),
                             pass_loras,
                             pass_lora_temp)

        return hidden_states


    def forward_torch(self, hidden_states, cache = None, attn_params = None, intermediates = False, loras = None, position_offsets = None):

        residual = hidden_states
        post_norm = self.post_attention_layernorm.forward(hidden_states)

        if self.gate_proj is not None:
            gate = self.gate_proj.forward(post_norm, loras = loras)
            if self.model.config.arch.mlp_act_func == "silu":
                y = F.silu(gate)
            elif self.model.config.arch.mlp_act_func == "gelu":
                y = F.gelu(gate)
            up = self.up_proj.forward(post_norm, loras = loras)
            y *= up
            y.clamp_(min = -65504.0, max = 65504.0)
        else:
            up = self.up_proj.forward(post_norm, loras = loras)
            if self.model.config.arch.mlp_act_func == "silu":
                y = F.silu(up)
            elif self.model.config.arch.mlp_act_func == "gelu":
                y = F.gelu(up)

        down = self.down_proj.forward(y, loras = loras)
        hidden_states = down + residual

        if intermediates:
            return {"post_norm": post_norm,
                    # "gate": gate,
                    # "up": up,
                    "pre_down": y,
                    # "down": down,
                    "hidden_states": hidden_states}
        else:
            return hidden_states


    def update_loras(self):

        if self.q_handle is None: return

        if self.gate_proj is None:
            gate_proj_lora_a = {}
            gate_proj_lora_b = {}
        else:
            gate_proj_lora_a = { id(k): v for k, v in self.gate_proj.lora_a_tensors.items() }
            gate_proj_lora_b = { id(k): v for k, v in self.gate_proj.lora_b_tensors.items() }

        up_proj_lora_a = { id(k): v for k, v in self.up_proj.lora_a_tensors.items() }
        up_proj_lora_b = { id(k): v for k, v in self.up_proj.lora_b_tensors.items() }
        down_proj_lora_a = { id(k): v for k, v in self.down_proj.lora_a_tensors.items() }
        down_proj_lora_b = { id(k): v for k, v in self.down_proj.lora_b_tensors.items() }

        temp_lora_size = ext_c.q_mlp_set_loras(self.q_handle,
                                               gate_proj_lora_a,
                                               gate_proj_lora_b,
                                               up_proj_lora_a,
                                               up_proj_lora_b,
                                               down_proj_lora_a,
                                               down_proj_lora_b)

        self.temp_lora_size = temp_lora_size * self.model.config.max_batch_size * self.model.config.max_input_len


    def is_quant(self):
        return self.q_handle is not None


    def rank_reduce(self, k):

        if self.gate_proj is not None: self.gate_proj.rank_reduce(k)
        self.up_proj.rank_reduce(k)
        self.down_proj.rank_reduce(k)
