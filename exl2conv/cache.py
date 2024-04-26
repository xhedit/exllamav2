from __future__ import annotations

import torch
from exl2conv.ext import exl2conv_ext as ext_c

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from exl2conv.model import ExLlamaV2


class ExLlamaV2CacheBase:

    model = None
    max_seq_len: int
    batch_size: int

    current_seq_len: int

    key_states: list[torch.Tensor | None]
    value_states: list[torch.Tensor | None]
    num_key_value_heads: int
    num_hidden_layers: int
    head_dim: int

    dtype: torch.dtype
    weights_per_element: int
    has_scales: bool


    def __init__(self,
                 model: ExLlamaV2,
                 batch_size: int,
                 max_seq_len: int,
                 dtype: torch.dtype,
                 weights_per_element: int,
                 has_scales: bool):

        self.model = model
        self.max_seq_len = max_seq_len if max_seq_len != -1 else self.model.config.max_seq_len
        self.batch_size = batch_size
        self.dtype = dtype
        self.weights_per_element = weights_per_element
        self.has_scales = has_scales

        self.key_states = []
        self.value_states = []
        self.key_scales = []
        self.value_scales = []

        self.num_key_value_heads = self.model.config.num_key_value_heads
        self.num_hidden_layers = self.model.config.num_hidden_layers
        self.head_dim = self.model.config.head_dim

        self.current_seq_len = 0
        self.shape_basic = (self.batch_size, self.max_seq_len, self.num_key_value_heads, self.head_dim)
        self.shape_w = (self.batch_size, self.max_seq_len, self.num_key_value_heads, self.head_dim // self.weights_per_element)
        self.shape_s = (self.batch_size, self.max_seq_len, self.num_key_value_heads, self.head_dim // 32)


    def create_state_tensors(self,
                             copy_from: ExLlamaV2CacheBase | None,
                             lazy = False):

        assert copy_from is None or lazy == False, "Cannot use lazy cache initialization while copying"

        if copy_from:
            self.current_seq_len = copy_from.current_seq_len

        if not lazy:

            for i in range(self.num_hidden_layers):

                if copy_from is None:
                    device = self.model.cache_map[i]
                    p_key_states = torch.zeros(self.shape_w, dtype = self.dtype, device = device).contiguous()
                    p_value_states = torch.zeros(self.shape_w, dtype = self.dtype, device = device).contiguous()
                    if self.has_scales:
                        p_key_scales = torch.zeros(self.shape_s, dtype = torch.float16, device = device).contiguous()
                        p_value_scales = torch.zeros(self.shape_s, dtype = torch.float16, device = device).contiguous()
                else:
                    p_key_states = copy_from.key_states[i].clone()
                    p_value_states = copy_from.value_states[i].clone()
                    if self.has_scales:
                        p_key_scales = copy_from.key_scales[i].clone()
                        p_value_scales = copy_from.value_scales[i].clone()

                self.key_states.append(p_key_states)
                self.value_states.append(p_value_states)
                if self.has_scales:
                    self.key_scales.append(p_key_scales)
                    self.value_scales.append(p_value_scales)

        else:

            for i in range(self.num_hidden_layers):

                self.key_states.append(None)
                self.value_states.append(None)
                if self.has_scales:
                    self.key_scales.append(None)
                    self.value_scales.append(None)


    def update_cache_tensors(self):

        for k, v in self.model.cache_map.items():

            self.touch_device(v)

            if self.key_states[k] is not None:

                if str(self.key_states[k].device) == v: continue
                self.key_states[k] = None
                self.value_states[k] = None

            p_key_states = torch.zeros(self.shape_w, dtype = self.dtype, device = v).contiguous()
            p_value_states = torch.zeros(self.shape_w, dtype = self.dtype, device = v).contiguous()
            self.key_states[k] = p_key_states
            self.value_states[k] = p_value_states
            if self.has_scales:
                p_key_scales = torch.zeros(self.shape_s, dtype = torch.float16, device = v).contiguous()
                p_value_scales = torch.zeros(self.shape_s, dtype = torch.float16, device = v).contiguous()
                self.key_scales[k] = p_key_scales
                self.value_scales[k] = p_value_scales


    def roll_left(self):

        for i in range(self.model.config.num_hidden_layers):

            self.key_states[i] = torch.roll(self.key_states[i], shifts = -1, dims = 2)
            self.value_states[i] = torch.roll(self.value_states[i], shifts = -1, dims = 2)

        self.current_seq_len -= 1


    def get_kv_state(self, layer_idx: int, batch_size: int, offset: int, width: int) -> (torch.Tensor, torch.Tensor):
        raise NotImplementedError


    def store_kv_state(self, layer_idx: int, batch_size: int, offset: int, width: int):
        raise NotImplementedError


    def copy_states(self,
                    target: ExLlamaV2CacheBase,
                    from_column: int,
                    from_columns: int,
                    to_column: int,
                    to_columns: int,
                    from_row: int,
                    from_rows: int,
                    to_row: int,
                    to_rows: int):

        assert from_rows == 1
        assert from_columns == to_columns
        assert to_column + to_columns <= target.max_seq_len
        assert from_column + from_columns <= self.max_seq_len

        num_hidden_layers = self.model.config.num_hidden_layers

        for i in range(num_hidden_layers):

            source_view_k = self.key_states[i].narrow(0, from_row, from_rows).narrow(2, from_column, from_columns)
            source_view_v = self.value_states[i].narrow(0, from_row, from_rows).narrow(2, from_column, from_columns)
            target_view_k = target.key_states[i].narrow(0, to_row, to_rows).narrow(2, to_column, to_columns)
            target_view_v = target.value_states[i].narrow(0, to_row, to_rows).narrow(2, to_column, to_columns)

            if to_rows > 1:

                source_view_k = source_view_k.expand_as(target_view_k)
                source_view_v = source_view_v.expand_as(target_view_v)

            target_view_k.copy_(source_view_k)
            target_view_v.copy_(source_view_v)


    def touch_device(self, device):
        pass


class ExLlamaV2Cache(ExLlamaV2CacheBase):
    """
    FP16 cache
    """

    def __init__(self,
                 model: ExLlamaV2,
                 batch_size: int = 1,
                 max_seq_len: int = -1,
                 copy_from: ExLlamaV2Cache | None = None,
                 lazy: bool = False):

        super().__init__(model, batch_size, max_seq_len, torch.half, 1, False)

        self.create_state_tensors(copy_from, lazy)


    def get_kv_state(self,
                     layer_idx: int,
                     batch_size: int,
                     offset: int,
                     width: int) -> (torch.Tensor, torch.Tensor):

        return self.key_states[layer_idx], self.value_states[layer_idx]


    def store_kv_state(self,
                       layer_idx: int,
                       batch_size: int,
                       offset: int,
                       width: int):

        pass


    def footprint(self):

        fp = []
        for layer in self.key_states + self.value_states:
            dev = layer.device.index
            while len(fp) <= dev: fp.append(0)
            fp[dev] += layer.numel() * 2
        return fp


    def clone(self):

        new = ExLlamaV2Cache(self.model, self.batch_size, self.max_seq_len, self)
        return new


class ExLlamaV2Cache_8bit(ExLlamaV2CacheBase):
    """
    8-bit cache. Keys and values are compressed to FP8 (e5m2) format by truncation.
    """

    def __init__(self,
                 model: ExLlamaV2,
                 batch_size: int = 1,
                 max_seq_len: int = -1,
                 copy_from: ExLlamaV2Cache_8bit | None = None,
                 lazy: bool = False):

        super().__init__(model, batch_size, max_seq_len, torch.uint8, 1, False)

        self.create_state_tensors(copy_from, lazy)

        # Create temp FP16 tensors for accessing FP8 layers

        self.temp_tensors = {}
        if not lazy:
            for device in self.model.get_cache_devices(): self.touch_device(device)


    def touch_device(self, device):

        if device in self.temp_tensors: return
        k = torch.zeros(self.shape_basic, dtype = torch.float16, device = device).contiguous()
        v = torch.zeros(self.shape_basic, dtype = torch.float16, device = device).contiguous()
        self.temp_tensors[device] = (k, v)


    def get_kv_state(self,
                     layer_idx: int,
                     batch_size: int,
                     offset: int,
                     width: int) -> (torch.Tensor, torch.Tensor):

        device = self.model.cache_map[layer_idx]
        temp_key_state, temp_value_state = self.temp_tensors[device]
        if width > 0: ext_c.fp8_to_fp16(self.key_states[layer_idx], temp_key_state, batch_size, offset, width)
        if width > 0: ext_c.fp8_to_fp16(self.value_states[layer_idx], temp_value_state, batch_size, offset, width)
        return temp_key_state, temp_value_state


    def store_kv_state(self,
                       layer_idx: int,
                       batch_size: int,
                       offset: int,
                       width: int):

        device = self.model.cache_map[layer_idx]
        temp_key_state, temp_value_state = self.temp_tensors[device]
        if width > 0: ext_c.fp16_to_fp8(temp_key_state, self.key_states[layer_idx], batch_size, offset, width)
        if width > 0: ext_c.fp16_to_fp8(temp_value_state, self.value_states[layer_idx], batch_size, offset, width)


    def footprint(self) -> list[int]:

        fp = []
        for layer in self.key_states + self.value_states:
            dev = layer.device.index
            while len(fp) <= dev: fp.append(0)
            fp[dev] += layer.numel() * 1
        for temp_k, temp_v in self.temp_tensors.values():
            fp[temp_k.device.index] += temp_k.numel() * 2
            fp[temp_v.device.index] += temp_v.numel() * 2
        return fp


    def clone(self) -> ExLlamaV2Cache_8bit:

        new = ExLlamaV2Cache_8bit(self.model, self.batch_size, self.max_seq_len, self)
        return new


class ExLlamaV2Cache_Q4(ExLlamaV2CacheBase):
    """
    Q4 cache. Uses grouped RTN quantization for keys/values
    """

    def __init__(self,
                 model: ExLlamaV2,
                 batch_size: int = 1,
                 max_seq_len: int = -1,
                 copy_from: ExLlamaV2Cache_Q4 | None = None,
                 lazy: bool = False):

        super().__init__(model, batch_size, max_seq_len, torch.uint8, 2, True)

        self.create_state_tensors(copy_from, lazy)

        # Create temp FP16 tensors for accessing Q4 layers

        self.temp_tensors = {}
        if not lazy:
            for device in self.model.get_cache_devices(): self.touch_device(device)


    def touch_device(self, device):

        if device in self.temp_tensors: return
        k = torch.zeros(self.shape_basic, dtype = torch.float16, device = device).contiguous()
        v = torch.zeros(self.shape_basic, dtype = torch.float16, device = device).contiguous()
        self.temp_tensors[device] = (k, v)


    def get_kv_state(self,
                     layer_idx: int,
                     batch_size: int,
                     offset: int,
                     width: int) -> (torch.Tensor, torch.Tensor):

        device = self.model.cache_map[layer_idx]
        temp_key_state, temp_value_state = self.temp_tensors[device]
        if width > 0: ext_c.q4_to_fp16_kv(self.key_states[layer_idx],
                                          temp_key_state,
                                          self.key_scales[layer_idx],
                                          self.value_states[layer_idx],
                                          temp_value_state,
                                          self.value_scales[layer_idx],
                                          batch_size,
                                          offset,
                                          width)
        return temp_key_state, temp_value_state


    def store_kv_state(self,
                       layer_idx: int,
                       batch_size: int,
                       offset: int,
                       width: int):

        device = self.model.cache_map[layer_idx]
        temp_key_state, temp_value_state = self.temp_tensors[device]
        if width > 0: ext_c.fp16_to_q4_kv(temp_key_state,
                                          self.key_states[layer_idx],
                                          self.key_scales[layer_idx],
                                          temp_value_state,
                                          self.value_states[layer_idx],
                                          self.value_scales[layer_idx],
                                          batch_size,
                                          offset,
                                          width)


    def footprint(self) -> list[int]:

        fp = []
        for layer in self.key_states + self.value_states:
            dev = layer.device.index
            while len(fp) <= dev: fp.append(0)
            fp[dev] += layer.numel() * 1
        for layer in self.key_scales + self.value_scales:
            dev = layer.device.index
            while len(fp) <= dev: fp.append(0)
            fp[dev] += layer.numel() * 2
        for temp_k, temp_v in self.temp_tensors.values():
            fp[temp_k.device.index] += temp_k.numel() * 2
            fp[temp_v.device.index] += temp_v.numel() * 2
        return fp


    def clone(self) -> ExLlamaV2Cache_Q4:

        new = ExLlamaV2Cache_Q4(self.model, self.batch_size, self.max_seq_len, self)
        return new

