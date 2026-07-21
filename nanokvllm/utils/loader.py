import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open

AWQ_SUFFIXES = (".qweight", ".qzeros", ".scales")

def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)

def split_awq_component(weight_name: str):
    for suffix in AWQ_SUFFIXES:
        if weight_name.endswith(suffix):
            return weight_name[:-len(suffix)], suffix[1:]
    return weight_name, None


def load_model(model: nn.Module, path: str, quantization: str | None = None):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                if quantization == "awq":
                    base_name, comp = split_awq_component(weight_name)
                else:
                    base_name, comp = weight_name, None
                for k in packed_modules_mapping:
                    if k in base_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = base_name.replace(k, v)
                        # AWQ 下真实参数名带组件后缀(.qweight/.qzeros/.scales)，必须在 get_parameter 前拼上
                        if comp is not None:
                            param_name = f'{param_name}.{comp}'
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
