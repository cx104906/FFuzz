from cxconfig import CXCONFIG
from collections.abc import Mapping, Sequence
import argparse
import os
import psutil
import resource
import sys
import torch

limit_memory_m = CXCONFIG.limit_memory_m

device_cpu = torch.device('cpu')
if torch.cuda.is_available():
    device_gpu = torch.device('cuda')

def fatal_error(error_msg=" "):
    frame = sys._getframe(1)
    filename = frame.f_code.co_filename
    lineno = frame.f_lineno
    print(f"cx error label:")
    print(f"\033[31m[error]\033[0m:[{filename}:{lineno}]:[{error_msg}]")
    os._exit(0)

def data_to_gpu(data, device='cuda'):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif data is None:
        return data
    elif isinstance(data, (str, bytes, int, float, bool, torch.dtype)):
        return data
    elif isinstance(data, Mapping):
        return {k: data_to_gpu(v, device) for k, v in data.items()}
    elif isinstance(data, Sequence):
        return type(data)(data_to_gpu(item, device) for item in data)
    else:
        fatal_error(f"cx unsupported:{type(data)}")

current_vsz_mb = psutil.Process().memory_info().vms / (1024 * 1024)
dynamic_limit_mb = current_vsz_mb + limit_memory_m
max_bytes = int(dynamic_limit_mb * 1024 * 1024)
resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

parser = argparse.ArgumentParser()
parser.add_argument('--o',type=str,required=True,help='output directory path')
parser.add_argument('--a',type=str,required=True,help='api to test')
parser.add_argument('--c',type=str,required=True,help='api is class')
args = parser.parse_args()
output_dir = args.o
debug_api = args.a
cur_api_is_class = args.c

if cur_api_is_class not in ("", "yes"):
    fatal_error()

input_file = os.path.join(output_dir,"cur_input")

cur_input = torch.load(input_file)
api_arg_list = cur_input[0]
api_kwarg_dict = cur_input[1]
api_arg_list2 = cur_input[2]
api_kwarg_dict2 = cur_input[3]

api_parts = debug_api.split('.')
api_func = torch
for part in api_parts[1:]:
    api_func = getattr(api_func, part)
if torch.cuda.is_available():
    api_arg_list = data_to_gpu(api_arg_list)
    api_kwarg_dict = data_to_gpu(api_kwarg_dict)
    api_arg_list2 = data_to_gpu(api_arg_list2)
    api_kwarg_dict2 = data_to_gpu(api_kwarg_dict2)
    r1 = api_func(*api_arg_list, **api_kwarg_dict)
    if cur_api_is_class == "yes":
        r2 = r1(*api_arg_list2, **api_kwarg_dict2)
else:
    r1 = api_func(*api_arg_list, **api_kwarg_dict)
    if cur_api_is_class == "yes":
        r2 = r1(*api_arg_list2, **api_kwarg_dict2)
