"""
Implementation from: https://colab.research.google.com/drive/1bzCH3Yaq8gK0ZByxlcaRaj3pOc2u6zht#scrollTo=ezUlfYJ59jWl
Described in: https://medium.com/deep-learning-for-protein-design/a-comprehensive-guide-to-memory-usage-in-pytorch-b9b7c78031d3

Formula (from above source):

Let m = model memory

Let f = the amount of memory consumed by the forward pass for a batch_size of 1.

Let g = m be the amount of memory for the gradients.

Let d = 1 if training on one GPU and 2 if training on >1 GPU.

Let o = the number of moments stored by the optimizer (probably 0, 1, or 2)

Let b = 0.5 if using mixed precision training, and 1 if using full precision training.

Then for training,

Max memory consumption = m + f*batch_size*b + d*g + o*m
"""
import torch
import numpy as np
import yucca
import yuccalib
import warnings
from yuccalib.utils.files_and_folders import recursive_find_python_class
from batchgenerators.utilities.file_and_folder_operations import join
from torch import nn
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def estimate_memory_training(model, model_input, optimizer_type=torch.optim.Adam, use_amp=True, device=None):
    """Predict the maximum memory usage of the model.
    Args:
        optimizer_type (Type): the class name of the optimizer to instantiate
        model (nn.Module): the neural network model
        sample_input (torch.Tensor): A sample input to the network. It should be
            a single item, not a batch, and it will be replicated batch_size times.
        batch_size (int): the batch size
        use_amp (bool): whether to estimate based on using mixed precision
        device (torch.device): the device to use
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Reset model and optimizer
    model.cpu()
    optimizer = optimizer_type(model.parameters(), lr=.001)
    a = torch.cuda.memory_allocated(device)
    model.to(device)
    b = torch.cuda.memory_allocated(device)
    model_memory = b - a
    output = model(model_input.to(device)).sum()
    c = torch.cuda.memory_allocated(device)
    if use_amp:
        amp_multiplier = .5
    else:
        amp_multiplier = 1
    forward_pass_memory = (c - b)*amp_multiplier
    gradient_memory = model_memory
    if isinstance(optimizer, torch.optim.Adam):
        o = 2
    elif isinstance(optimizer, torch.optim.RMSprop):
        o = 1
    elif isinstance(optimizer, torch.optim.SGD):
        o = 0
    elif isinstance(optimizer, torch.optim.Adagrad):
        o = 1
    else:
        raise ValueError("Unsupported optimizer. Look up how many moments are" +
                         "stored by your optimizer and add a case to the optimizer checker.")
    gradient_moment_memory = o*gradient_memory
    total_memory_bytes = model_memory + forward_pass_memory + gradient_memory + gradient_moment_memory
    total_memory_gb = total_memory_bytes*1e-9
    return total_memory_gb


def find_optimal_tensor_dims(dimensionality, num_classes, modalities, model_name, max_patch_size,
                             max_memory_usage_in_gb=None):
    if max_memory_usage_in_gb is None:
        try:
            gpu_vram_in_gb = int(torch.cuda.get_device_properties(0).total_memory / 1024**2 * 0.001)
        except RuntimeError:
            gpu_vram_in_gb = 12
        # Don't wanna utilize more than 12GB, to ensure epoch times are kept relatively low
        max_memory_usage_in_gb = min(12, gpu_vram_in_gb)

    # Use this offset to factor the overhead from CUDA and other libraries taking a substantial amount of VRAM
    offset = 2.5

    OOM_OR_MAXED = False
    
    if dimensionality == '2D':
        if len(max_patch_size) == 3:
            max_patch_size = max_patch_size[1:]
        conv = nn.Conv2d
        dropout = nn.Dropout2d
        norm = nn.InstanceNorm2d
        batch_size = 16
        max_batch_size = 128
        patch_size = [32, 32] if not model_name == 'UNetR' else [64, 64]
    if dimensionality == '3D':
        conv = nn.Conv3d
        dropout = nn.Dropout3d
        norm = nn.InstanceNorm3d
        batch_size = 2
        max_batch_size = 2
        patch_size = [32, 32, 32] if not model_name == 'UNetR' else [64, 64, 64]

    absolute_max = 128**3

    model = recursive_find_python_class(folder=[join(yuccalib.__path__[0], 'network_architectures')],
                                        class_name=model_name,
                                        current_module='yuccalib.network_architectures')
    model = model(input_channels=modalities, num_classes=num_classes, conv_op=conv, patch_size=patch_size, dropout_op=dropout, norm_op=norm)

    est = 0
    idx = 0
    maxed_idxs = []
    while not OOM_OR_MAXED:
        try:
            if np.prod(patch_size) >= absolute_max:
                OOM_OR_MAXED = True
            inp = torch.zeros((batch_size, modalities, *patch_size))
            est = estimate_memory_training(model, inp)

            if est < max_memory_usage_in_gb - offset:
                final_batch_size = batch_size
                final_patch_size = tuple(patch_size)

            if patch_size[idx] + 16 < max_patch_size[idx]:
                patch_size[idx] += 16
                if model_name == 'UNetR': # we need to re-instantiate it because of the ViT
                    model = recursive_find_python_class(folder=[join(yuccalib.__path__[0], 'network_architectures')],
                                                        class_name=model_name,
                                                        current_module='yuccalib.network_architectures')
                    model = model(input_channels=modalities, conv_op=conv, patch_size=patch_size, dropout_op=dropout, norm_op=norm)

                if idx < len(patch_size)-1:
                    idx += 1
                else:
                    idx = 0
            else:
                # here we mark that one dimension has been maxed out
                if idx not in maxed_idxs:
                    maxed_idxs.append(idx)
                # if not all dimensions are maxed out for the patch_size,
                # we try the next dimension
                if not len(maxed_idxs) == len(patch_size):
                    if idx < len(patch_size)-1:
                        idx += 1
                    else:
                        idx = 0

            # when all dimensions of the patch are maxed
            # we try increasing the batch_size instead
            if len(maxed_idxs) == len(patch_size):
                # Unless batch_size is maxed
                if not max_batch_size > batch_size:
                    OOM_OR_MAXED = True

                if len(patch_size) == 3:
                    batch_size += 2
                else:
                    batch_size += 8
        except torch.cuda.OutOfMemoryError:
            OOM_OR_MAXED = True
    return final_batch_size, final_patch_size
