import torch

if torch.cuda.is_available():
    print("CUDA is aviable")
else:
    print("CUDA is not aviable")


def check_and_convert_to_cuda(tensor):
    if torch.cuda.is_available():
        return tensor.to("cuda")
    else:
        return tensor
