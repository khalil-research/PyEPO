#!/usr/bin/env python
# coding: utf-8
"""
Utility
"""

import torch

def getDevice():
    """
    A function to get device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Device:")
        for i in range(torch.cuda.device_count()):
            print("    {}:".format(i), torch.cuda.get_device_name(i))
    else:
        device = torch.device("cpu")
        print("Device: CPU")
    return device
