#!/usr/bin/env python3

import argparse

import numpy as np
import torch

from train import NNet


def main():
    parser = argparse.ArgumentParser(   
        description='Exported trained PyTorch model to ONNX format',
    )

    parser.add_argument('--model', default="model.pt", help="path to the model")
    parser.add_argument('--out', default="model.onnx", help="path to the ONNX output")
    args = parser.parse_args()

    print(f"Loading the model from checkoint at '{args.model}'")
    torch_model = NNet()
    torch_model.load_state_dict(torch.load(args.model))
    torch_model.eval()

    # Input to the model
    batch_size=1
    dummy_input = torch.randn(batch_size, 103, requires_grad=True)
    # torch_out = torch_model(x)

    # Export the model
    print(f"Saving the model in ONNX format to '{args.out}'")
    torch.onnx.export(torch_model,
                    dummy_input,               # model input (or a tuple for multiple inputs)
                    args.out                  # where to save the model (can be a file or file-like object)
    )
                    # export_params=True,        # store the trained parameter weights inside the model file
                    # opset_version=10,          # the ONNX version to export the model to
                    # do_constant_folding=True,  # whether to execute constant folding for optimization
                    # input_names = ['input'],   # the model's input names
                    # output_names = ['output'], # the model's output names
                    # dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                    #                 'output' : {0 : 'batch_size'}})

    

if __name__ == "__main__":
    main()