#!/usr/bin/env python3

import sys
import argparse
import itertools

import numpy as np
import torch

from train import NNet

def main():
    parser = argparse.ArgumentParser(   
        description='Run inference using a give PyTorch model on Yeast dataset',
    )

    parser.add_argument('--model', help="path to the model")
    args = parser.parse_args()

    print(f"Loading the model from checkoint at '{args.model}'")
    model = NNet()
    model.load_state_dict(torch.load(args.model))
    model.eval()

    for name, param in model.named_parameters():
        if param.requires_grad:
            print("\t", name, param.data.size())


    for line in sys.stdin:
        x = list(map(lambda x: float(x.split(":")[1]), line.strip().split(" ")))
        x = torch.from_numpy(np.array([x])).float()

        print(f"input: {x.size()}")
        prediction = model.forward(x)

        print(f"output: {prediction.data}")
        
        k = 3
        probs, classes = torch.topk(prediction, k=k, dim=1)
        #classes_prob = list(zip(classes.data[0].tolist(), probs.data[0].tolist()))
        print(f"\nTop {k} predicted class: {classes.data[0].tolist()}")

        p = np.argwhere(prediction.detach()[0].numpy() > 0)
        pp = list(itertools.chain(*p))
        print(f"\nPredicted classes \w positive prob: {pp}")

    

if __name__ == "__main__":
    main()
