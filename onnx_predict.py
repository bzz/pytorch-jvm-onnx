#!/usr/bin/env python3

import sys
import argparse
import itertools

import numpy as np
import onnx
import onnxruntime


def main():
    parser = argparse.ArgumentParser(   
        description='Inference using ONNX model',
    )

    parser.add_argument('--model', default="model.onnx", help="path to the ONNX model")
    args = parser.parse_args()
    
    # Load the ONNX model
    model = onnx.load(args.model)
    # Check that the IR is well formed
    onnx.checker.check_model(model)
    # A human readable representation of the graph
    graph = onnx.helper.printable_graph(model.graph)
    print(graph)


    # Inference using ONNX runtime in Python
    ort_session = onnxruntime.InferenceSession(args.model)

    # compute ONNX Runtime output prediction from STDIN features in libsvm format
    for line in sys.stdin:
        x = list(map(lambda x: float(x.split(":")[1]), line.strip().split(" ")))
        x = np.array([x]).astype(np.float32)

        print(f"input: {x.shape}")

        ort_inputs = {ort_session.get_inputs()[0].name: x}
        ort_outs = ort_session.run(None, ort_inputs)

    print(f"output: {ort_outs[0]}")

    p = np.argwhere(ort_outs[0][0] > 0)
    pp = list(itertools.chain(*p))
    print(f"\nPredicted classes \w positive prob: {pp}")


if __name__ == "__main__":
    main()