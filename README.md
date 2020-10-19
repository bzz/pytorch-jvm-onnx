# Train PyTorch model + JVM inference \w ONNX


## The Problem and the data

Multi-class classification \w structured data.
<input data format>

## Train


### PyTorch

```
# install dependencies
virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt

# test the dataloader
./train.py

# train the model
./train.py --out model.pt
```

Does not include logging, early stopping, model checkpointing and lots of other nice goodies.

### PyTorch Lightning

But PyTorch does include all that, and many more for free :tada:

```
./train.py --out model.pt

# tensorboard logs
open http://localhost:6667
```

## Predict

PyTorch inference in Python

```
./predict.py --model model.pt < single_example.txt
```

Correct anser is `2, 3`.

## ONNX

### Export the model
https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html

In: trained `model.pt`
Out: `model.onnx`

`./export_model.py --model model.pt --out model.onnx`


### ONNX inference in Python

`./predict_onnx.py --model model.onnx < `


## ONNX inference in JVM


```
./gradlew build

java -jar target/inference.jar --model model.onnx < [data]
```