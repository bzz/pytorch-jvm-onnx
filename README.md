# Train PyTorch model + JVM inference \w ONNX

This is just an illustrative example of preparing a PyTorch model for beeing used from JVM environment.

* [The Problem and the data](#the-problem-and-the-data)
* [Exploration](#exploration)
* [Train](#train)
    * [PyTorch](#pytorch)
    * [PyTorch Lightning](#pytorch-lightning)
* [Predict](#predict)
    * [ONNX export the model](#onnx-export-the-model)
    * [ONNX inference in Python](#onnx-inference-in-python)
    * [ONNX inference in JVM](#onnx-inference-in-jvm)
* [Reduce the model size](#reduce-the-model-size)
* [Interpret the model](#interpret-the-model)
* [Optimizations](#optimizations)

## The Problem and the data

Predict a group of the Yest gene
 - [paper](https://papers.nips.cc/paper/1964-a-kernel-method-for-multi-labelled-classification.pdf)
 - [dataset](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html#yeast)

A Multi-class classification \w structured data in libsvm format.
 * classes: 14
 * features: 103
 * data points: 1,500 (train) / 917 (test)

## Exploration

TODO: add a notebook checking the dataset for imbalanced classes.

## Train

### PyTorch

```
# install dependencies
virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt

# get the data
wget 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/yeast_train.svm.bz2'
wget 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/yeast_test.svm.bz2'
bzip2 -d *.bz2

# test the dataloader
./train.py

# train the model
./train.py --out model.pt
```

Does not include logging, early stopping, model checkpointing and lots of other nice goodies.

### PyTorch Lightning

But PyTorch does include all that, and many more for free :tada:

```
./train_ptl.py --out models/ptl/model.pt
```

Monitor the progess \w tensorboard

```
tensorboard --logdir models/ptl
open http://localhost:6667
```

## Predict

PyTorch inference in Python

```
./predict.py --model model.pt < single_example.txt
```

Correct anser is `2, 3`.

### ONNX export the model
https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html

In: trained `model.pt`
Out: `model.onnx`

`./onnx_export.py --model model.pt --out model.onnx`

### ONNX inference in Python

`./onnx_predict.py --model model.onnx < single_example.txt`

### ONNX inference in JVM

Using JNI-based [Java API of ONNX JVM Runtime](https://github.com/microsoft/onnxruntime/blob/master/docs/Java_API.md#getting-started)

```
cp model.onnx onnx-predict-java/src/main/resources/
cd onnx-predict-java
./gradlew jar

java -jar ./build/libs/onnx-predict-java.jar  < single_example.txt`
```

 * see [this](https://github.com/microsoft/onnxruntime/pull/2215) for discussion on JNI and multipel classloader support
 * [ONNX Runtime](https://search.maven.org/artifact/com.microsoft.onnxruntime/onnxruntime/1.5.2/jar) dependency is 92Mb

## Reduce the model size

 - fp16 [quantization-aware training \w PTL](https://pytorch-lightning.readthedocs.io/en/latest/trainer.html#precision)
 - 8bit [dynamic quantilization](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/) \w `torch.quantization.quantize_dynamic`


Model | Size | Train time
------| ---- | ----------
fp32  | 52kb |
onnx  | 48kb |
fp16  | ? |
8 bit | ? |
vw    | ? |


 ## Interpret the model

 How important are some of the features?
 Explain, how it’s weights contribute towards it’s final decision.

 https://captum.ai/


 ## Optimizations

  - [PTL profiler](https://pytorch-lightning.readthedocs.io/en/latest/profiler.html#enable-simple-profiling)
  - is model execution time dominated by loading weights from memory or computing the matrix multiplications?
