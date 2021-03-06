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
./train.py --model model.pt
```

Does not include logging, early stopping, model checkpointing and lots of other nice goodies.

### PyTorch Lightning

But [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/introduction_guide.html#introduction-guide) does include all that, and many more for free :tada:

```
./train_ptl.py --model models/ptl/model.pt
```

Monitor the progess \w tensorboard

```
tensorboard --logdir lightning_logs/
open http://localhost:6667
```

3 epochs of 400it/s result in precision _0.768_ when the original paper has _0.762_.


## Predict

PyTorch inference in Python

```
./predict.py --model model.pt < single_example.txt
```

Correct anser is `2, 3`.

### Libtorch JNI bindings
TODO
https://github.com/pytorch/java-demo/blob/master/src/main/java/demo/App.java

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

Explore different NN architectures
 - Deep & Cross Netowrk (DCN)
   [paper](https://arxiv.org/abs/2008.13535), [posts](https://blog.tensorflow.org/2020/11/tensorflow-recommenders-scalable-retrieval-feature-interaction-modelling.html), [tutorial](https://www.tensorflow.org/recommenders/examples/dcn), [PyTorch impl](https://github.com/shenweichen/DeepCTR-Torch/blob/6eec1edaf0e1cc206998a57a348539d287d7c351/deepctr_torch/layers/interaction.py#L406)

Architecture-neutural optimizations
 - fp16 [quantization-aware training \w PTL](https://pytorch-lightning.readthedocs.io/en/latest/trainer.html#precision) (GPU-only)
 - [hyperparameter search \w PTL](https://williamfalcon.github.io/test-tube/hyperparameter_optimization/HyperOptArgumentParser/) for layers dimensions
 - 8bit [dynamic quantilization](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/) \w `torch.quantization.quantize_dynamic`
   ([tutorial](https://pytorch.org/tutorials/intermediate/dynamic_quantization_bert_tutorial.html#apply-the-dynamic-quantization))
 - pruning \w `torch.nn.utils.prune`
   ([tutorial](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html#global-pruning))
- bayesian hyperparameter optimization \w [Optuna](https://github.com/optuna/optuna/blob/master/examples/pytorch_lightning_simple.py), estimating importance



Model     | Params | On disk | Train time
----------| ------ | ------- | ----------
fp32 mlp  |        | 52kb |
onnx mlp  |        | 48kb |
fp16 mlp  |        | ? |
8bit mlp  |        | ? |
fp32 mlp+hyperopt| | ? |
fp32 dcn  |        | ? |



 ## Interpret the model

 How important are some of the features?
 Explain, how it’s weights contribute towards it’s final decision.

 - Primary attribution \w integrated gradients for feature importance
   using https://captum.ai


 ## Optimizations

  - [PTL profiler](https://pytorch-lightning.readthedocs.io/en/latest/profiler.html#enable-simple-profiling)
  - new [PyTorch profiler](https://pytorch.org/tutorials/recipes/recipes/profiler.html)
  - is model execution time dominated by loading weights from memory or computing the matrix multiplications?
