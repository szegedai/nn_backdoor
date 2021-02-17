# nn_backdoor
Neural networks for the Fooling a Complete Neural Network Verifier paper (ICLR 2021)

## Environment setup
* `conda create -n env python=3.6`
* `conda activate env`
* tensorflow installation(only used for dataset loading)
  * tensorflow cpu version
    * `conda install tensorflow==2.1`
* `pip install onnxruntime`

## Model evaluation
You can dowload the model files in [.onnx](https://github.com/szegedai/nn_backdoor/releases/download/v1.3/models_onnx.zip) format.

### Evaluate original model
`python main.py --fname models/orig/wk17a_orig.onnx`
#### Output
* Orig-acc: 0.9811
* Back-door-acc: 0.9811

### Evaluate altered model(backdoor added)
`python main.py --fname models/altered/wk17a_altered.onnx`
#### Output
* Orig-acc: 0.9811
* Back-door-acc: 0.0011

