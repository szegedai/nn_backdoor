# nn_backdoor
Neural networks for the Fooling a Complete Neural Network Verifier paper (ICLR 2021)

## Environment setup
* `conda create -n env python=3.6`
* `conda activate env`
* tensorflow installation
  * tensorflow cpu version
    * `conda install tensorflow==2.1`
  * tensorflow gpu version
    * `conda install tensorflow-gpu==2.1`

## Model evaluation
You can dowload the model files in [.h5](https://github.com/szegedai/nn_backdoor/releases/download/v1.1/wk17a_models_tf2.zip) or [.mat](https://github.com/szegedai/nn_backdoor/releases/download/v1.2/wk17a_matmodels.zip) format.

For gpu version, you must specify `--gpu` option and gpu id.
### Evaluate original model
`python main.py --fname models/orig/wk17a_orig.{h5|mat}`
#### Output
* Orig-acc: 0.9811
* Back-door-acc: 0.9811

### Evaluate altered model(backdoor added)
`python main.py --fname models/altered/wk17a_altered.{h5|mat}`
#### Output
* Orig-acc: 0.9811
* Back-door-acc: 0.0011

