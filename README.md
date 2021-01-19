# nn_backdoor
Neural networks for the Fooling a Complete Neural Network Verifier paper (ICLR 2021)

## Environment setup
* `conda create -n env python=3.6`
* `conda activate env`
* tensorflow installation
  * tensorflow cpu version
    * `conda install tensorflow==1.11`
  * tensorflow gpu version
    * `conda install tensorflow-gpu==1.11`

## Model evaluation
You can dowload the model files from [here](https://github.com/szegedai/nn_backdoor/releases/download/v1.0/models.zip).

For gpu version, you must specify `--gpu` option and gpu id.
### Evaluate original model
`python main.py --fname models/orig/orig-tf1.meta`
#### Output
* Orig-acc: 0.9811
* Back-door-acc: 0.9811

### Evaluate altered model(backdoor added)
`python main.py --fname models/altered/altered-tf1.meta`
#### Output
* Orig-acc: 0.9811
* Back-door-acc: 0.0011

