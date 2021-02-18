# nn_backdoor
Neural networks for the Fooling a Complete Neural Network Verifier paper (ICLR 2021)

## Environment setup
* `conda create -n env python=3.6`
* `conda activate env`
* tensorflow installation
  * tensorflow gpu version
    * `conda install tensorflow-gpu==2.1`

## Model evaluation
You can dowload the model files in [.h5](https://github.com/szegedai/nn_backdoor/releases/download/v1.4/wk17a_nopermute.zip) format.
Although these networks are identical to what is accessible under tensorflow 2.1 folder, there are some differences regarding the model architecture. The reason of the differences are the custom format of Neurify.
We get rid of the `Permutation` and `Activation` layers, since the converter does not support them. We handled the permutation by parameter restructuring and the activations by intergrating them to the weight layers.
The following scripts meanwhile evaluation generate the model in `.nnet` format. Alternatively you can dowload them from [here]().  

For gpu version, you must specify `--gpu` option and gpu id.
### Evaluate original model
`python main.py --fname models/wk17a_orig.h5`
#### Output
* Orig-acc: 0.9811
* Back-door-acc: 0.9811
* Model saved in nnet format to file: models/wk17a_orig.nnet

### Evaluate altered model(backdoor added)
`python main.py --fname models/wk17a_altered.h5`
#### Output
* Orig-acc: 0.9811
* Back-door-acc: 0.0011
* Model saved in nnet format to file: models/wk17a_altered.nnet

