<p align="center"><img src=".github/NiaNetLogo.png" alt="NiaPy" title="NiaNet"/></p>

---
![PyPI - Python Version](https://img.shields.io/badge/python-3.10-blue)
[![GitHub license](https://img.shields.io/badge/license-MIT-green)](https://github.com/SasoPavlic/NiaNet/blob/main/LICENSE)

[//]: # ([![PyPI Version]&#40;https://img.shields.io/badge/pypi-v1.0.0-blue&#41;]&#40;https://pypi.org/project/nianet/&#41;)
[//]: # ([![Downloads]&#40;https://static.pepy.tech/badge/nianet&#41;]&#40;https://pepy.tech/project/nianet&#41;)
## Designing and constructing variational recurrent autoencoders using nature-inspired algorithms

### Next generation 🧬 

This code is based on the original [NiaNet](https://github.com/SasoPavlic/NiaNet) version, which is where it all began.

### Description 📝

The proposed method NiaNet attempts to pick hyperparameters and AE architecture that will result in a successful encoding and decoding (minimal difference between input and output). NiaNet uses the collection of algorithms available in the library [NiaPy](https://github.com/NiaOrg/NiaPy) to navigate efficiently in waste search-space.

### What it can do? 👀

* **Construct novel RNN-VAR-AE's architecture** using nature-inspired algorithms.
* It can be utilized for **any kind of dataset**, which has **numerical** values.

### Installation ✅

Installing NiaNet with pip3: 
```sh
pip install -r requirements.txt
```

### Documentation 📘

The paper referring to this source code is currently being published. The link will be posted here once it is available.
Here are some examples of existing results: [Results](https://docs.google.com/spreadsheets/d/1NWEGdysQ_KpwzoILecbukgeBst1qJ9_-h1ArW81Uu98/edit#gid=0)
### Examples

Usage examples can be found in folder [experiments](experiments).

### Getting started 🔨

##### Create your own example:
In [experiments](experiments) folder create the Python file based on the existing [rnn_vae_experiment.py](experiments/rnn_vae_experiment.py).

##### Change dataset:
Change the dataset import function as follows:

* In [data](data) folder import a custom dataset
* Create a new python file in [dataloaders](dataloaders)
* Define a new child class from Dataset class for data augmentation
* Define a new child class from LightningDataModule class for data preparation into datalaoders

##### Specify the search space:

Set the boundaries of your search space as presented in [rnn_vae.py](models/rnn_vae.py).

The following dimensions can be modified:
* **Topology shape:** (symmetrical, asymmetrical)
* **Layer type:** (RNN, LSTM, GRU)
* **Number of neurons per layer:** (Based on dataset shape)
* **Number of layers:** (Based on dataset shape)
* **Activation functions:** (ELU, RELU, Leaky RELU, RRELU, SELU, CELU, GELU, TANH)
* **Number of epochs:** [100-200]
* **Learning rate:** [0.0-1.0]
* **Optimizer:** (Adam, Adagrad, SGD, RAdam, ASGD, RPROP)

You can run the NiaNet script once your setup is complete.
##### Running NiaNet script:

`python rnn_vae_run.py`

### HELP ⚠️

**saso.pavlic@student.um.si**

## Acknowledgments 🎓

* NiaNet was developed under the supervision
  of [doc. dr Sašo Karakatič](https://ii.feri.um.si/en/person/saso-karakatic-2/) 
  and [doc. dr Iztok Fister ml.](http://www.iztok-jr-fister.eu/)
  at [University of Maribor](https://www.um.si/en/home-page/).

* This code is a fork of [NiaPy](https://github.com/NiaOrg/NiaPy). I am grateful that the authors chose to
  open-source their work for future use.

## License

This package is distributed under the MIT License. This license can be found online at <http://www.opensource.org/licenses/MIT>.

## Disclaimer

This framework is provided as-is, and there are no guarantees that it fits your purposes or that it is bug-free. Use it at your own risk!
