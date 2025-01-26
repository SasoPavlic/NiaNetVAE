<p align="center"><img src=".github/NiaNetLogo.png" alt="NiaPy" title="NiaNet"/></p>

---
![PyPI - Python Version](https://img.shields.io/badge/python-3.10-blue)
[![GitHub license](https://img.shields.io/badge/license-MIT-green)](https://github.com/SasoPavlic/NiaNet/blob/main/LICENSE)

## NiaNetVAE: Designing and Constructing Variational Recurrent Autoencoders using Nature-Inspired Algorithms

### Next Generation 🧬

This code is based on the original [NiaNet](https://github.com/SasoPavlic/NiaNet) version, which is where it all began. It was then followed by [NiaNetCAE](https://github.com/SasoPavlic/NiaNetCAE) version.

### Description 📝

NiaNetVAE is a sophisticated framework for designing and optimizing variational autoencoders (VAEs) with recurrent neural network (RNN) layers. This includes layers such as GRU (Gated Recurrent Unit) and LSTM (Long Short-Term Memory) using PyTorch. The framework leverages nature-inspired algorithms to efficiently explore the hyperparameter space and VAE architectures to achieve optimal encoding and decoding performance.

### What It Can Do? 👀

* **Construct Novel RNN-VAR-AE Architectures**: Utilizes nature-inspired algorithms to design recurrent variational autoencoders (RNN-VAR-AEs) with RNN, LSTM, and GRU layers.
* **Versatile Time-Series Analysis**: Can be applied to any time-series dataset with numerical values to discover efficient encoding and decoding architectures.

### Installation ✅

To install NiaNetVAE using pip3 (pending publication to PyPi):

```sh
pip3 install nianetvae
```

### Documentation 📘

The purpose of this paper is to get an understanding of the NiaNetVAE approach.

**TODO - Future Journal:**
[NiaNetVAE for anomaly detection in time-series]()

### Examples

Usage examples can be found [here](nianetcae/experiments). Currently, there is an example for finding the appropriate Recurrent Variational Autoencoder on ECG 500 Dataset.

### Getting started 🔨

##### Create your own example:

1. Replace the dataset in [data](data) folder.
2. Modify the parameters in [main_config.py](configs/main_config.yaml)
2. Adjust the dataloader logic in [dataloaders](nianetvae/dataloaders) folder.
3. Specify the search space in [rnn_vae.py](nianetvae/models/rnn_vae.py) from your problem domain.
3. Redesign the fitness function in [rnn_vae_architecture_search.py](nianetvae/rnn_vae_architecture_search.py) based on your optimization.

##### Changing dataset:

Once the dataset is changed, dataloaders needs to be modified to be able for forwarding new shape of data to models.


##### Specify the search space:

Set the boundaries of your search space as presented in [rnn_vae.py](nianetvae/models/rnn_vae.py).

The following dimensions can be modified:

* **Topology shape:** (symmetrical, asymmetrical)
* **Layer type:** (RNN, LSTM, GRU)
* **Layer step:** (Determined by dataset shape)
* **Number of layers:** (Determined by dataset shape)
* **Activation functions:** (ELU, RELU, Leaky RELU, RRELU, SELU, CELU, GELU, TANH)
* **Optimizer:** (Adam, Adagrad, SGD, RAdam, ASGD, RPROP)

You can run the NiaNet script once your setup is complete.

##### Running NiaNetVAE script with Docker:

```docker build --tag spartan300/nianet:vae . ```

```
docker run \
  --name=nianet-vae \
  -it \
  -v $(pwd)/logs:/app/nianetvae/logs \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/configs:/app/configs \
  -w="/app" \
  --shm-size 8G \
  --gpus all spartan300/nianet:vae \
  python main.py
```

##### Running NiaNetVAE script with Poetry [help](https://github.com/python-poetry/poetry/issues/4231#issuecomment-1182766775):
1. Run the installation via ```poetry install ```
2. Then run the task with```poetry run poe autoinstall-torch-cuda```

##### Running NiaNetVAE script with HPC SLURM:

1. First build an image with docker (above example)
2. Docker push to Docker Hub: ```docker push username/nianet:vae```
3. SSH into a HPC Cluster via your access credentials
4. Create the following _nianetvae.sh_ script: ```cat > nianetvae.sh```
```
#!/bin/bash
## Running code on SLURM cluster
##https://pytorch-lightning.readthedocs.io/en/stable/clouds/cluster_advanced.html
#SBATCH -J nianet-vae-pso
#SBATCH -o nianet-vae-pso-%j.out
#SBATCH -e nianet-vae-pso-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --mem-per-gpu=80GB  # memory per GPU
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00

singularity exec -e \
    --pwd /app \
    -B $(pwd)/logs:/app/logs,$(pwd)/data:/app/data,$(pwd)/configs:/app/configs \
    --nv docker://spartan300/nianet:vae \
    python main.py -alg particle_swarm
```
1. Make script executable: ```chmod +x nianetvae.sh```
2. Make sure that you have the following folders in your current directory: logs, data, configs
3. Set folder permissions to 777: ```chmod -R 777 logs data configs```
4. Submit your script to a job scheduler: ```SBATCH nianetvae.sh```

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

This package is distributed under the MIT License. This license can be found online
at <http://www.opensource.org/licenses/MIT>.

## Disclaimer

This framework is provided as-is, and there are no guarantees that it fits your purposes or that it is bug-free. Use it
at your own risk!



# Fitness Function Overview

This summary explains how the code calculates a single **fitness** value, balancing **reconstruction error** and **model complexity**.

---

## 1. Metric Normalization

For each metric (e.g., **MAE**, **MSE**, **RMSE**, **R²**, **DTW**), the code retrieves **min** and **max** values from a database and **normalizes** the current metric into the range \([0,1]\). The basic formula is:

$$
\text{normalized} \;=\; \frac{\text{value} - \text{min\_val}}{\text{max\_val} - \text{min\_val}}
$$

- If a metric is **better when higher** (e.g., **R²**), the function inverts that range with:
  \[
     1 - \text{normalized}
  \]
- If no prior data exist, the code defaults to using the current value as both min and max.

---

## 2. Error Calculation

The code sums various normalized **reconstruction** metrics:

1. **MAE**, **MSE**, and **RMSE** (all lower-is-better)  
2. **DTW** (if valid for univariate data)  
3. **R²** (treated inversely, so poor R² increases overall error)

In simplified form:

$$
\text{Error} \;\approx\; 
\bigl(\text{Norm(MAE)} + \text{Norm(MSE)} + \text{Norm(RMSE)}\bigr) 
\;+\; \text{optional Norm(DTW)} 
\;+\; \text{Norm(R²)}
$$

> Since **R²** is higher-is-better, its normalization effectively becomes \((1 - R²)\) when added to the error term.

---

## 3. Complexity Calculation

The code **penalizes** large or deep architectures. It looks at:

- Number of encoding layers  
- Number of decoding layers  
- Bottleneck size

Each is **divided** by the time-series length \(\text{seq\_len}\) to keep values in \([0,1]\). They are summed and scaled so that:

$$
\text{Complexity} \;=\; 
\frac{\text{EncLayers}}{\text{seq\_len}} 
\;+\;
\frac{\text{DecLayers}}{\text{seq\_len}}
\;+\;
\frac{\text{BottleneckSize}}{\text{seq\_len}}
$$

*(This result is then normalized further to a maximum of 3.0.)*

---

## 4. Final Fitness

The final **Fitness** is simply:

$$
\text{Fitness} 
\;=\; 
\text{Error} 
\;+\; 
\text{Complexity}.
$$

- The optimization algorithm (e.g., PSO, DE) **minimizes** this value.
- **Lower fitness** indicates better reconstruction (less error) **and** a simpler model (lower complexity).

---

## 5. Handling Invalid Values

If any metric is missing or NaN, the code assigns a **high penalty** \((9 \times 10^{10})\) so that invalid solutions are effectively discarded.

---

**Overall**, this balanced approach encourages **accurate** yet **efficient** VRAE architectures.
