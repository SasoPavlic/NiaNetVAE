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
3. Redesign objective/search behavior in [runner.py](nianetvae/search/runner.py) and [objective_engine.py](nianetvae/search/objective_engine.py) based on your optimization.

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

Current methodology note:
- optimizer is **not** part of the searched genome,
- all candidates are trained with the same fixed policy from `exp_params`,
- default fixed policy is `Adam` with shared learning-rate / weight-decay settings.

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

##### Workflow Mode

Set `workflow.mode` in `configs/main_config.yaml`:

- `baseline_search` (default): current architecture-search workflow.
- `per_maint_finetune`: Experiment A workflow.

Notes:
- `per_maint_finetune` requires `data_params.regime: "per_maint"` and `data_params.cycle_id` to be set.
- `per_maint_finetune` behavior:
  - `cycle_id=0`: runs baseline architecture search to initialize the first model.
  - `cycle_id>0`: reuses latest previous trained cycle architecture/weights and performs fine-tune training.
  - If a cycle is non-trainable (for example zero rows after phase filtering), run is skipped gracefully and `cycle_status.json` is written with `status=skipped_non_trainable`.
- Controlled fine-tune policy (cycle `>0`) is config-driven:
  - Fixed optimizer: `exp_params.optimizer` (default `Adam`).
  - Base LR: `exp_params.learning_rate` (default `0.003`).
  - Fine-tune LR: `base_lr * workflow.finetune.learning_rate_scale` (default scale `0.1`).
  - Fine-tune epoch cap: `workflow.finetune.max_epochs` (default `3`).
- If previous cycle artifacts are missing (`model.pt`, `model_meta.json`), run exits with an explicit error.
- Workflow mode is config-only (`workflow.mode` in YAML).

##### Running NiaNetVAE script with HPC SLURM:

1. First build an image with docker (above example)
2. Docker push to Docker Hub: ```docker push username/nianet:vae```
3. SSH into a HPC Cluster via your access credentials
4. Copy the scripts from `slurm_scripts/` to your HPC working directory:
   - `train_per_maint_cycles.sbatch` (single-cycle training job; cycle id passed via env)
   - `build_cycle_manifest.sbatch` (single manifest job)
   - `submit_per_maint_pipeline.sh` (submits sequential cycle chain + final manifest dependency)
5. Make scripts executable: ```chmod +x submit_per_maint_pipeline.sh```
6. Make sure folders `logs`, `data`, `configs` exist in your HPC working directory.
7. Submit the full pipeline (sequential cycles + final manifest):
   ```bash
   ./submit_per_maint_pipeline.sh
   ```
8. Optional resume controls:
   - `RESUME_FROM=auto` (default): starts from first missing cycle artifact.
   - `RESUME_FROM=<cycle_id>`: force resume from a specific cycle.
   - `START_CYCLE=<n> END_CYCLE=<m>`: limit submission range.
   - In auto mode, cycles with `cycle_status.json` marker `skipped_non_trainable` are treated as already handled.

##### Per-maint exported artifacts and manifest (for metropt consumption)

1. In `configs/main_config.yaml` set:
   - `logging_params.export_enabled: true`
   - `logging_params.model_export_dir: logs/per_maint_models`
2. Run per-maint cycles sequentially via `submit_per_maint_pipeline.sh`.
3. Manifest generation runs automatically as the final dependent job.
4. If you run only `train_per_maint_cycles.sbatch`, generate manifest manually:
   ```bash
   python -m nianetvae.tools.generate_cycle_manifest --config configs/main_config.yaml --cycles 0-21
   ```
5. This writes:
   - `logs/per_maint_models/MetroPT/cycle_XX/model.pt`
   - `logs/per_maint_models/MetroPT/cycle_XX/model_meta.json`
   - `logs/per_maint_models/MetroPT/cycle_XX/search_summary.json`
   - `logs/per_maint_models/MetroPT/cycle_XX/cycle_status.json` (only for skipped non-trainable cycles)
   - `logs/per_maint_models/MetroPT/cycle_manifest.json`
6. Manifest artifact paths are stored relative to the manifest directory for cross-platform portability (HPC Linux -> local Windows).
7. Manifest now includes top-level `workflow_mode` (for example `per_maint_finetune`) to make run context explicit downstream.

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

For each metric (e.g., **MAE**, **MSE**, **RMSE**, **MAPE**, **RMAPE**, **RMAPE**), the code retrieves **min** and **max** values from a database and **normalizes** the current metric into the range \([0,1]\). The basic formula is:

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

In simplified form:

$$
\text{Error} \;\approx\; 
\bigl(\text{Norm(MAE)} + \text{Norm(MSE)} + \text{Norm(RMSE)}\bigr)
$$

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
