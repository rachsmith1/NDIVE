# Differentiable Vertex Fitting for Jet Flavour Tagging (NDIVE)

This is the code repository for the paper *Differentiable Vertex Fitting for Jet Flavour Tagging*.

## Instructions for setup at SLAC Shared Data Facility 

- Go to https://sdf.slac.stanford.edu/
- Login with SLAC Windows credentials
- Go to "My Interactive Sessions"
- Under "Services" go to "Jupyter"
- Under "Jupyter Instance" choose "Custom Singularity Image..."
- Under "Commands to initiate Jupyter" paste the following:
```
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export SINGULARITY_IMAGE_PATH=/gpfs/slac/atlas/fs1/d/recsmith/Vertexing/diva.sif
function jupyter() { singularity exec --nv -B /sdf,/gpfs,/scratch,/lscratch ${SINGULARITY_IMAGE_PATH} jupyter $@; }
```
- Tick the box next to "Use JupyterLab instead of Jupyter Notebook?"
- The following settings are recommended:
    - Partition: 'shared' or 'atlas'
    - Number of hours: 12 (or however long you want your instance to be)
    - Number of CPU cores: 1
    - Total Memory to allocate: 100096
    - Number of GPUs: 2
    - GPU Type: Nvidia Tesla A100
- Press "Launch"
- When job is ready, press "Connect to Jupyter"

You may also need to install this project as a repo, depending on the singularity shell status. 
- If you try running and see 'no module named diffvert', this is the case. 
- To resolve, navigate to where you have cloned this repo. Then run:
```
pip install -e .
```

## Instructions for training

- In order to checkpoint models you must set an environment variable ```NDIVE_MODEL_PATH``` to a directory
    - A natural place for this is where this project is downloaded followed by '/diffvert/models/saved_models'
    - If concerned about space, can also make this in a place with larger partition
    - Checkpoints of your model every 25 epochs will be cached here
    - When running inference plots, inference outs will be cached here

- Suggested workflow:
    - Define a new TrainConfig object in diffvert/configs.py
    - Give this a meaningful name as a variable and give it an instructive config_name string. config_name dictates which directory the model is saved as (if left blank it will be saved as a hash of the config object which is uninterpretable)
    - Run   ```python3 train.py -t {train config variable name as defined in configs.py}```
    - If run without configs, models will always be saved under a hash of config generated from parse variables

- Available arguments:
    - `-m` `--model`
        - choose which model to train (see available models in `models` folder)
    - `-s` `--samples` 
        - choose which samples to use for training
        - default: `'/gpfs/slac/atlas/fs1/d/recsmith/Vertexing/samples/all_flavors/all_flavors'`
    - `-n` `--num_gpus`
        - number of gpus 
        - default: 2
    - `-b` `--batch_size`
        - batch size for training
        - default: 100
    - `-e` `--epochs`
        - number of epochs
        - default: 300
    - `-l` `--learning_rate`
        - learning rate for training
        - default: 1e-5
    - `-p` `--pretrained_NDIVE`
        - train the larger ftag model with pretrained NDIVE
        - default: False
    - `-c` `--cont`
        - continue training a model starting at last saved checkpoint
        - default: False
    - `-t` `--train_config`
        - train a model from a pre-defined TrainConfig object, defined in diffvert/configs.py
        - default: 'none'
    - `--train_count`
        - specify count of identical model to train
        - default: 0

## Evaluation of models

All saved models are accessed using ```${NDIVE_MODEL_PATH}```. 

The code for making plots is under ```diffvert/evaluation```.

There are separate notebooks for inference_graphs (plots of model performance), roc_curves (solely roc curves), and loss_plots (how losses behave over epochs).

```plot_helpers``` is a library of functions used by the plotting jupyter notebooks. One particularly important function is ```get_test_output```. You pass to this a config name (listed in diffvert/configs.py or just under NDIVE_MODEL_PATH directory), as well as a trained epoch (or none to get final epoch), as well as a model number if multiple copies were trained (pass as none if only one copy trained). If this combination of name, epoch, and number has been requested before the outputs will be cached and no inference will be done.

The outputs are stored as a dict of multi-dimensional numpy arrays with keys b_output_arr, b_input_arr, c_output_arr, b_input_arr, u_output_arr, u_input_arr. The convention for these names is truth flavor followed by input or output. Note that the full input data also contains the truth output data. 

The final plots produced for the paper are in ```diffvert/evaluation/paper-plots.ipynb```.
