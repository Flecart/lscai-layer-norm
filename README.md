# lscai-layer-norm
Poorly written readme, this is going to get better ;)

### Bootstrap environment
To boostrap the training environment run:
```bash
./bootstrap.sh
```

This creates also a virtual environment in `.venv`, so that it matches the container's pytorch
installation. 

The dependencies are installed in the `.venv`, outside of the docker image build, but inside the container, as suggested by the [Documentation](https://docs.cscs.ch/software/ml/pytorch/#optionally-extend-container-with-virtual-environment)


if you get a compute node with the correct environment we built in the bootstrap.
```bash
srun --account=large-sc-2 -p debug --time=60:00 --environment=ngc_pt_jan --pty bash
```

activate the venv
```bash
source .venv/bin/activate
```

you'll see that cuda is correctly available from the venv
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### wandb
Create `.wandb.env` file and paste the contents of `wandb.env.example`.
Use your own api key.

### Test the tokenizer
```bash
sbatch run_tok_train.sbatch
```

### Train nanochat
```bash
sbatch run_nanochat.sbatch
```

### Logs
All logs are inside `/logs`




## Nematus
run the bootstrap script:

```bash
cd nematus
./scripts/bootstrap_nematus.sh
```

this might take around 30 minutes.


from `nematus/`, train the model with:

```bash
sbatch scripts/experiments/run_nematus.sbatch
```

all logs are inside `nematus/logs`.
