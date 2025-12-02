# Implementing and documenting mlp scaling

Question: We want to investigate whether the benefits attributed to normalization (like stability and gradient conditioning) can instead be reproduced by structured optimizer adaptivity (per-row, per-column, per-block moments of W) — and whether per-parameter second moments are actually necessary


### Model
For this experiments we have modified karpathy's Nanochat, a minimal hackable implementation of an LLM like ChatGPT.

### Task
The training data is [fineweb-edu-100b-shuffle](https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle)


# Baselines
This is the original Nanochat model without architectural modifications

config params:

```
NORM_TYPE=rms
NORM_EPS=1e-6
MLP_TYPE=default

QK_NORM_TYPE=rms
PRE_ATTN_NORM_TYPE=rms
```

Next we change the norm type to a learnable-rms and compare the two. Surprisingly we notice that learnable-rms makes the training slower.

```
NORM_TYPE=learnable-rms
NORM_EPS=1e-6
MLP_TYPE=default

QK_NORM_TYPE=rms
PRE_ATTN_NORM_TYPE=rms
```

insert picture:


## Column fusion weights - Sanity Check
We first try incoporating in the MLP layers the column-wise scale parameters. When we set the norm_type to "rms", this MLP column fusion should be equivalent to the learnable-rms baseline

config params:
```
NORM_TYPE=rms
NORM_EPS=1e-6
MLP_TYPE=column

QK_NORM_TYPE=rms
PRE_ATTN_NORM_TYPE=rms
```