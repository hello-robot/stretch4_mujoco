"""
Behaviour-cloning a Stretch policy from scratch, off the simple_ik expert.

    collect.py   generated rollouts -> .npz shards
    dataset.py   the shard format, and the torch-side reader
    train_bc.py  fit `policies/networks.py`'s net, write a checkpoint

The demonstrations come from `../finetuning/generate_dataset.py`, which samples
scenes procedurally: a benchmark's own episodes are the test set, so cloning
them measures memorisation. `../finetuning/` is the other road from the same
data -- fine-tuning a pretrained VLA rather than fitting a small net.
"""
