# Experiment scripts

The `run_*_local.sh` scripts are the interface for reviewers. They run
directly with Python and do not require a cluster scheduler or the original
cluster paths:

```bash
./run_icl_local.sh --help
./run_sft_train_local.sh --help
./run_sft_eval_local.sh --help
```

`generate_results_table.sh` formats CSV metrics already produced by an
evaluation run. It does not download models or run inference.
