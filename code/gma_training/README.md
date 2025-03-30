# Training Script

Code to train a Gated-MIL Attention (GMA) aggregation model for all benchmarking tasks.
Usage:
```
python train.py \
       --output output/directory \
       --data benchmark_task # one of the 20 tasks included\
       --encoder foundation_model \
       --mccv 1 # 1-20 monte carlo cross validation runs
       --lr 0.0001
```
The script will produce a log file named `convergence.csv` with training loss and validation AUC.
