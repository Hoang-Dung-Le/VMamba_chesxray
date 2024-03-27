**Introduction**<br>
This project utilizes Vmamba for the chesxray dataset (multi labels), achieving a significant accuracy improvement with a ROC AUC score of 0.847 using the vssm_2020 pretrain. <br>
**Installation** <br>
&emsp; follow <a herf='https://github.com/MzeroMiko/VMamba?tab=readme-ov-file#getting-started'>here</a>.<br>
**Getting started**
```bash
cd classification/
```
```bash
python prepare.py
```
Training
```python
python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29501 main.py  --cfg configs/vssm1/vssm_tiny_224_0220.yaml  --batch-size 32 --data-path data_splits/chestxray --output checkpoint/
```
