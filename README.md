# LMTA: Learning Multi-Timescale Abstractions for Hierarchical Combinatorial Planning

PyTorch implementation of *Learning Multi-Timescale Abstractions for Hierarchical Combinatorial Planning*
by Vivienne Huiling Wang, Tinghuai Wang and Joni Pajarinen ([arXiv:2605.17058](https://arxiv.org/abs/2605.17058)).

## Dependencies
- Python 3.11
- PyTorch 2
- NumPy
- NetworkX

## Usage
Adaptive influence maximization (`aim/`):
```
cd aim
python Main.py --T 10 --budget 70 --test_budget 70 --seed 42
```

Stochastic orienteering (`sop/`):
```
cd sop
python Main.py --T 10 --budget 60 --seed 42
```

Each run trains one seed, keeps the checkpoint with the best validation reward, and writes the held-out test rewards to `heldout.json`.

## Citation
If you find this work useful in your research, please consider citing the paper:
```
@inproceedings{vhwang2026lmta,
  title={Learning Multi-Timescale Abstractions for Hierarchical Combinatorial Planning},
  author={Vivienne Huiling Wang and
          Tinghuai Wang and
          Joni Pajarinen},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```
