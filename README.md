# RSCMR_Baseline

## Set up python environment
we test our code on linux with python3.10 with specific package version in requirement.txt<br>
just pip install -r install requirement.txt <br>

⚠️⚠️⚠️注意！如果pip用了清华源，那么安装pytorch需要去pytorch.org官网上去安装GPU版本的pytorch

## Detail in Dataset
### RSITMD
We follow AMFNM's dataset split, adapt the input for the transformer
<details>
<summary>RSITMD</summary>

| split | num                        |
|-------|----------------------------|
| train | 3432(5 captions per image) |
| val   | 452 (5 captions per image) |
| test  | 2260(1 captions per image) |

code of split dataset in make_series dir.
#### step of make dataset
* first run all the code make_series/make_rsitmd_dataset.ipynb
* then run all the code make_series/make_ours_dataset.py
</details>

## Reference
The dataset splitting approach is inspired by the method proposed in AMFMN<br>
```bibtex
@article{yuan2021exploring,
title={Exploring a Fine-Grained Multiscale Method for Cross-Modal Remote Sensing Image Retrieval},
author={Yuan, Z. and others},
journal={IEEE Transactions on Geoscience and Remote Sensing},
doi={10.1109/TGRS.2021.3078451},
year={2021}
}
```
The input data format is based on the design presented in HarMa<br>
```bibtex
@article{huang2024efficient,
  title={Efficient Remote Sensing with Harmonized Transfer Learning and Modality Alignment},
  author={Huang, Tengjun},
  journal={arXiv preprint arXiv:2404.18253},
  year={2024}
}
```
related metric to evaluate the model in https://github.com/LAION-AI/CLIP_benchmark


### To do list

* processing dataset <progress value="33" max="100">60%</progress>
*  baseline model <progress value="33" max="100">60%</progress>
* metric calculating <progress value="33" max="100">60%</progress>
