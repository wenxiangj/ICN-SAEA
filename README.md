# ICN-SAEA

This repository provides implementations for the paper: "Surrogate-Assisted Evolutionary Optimization Based on Interpretable Convolution Network"

## Absract

When performing evolutionary optimization for computationally expensive objective, surrogate-assisted evolutionary algorithm(SAEA) is an effective approach. However, due to the limited availability of data in these scenarios, it can be challenging to create a highly accurate surrogate model, leading to reduced optimization effectiveness. To address this issue, we propose an Interpretable Convolution Network(ICN) for offline surrogate-assited evolutionary optimization. ICN retains the non-linear expression ability of traditional neural networks, while possessing the advantages of clear physical structure and the ability to incorporate prior knowledge during network parameter design and training process. We compare ICN-SAEA with tri-training method(TT-DDEA) and model-ensemble method(DDEA-SA) in several benchmark problems. Experimental results show that ICN-SAEA is better in searching optimal solution than compared algorithms.

## Citation

If you find our work and this repository useful. Please consider giving a star and citation.

Bibtex:
```
@inproceedings{jiang2023surrogate,
  title={Surrogate-Assisted Evolutionary Optimization Based on Interpretable Convolution Network},
  author={Jiang, Wenxiang and Xu, Lihong},
  booktitle={2023 IEEE International Conference on Systems, Man, and Cybernetics (SMC)},
  pages={542--547},
  year={2023},
  organization={IEEE}
}
```
