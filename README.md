# Towards Understanding LLM Latent Inference through Fuzzy Reasoning Dynamics Modeling

Official implementation of **Fuzzy Reasoning Dynamics Modeling (FRDM)**.

---


## Main Results

### Transition Tracking

| Backbone | Dataset | R²@1 ↑ | R²@2 ↑ | R²@4 ↑ | Δ CosSim ↑ |
|---|---|---:|---:|---:|---:|
| Llama3-8B | 2Wiki | **0.960** | **0.530** | **0.296** | **0.942** |
| Llama3-8B | SOCRATES | **0.921** | **0.778** | **0.578** | **0.848** |
| Pythia-6.9B | 2Wiki | **0.976** | **0.342** | **0.360** | **0.943** |
| Pythia-6.9B | SOCRATES | **0.950** | **0.653** | **0.660** | **0.832** |

FRDM maintains positive recursive prediction performance at both `R²@2` and `R²@4` across all four dataset-backbone configurations.

## Data Preparation

FRDM uses:

- [2WikiMultiHopQA](https://aclanthology.org/2020.coling-main.580/)
- [SOCRATES](https://aclanthology.org/2025.findings-acl.205/)




## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{li2026frdm,
  title     = {Towards Understanding LLM Latent Inference through Fuzzy Reasoning Dynamics Modeling},
  author    = {Li, Fusheng and Wu, Hao and Yang, Wangli and Li, Wanqing and Zhang, Wenbin and Guo, Yi and Yang, Jie},
  booktitle = {NeurIPS 2026 Workshop on AI for Stochastic Dynamics},
  year      = {2026}
}
```

---

## Contact

For questions regarding the paper or code, please contact:

- **Fusheng Li** — `fl443@uowmail.edu.au`
- **Jie Yang** — `jiey@uow.edu.au`



