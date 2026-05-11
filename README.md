## RE-IMPLEMENTATION DETAILS

Fine-tunes RoBERTa-large on SST-2 with LoRA adapters injected into query and key matrices, trained for 6 epochs, lr=1Ã—10â»â´, AdamW, cosine schedule. Also implements LoRA3 (Î”W = BCA), adding a square matrix C between A and B for extra low-rank computation at very low parameter cost.

## REPRODUCTION STEPS

```bash
pip install torch transformers datasets tqdm accelerate
git clone https://github.com/j2399/LoRA-reimp.git
```

Open `code/LoRA.ipynb` in Google Colab (GPU recommended, ~16GB VRAM), run all cells, then:

```python
set_seed(42)
run_all_experiments(data_used=0.5)  # set 1.0 for full dataset
```

---

## RESULTS / INSIGHTS

| Model | Rank | Trainable Params | Val Accuracy |
|---|---|---|---|
| LoRA (AÂ·B) | 4 | 395,266 (0.111%) | 95.64% |
| LoRA (AÂ·B) | 8 | 788,482 (0.222%) | 95.87% |
| LoRA3 (AÂ·CÂ·B) | 4 | 396,034 (0.111%) | 95.41% |
| LoRA3 (AÂ·CÂ·B) | 8 | 791,554 (0.222%) | 95.87% |
| **Paper (Hu et al.)** | **â€”** | **~0.2%** | **96.2 Â± 0.5%** |

---

## CONCLUSION

LoRA adapters are effective at reducing parameters while maintaining high accuracy â€” best result of 95.64% achieved with 011% of RoBERTa-large's 355M parameters, which is withing the error bound of the paper.

---

## REFERENCES

- Hu, E.J. et al. *LoRA.* ICLR 2022. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Liu, Y. et al. *RoBERTa.* arXiv:1907.11692, 2019.
- Wolf, T. et al. *HuggingFace Transformers.* EMNLP 2020.
- Paszke, A. et al. *PyTorch.* NeurIPS 2019.

---

## ACKNOWLEDGEMENTS

Completed as part of coursework at Cornell University. Dataset via [HuggingFace](https://huggingface.co/datasets/nyu-mll/glue), base model [RoBERTa-large](https://huggingface.co/roberta-large).
