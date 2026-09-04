# Uporedna tabela svih modela

Svi eksperimenti koriste istu stratifikovanu podelu (`data/splits/`, seed fiksiran).
Test skup (2706 slika) se koristi samo na kraju za finalnu evaluaciju.

## Rezultati


| Model              | Parametara | Trenabilnih | Ulaz    | Epoha | Najbolji val acc | **Test acc** | Test loss  | Macro F1   |
| ------------------ | ---------- | ----------- | ------- | ----- | ---------------- | ------------ | ---------- | ---------- |
| VGG-lite           | 1.242.674  | 1.242.674   | 224×224 | 20    | 33.85%           | 35.14%       | 1.9121     | 0.3302     |
| Hibrid             | 57.434     | 57.434      | 224×224 | 20    | 28.09%           | 27.72%       | 2.1002     | 0.2383     |
| Transfer dense     | 21.839.666 | 36.882      | 299×299 | 100   | 52.77%           | 49.93%       | 1.4698     | 0.4965     |
| Transfer fine-tune | 21.839.666 | 6.107.154   | 299×299 | 28    | **59.57%**       | **56.91%**   | **1.3791** | **0.5704** |


Nasumična osnova za 18 klasa: **5.56%**.

## Zaključak

**Najbolji model je transfer fine-tune** (56.91% test accuracy, macro F1 0.570).
Dostignut je referentni opseg iz rada koji repliciramo (56–57%).

### Custom modeli

- **VGG-lite** (35.14%) je bolji od **hibrida** (27.72%), iako hibrid ima manje parametara.  
Dodavanje Inception-stil grane nije pomoglo na ovom skupu verovatno zato što je model  
previše mali za složeniju arhitekturu uz isti broj epoha.
- Oba custom modela su daleko ispod transfer pristupa, što potvrđuje da pretrenirane
ImageNet feature reprezentacije daju veliku prednost za ovaj zadatak.

### Transfer modeli

- **Dense glava** (49.93%) daje solidan skok u odnosu na custom modele (+15 p.p. vs VGG-lite).
- **Fine-tuning** poslednjeg bloka (`mixed9`) podiže rezultat za još **+6.98 p.p.** na testu.
Detaljno poređenje dense vs fine-tune: `experiments/comparison_transfer.md`.

## Figure


| Model              | Learning curves                                | Confusion matrix                                  |
| ------------------ | ---------------------------------------------- | ------------------------------------------------- |
| VGG-lite           | `reports/figures/vgglite_curves.png`           | /                                                 |
| Hibrid             | `reports/figures/hybrid_curves.png`            | /                                                 |
| Transfer dense     | `reports/figures/transfer_dense_curves.png`    | `reports/figures/transfer_dense_confusion.png`    |
| Transfer fine-tune | `reports/figures/transfer_finetune_curves.png` | `reports/figures/transfer_finetune_confusion.png` |


## Izvori podataka

```
experiments/custom_vgglite/test_metrics.json
experiments/custom_hybrid/test_metrics.json
experiments/transfer_dense/test_metrics.json
experiments/transfer_finetune/test_metrics.json
```

