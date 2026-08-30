# Transfer learning: dense glava vs fine-tuning poslednjeg bloka

Oba modela dele istu zamrznutu podelu (`data/splits/`), istu InceptionV3 bazu i istu
veličinu ulaza (299x299). Razlika je samo u tome koliko se mreže uči.

## Rezultati

| | `transfer_dense` | `transfer_finetune` |
|---|---|---|
| Šta se uči | samo Dense glava | Dense glava + blok od `mixed9` |
| Parametara ukupno | 21.839.666 | 21.839.666 |
| Trenabilnih | 36.882 | 6.107.154 |
| Learning rate | 0.001 | 1e-05 |
| Keširani feature-i | da | ne |
| Augmentacija | ne | da |
| Odrađenih epoha | 100 | 28 (early stopping, limit 30) |
| Najbolji val accuracy | 52.77% | **59.57%** |
| **Test accuracy** | 49.93% | **56.91%** |
| Test loss | 1.4698 | 1.3791 |
| Macro F1 | 0.4965 | **0.5704** |
| Weighted F1 | 0.4912 | **0.5657** |

## Da li fine-tuning pomaže?

Da, i to nedvosmisleno: test accuracy raste sa 49.93% na
56.91%, dakle **+6.98 procentnih poena**. Macro F1 raste sa
0.4965 na 0.5704, što znači da dobitak nije koncentrisan u
nekoliko velikih klasa nego je raspoređen po celom skupu.

Time je dostignut referentni opseg iz rada koji repliciramo (56-57%). Objašnjenje je da
ImageNet feature-i iz poslednjeg bloka opisuju objekte i scene, a ne slikarski postupak;
kad se taj blok pusti da se pomeri, mreža počne da hvata teksturu poteza i paletu, što je
ono po čemu se stilovi zapravo razlikuju.

Cena je vreme: dense faza radi nad keširanim vektorima i epoha traje sekundama, dok
fine-tune mora kroz same slike i epoha traje oko 4.5 minuta.

## Prepajanje

Fine-tune model se osetno prepaja. Na kraju treninga train accuracy je preko 84%, dok val
stoji na oko 59% — jaz od preko 25 poena. Val loss je najniži u 13. epohi (1.2499) pa
krene naviše, iako val accuracy nastavi da se penje još malo. Zbog `ModelCheckpoint` po
`val_accuracy` to ne kvari prijavljeni rezultat, ali pokazuje da su dropout 0.5,
augmentacija i zamrznut BatchNorm bili nužni, a ne opcioni.

Early stopping je presekao trening u 28. od 30 epoha.

## Po klasama

Promena recall-a, sortirano po dobitku:

| Klasa | dense | finetune | razlika |
|---|---|---|---|
| `15_Surrealism` | 36.65% | 59.63% | +23.0 |
| `05_Baroque` | 29.05% | 49.32% | +20.3 |
| `10_Post_Impressionism` | 22.92% | 40.10% | +17.2 |
| `18_PopArt` | 69.64% | 81.55% | +11.9 |
| `07_Romanticism` | 30.60% | 41.79% | +11.2 |
| `08_Realism` | 34.44% | 45.00% | +10.6 |
| `16_AbstractArt` | 57.86% | 66.67% | +8.8 |
| `01_Byzantin_Iconography` | 85.04% | 93.70% | +8.7 |
| `02_Early_Renaissance` | 66.96% | 74.11% | +7.1 |
| `17_NaiveArt` | 45.57% | 51.90% | +6.3 |
| `12_Symbolism` | 46.54% | 52.83% | +6.3 |
| `04_High_Renaissance` | 60.00% | 65.60% | +5.6 |
| `06_Rococo` | 55.20% | 57.60% | +2.4 |
| `11_Expressionism` | 28.57% | 27.27% | -1.3 |
| `03_Northern_Renaissance` | 62.60% | 60.98% | -1.6 |
| `14_Cubism` | 69.57% | 67.39% | -2.2 |
| `13_Fauvism` | 42.59% | 37.96% | -4.6 |
| `09_Impressionism` | 66.14% | 58.20% | -7.9 |

Najveći dobitak je `15_Surrealism` (+23.0 poena), najveći gubitak
`09_Impressionism` (-7.9).

Taj pad kod `09_Impressionism` je zapravo poboljšanje, i vidi se tek kad se pogleda
precision: dense model je imao recall 0.66 uz precision svega 0.39, tj. koristio je
impresionizam kao korpu za sve što ne prepozna. Kod fine-tune modela precision skače na
0.51, pa iako hvata nešto manje pravih impresionista, mnogo ređe tu gura tuđe slike. Isto
važi i za `13_Fauvism` (precision 0.37 -> 0.48). Zato je macro F1 ovde pošteniji sud od
recall-a po klasi.

Struktura grešaka ostaje ista i posle fine-tuning-a: rani, ikonografski stilovi
(`01_Byzantin_Iconography`, `18_PopArt`) su i dalje najlakši, a susedni pokreti sa
preklapajućom paletom i motivima (`10_Post_Impressionism`, `11_Expressionism`,
`07_Romanticism`) ostaju najteži. Fine-tuning podiže nivo, ali ne menja koje su klase
teške — detaljna analiza tih zabuna je u `notebooks/07_error_analysis.ipynb`.

## Reprodukcija

```bash
python src/transfer.py --stage cache
python src/transfer.py --stage dense --epochs 100
python src/transfer.py --stage finetune
python src/evaluate.py --model transfer_finetune --exp-dir experiments/transfer_finetune
python src/plot_curves.py --model transfer_finetune
```
