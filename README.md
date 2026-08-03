# Artistic Movement Recognition

Klasifikacija umetničkih stilova na Pandora18K datasetu pomoću CNN-ova.

## Dataset

Slike su u `Pandora_18k/` (18 klasa). Detalji u `Pandora_18k/Readme_Pandora18k.txt`.

## Setup

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Plan

- EDA i priprema podataka
- Custom CNN (trenirano od nule)
- Transfer learning (pretrained model)
- Evaluacija i izveštaj
