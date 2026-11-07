# Handwritten Digit Recognition (scikit-learn + Streamlit)

Pipeline: **Image → Grayscale → 8×8 Pixel Matrix → NumPy Array (64 features) → MLPClassifier → Digit (0–9)**

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit interface |
| `preprocessing.py` | Grayscale, invert, threshold, crop, resize to 8×8, normalize to 0–16, flatten |
| `model.py` | Train / save / load / predict with `MLPClassifier` on `load_digits()` |
| `requirements.txt` | Dependencies |

## Install
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
streamlit run app.py
```
The model trains automatically on first launch (a few seconds) and is saved to `digits_mlp.joblib`.
To retrain manually: `python model.py`

## Why preprocessing matches the dataset
`load_digits()` images are 8×8, integer values 0–16, **light digit on dark background**, centred with a margin.
So the app: converts to grayscale → inverts dark-on-light images → Otsu-thresholds the background →
crops and centres the digit → resizes to 8×8 → scales to 0–16 → flattens to 64 values.
The model divides by 16 (same as in training) before predicting.

## Tips
Use one thick, dark digit on a plain background. Thin pen strokes vanish when shrunk to 8×8.
