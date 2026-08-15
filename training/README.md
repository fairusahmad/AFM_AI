# Training Scripts

Run these from the project root:

```bash
python training/train_remount_real.py --device cuda
python training/train_remount_5w.py --device cuda
python training/train_ml_models.py
python training/train_repositioning_ai.py
python training/preprocess_data.py
python training/train_inverse_model.py
```

Main inputs:
- `collected_data/site_memories/`
- `collected_data/*.csv`
- `inverse_model_data.pkl`

Main outputs:
- `collected_data/models/*.pkl`
- `inverse_model.pkl`
