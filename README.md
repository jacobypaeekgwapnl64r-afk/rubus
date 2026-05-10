# Rubus Endocarp Identification

A practical deep-learning pipeline for classifying modern Rubus endocarp images and predicting archaeological specimens.

## Features
- Supervised classification using transfer learning (`timm` + PyTorch)
- Train/val/test split from folder dataset
- Class imbalance handling with weighted loss
- Early stopping and best-checkpoint saving
- Folder inference with confidence threshold and unknown rejection
- Embedding export + PCA visualization + nearest-neighbor retrieval
- Confusion matrix and per-class metrics

## Recommended workflow
1. Organize modern reference images by species folders.
2. Train on modern images.
3. Evaluate on held-out modern images.
4. Predict archaeological images with thresholded rejection.
5. Export embeddings and inspect nearest modern neighbors for each archaeological sample.
6. Use model output as evidence, not as the sole taxonomic decision.

## Folder structure
```text
data/
  modern/
    R_parvifolius/
      img001.jpg
      img002.jpg
    R_hirsutus/
      ...
    R_rosifolius/
      ...
  archaeological/
    site_A/
      arch_001.jpg
      arch_002.jpg
```

## Quick start
```bash
pip install -r requirements.txt
python train.py \
  --data_dir data/modern \
  --output_dir outputs/exp1 \
  --model tf_efficientnetv2_s.in21k_ft_in1k \
  --img_size 384 \
  --batch_size 16 \
  --epochs 30

python predict.py \
  --checkpoint outputs/exp1/best.pt \
  --input_dir data/archaeological/site_A \
  --output_csv outputs/exp1/site_A_predictions.csv \
  --threshold 0.60

python export_embeddings.py \
  --checkpoint outputs/exp1/best.pt \
  --modern_dir data/modern \
  --arch_dir data/archaeological/site_A \
  --output_dir outputs/exp1/embeddings
```

## Notes for better accuracy
- Keep imaging setup as consistent as possible: magnification, lighting, background, orientation.
- If possible, crop each image to a single endocarp and remove scale bars/text.
- Add multiple individuals per species and multiple populations when available.
- Archaeological images often have domain shift. If you have a small labeled archaeological set, fine-tune on it.
- Use species-level output together with morphology, measurements, and archaeological context.
