# Methodology

## Data
Himawari imagery ingested on a schedule.

## Preprocessing
- Resize to a fixed square size
- Normalize pixels to [0,1]
- Pseudo-label baseline: brightness threshold mask (placeholder)

## Models (planned)
- U-Net segmentation (probabilistic masks)
- ConvLSTM nowcasting (next-frame prediction)
- Autoencoder embeddings + PCA for regime discovery
