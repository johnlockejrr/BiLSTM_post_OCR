# Hebrew OCR Correction

This project corrects OCR errors in Hebrew text using a BiLSTM-based neural network.

## Model Architecture

The model is a BiLSTM (Bidirectional Long Short-Term Memory) with the following architecture:

- **Embedding Layer**: Converts input characters into dense vectors.
- **BiLSTM Layer**: Processes the embedded sequences bidirectionally.
- **Attention Mechanism**: Weights the importance of each position in the sequence.
- **Output Layer**: Predicts the corrected character for each position.

### Hyperparameters

- **Embedding Dimension**: 256
- **Hidden Dimension**: 512
- **Number of Layers**: 2
- **Dropout**: 0.3
- **Learning Rate**: 1e-4 (or as found by the learning rate finder)
- **Batch Size**: 32
- **Gradient Accumulation Steps**: 2
- **Gradient Clipping Norm**: 1.0

## Dataset Preprocessing

The dataset is preprocessed as follows:

1. **Tokenization**: Each character is mapped to a unique index.
2. **Special Tokens**: `<PAD>`, `<UNK>`, `<START>`, and `<END>` tokens are added.
3. **Padding**: Sequences are padded to a fixed length (100 characters).
4. **DataLoader**: Batches are created for training and validation.

## Training

To train the model, run:

```bash
python train.py --use_lr_finder --clip_norm 1.0
```

This will:
- Find the optimal learning rate using the learning rate finder.
- Train the model for 50 epochs (or until early stopping).
- Save the best model based on validation loss.

## Prediction

To correct OCR errors in Hebrew text, run:

```bash
(BiLSTM) $ python predict.py --model_path best_model.pt --config_path best_model_config.json --text "מאימתי קורינ את שמע בערבית משעה שהכחנים נכנסים"
Original text: מאימתי קורינ את שמע בערבית משעה שהכחנים נכנסים
Corrected text: מאימתי קורין את שמע בערבית משעה שהכהנים נכנסים
```

This will:
- Load the model and configuration.
- Preprocess the input text.
- Predict the corrected text.

## Project Structure

- `train.py`: Training script.
- `predict.py`: Prediction script.
- `predict_simple.py`: Alternative prediction script (simpler and more reliable).
- `data/processed/`: Processed dataset files.
- `best_model.pt`: Trained model weights.
- `best_model_config.json`: Model configuration.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- tqdm
- wandb (optional)

## License

This project is licensed under the MIT License. 
