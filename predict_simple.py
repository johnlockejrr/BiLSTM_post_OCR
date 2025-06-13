import argparse
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from train import HebrewOCRCorrector  # imports your model class

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def predict(model, text, char_to_idx, idx_to_char, device, max_len=100):
    model.eval()
    with torch.no_grad():
        # Encode input
        input_ids = [char_to_idx.get(c, char_to_idx["<UNK>"]) for c in text]
        input_ids = input_ids[:max_len] + [char_to_idx["<PAD>"]] * (max_len - len(input_ids))
        input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)  # (1, seq_len)

        # Forward pass
        logits = model(input_tensor)  # (1, seq_len, vocab_size)
        preds = torch.argmax(F.log_softmax(logits, dim=-1), dim=-1)  # (1, seq_len)

        # Decode
        decoded = ''.join(idx_to_char[idx.item()] for idx in preds[0] if idx_to_char[idx.item()] not in ["<PAD>", "<END>", "<START>"])
        return decoded.strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="Path to model weights (.pt)")
    parser.add_argument('--config_path', type=str, required=True, help="Path to config JSON")
    parser.add_argument('--text', type=str, required=True, help="OCR text to correct")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    config = load_config(args.config_path)
    char_to_idx = config["char_to_idx"]
    idx_to_char = {int(k): v for k, v in config["idx_to_char"].items()}  # fix JSON str keys

    # Load model
    model = HebrewOCRCorrector(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"]
    )
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    # Run prediction
    corrected = predict(model, args.text, char_to_idx, idx_to_char, device)
    print(f"\n📝 Original:  {args.text}\n✅ Corrected: {corrected}")

if __name__ == "__main__":
    main()

