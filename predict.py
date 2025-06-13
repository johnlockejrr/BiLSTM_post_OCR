import torch
import argparse
import json
from train import HebrewOCRCorrector, create_vocab
import torch.nn.functional as F

# Character substitution rules
CHAR_SUBSTITUTIONS = {
    'ח': ['ה', 'ת'],
    'ה': ['ח'],
    'ת': ['ח'],
    'ד': ['ר'],
    'ר': ['ד'],
    'ג': ['נ'],
    'נ': ['ג'],
    'ב': ['כ'],
    'כ': ['ב'],
    'ו': ['י'],
    'י': ['ו'],
    'ז': ['ו'],
    'ט': ['ס'],
    'ס': ['ט'],
    'מ': ['ט'],
    'ט': ['מ'],
    'צ': ['ע'],
    'ע': ['צ']
}

# Final forms and their regular counterparts
FINAL_FORMS = {
    'ם': 'מ',  # mem
    'ן': 'נ',  # nun
    'ץ': 'צ',  # tsadi
    'ף': 'פ',  # pe
    'ך': 'כ'   # kaf
}

# Regular forms and their final counterparts
REGULAR_FORMS = {v: k for k, v in FINAL_FORMS.items()}

def is_final_position(text, pos):
    """Check if position is at the end of a word."""
    return pos == len(text) - 1 or text[pos + 1] == ' '

def postprocess_prediction(text, original_text):
    """Post-process the model's prediction using character substitution rules."""
    result = []
    words = text.split()
    original_words = original_text.split()
    
    for word, original_word in zip(words, original_words):
        new_word = []
        for i, (char, orig_char) in enumerate(zip(word, original_word)):
            is_final = is_final_position(word, i)
            
            # Handle final forms
            if is_final:
                if char in FINAL_FORMS:
                    # If it's a final form, might be confused with regular form
                    if orig_char in REGULAR_FORMS:
                        new_word.append(REGULAR_FORMS[orig_char])
                    else:
                        new_word.append(char)
                elif char in REGULAR_FORMS:
                    # If it's a regular form, might be confused with final form
                    if orig_char in FINAL_FORMS:
                        new_word.append(FINAL_FORMS[orig_char])
                    else:
                        new_word.append(char)
                else:
                    new_word.append(char)
            else:
                # In non-final position, can only use regular forms
                if char in FINAL_FORMS:
                    new_word.append(FINAL_FORMS[char])
                else:
                    new_word.append(char)
            
            # Only apply character confusions if the model's prediction matches the original
            # This means the model didn't make a correction
            if new_word[-1] == orig_char and new_word[-1] in CHAR_SUBSTITUTIONS:
                # Try to find a better correction
                for possible_char in CHAR_SUBSTITUTIONS[new_word[-1]]:
                    if possible_char != orig_char:
                        new_word[-1] = possible_char
                        break
        
        result.append(''.join(new_word))
    
    return ' '.join(result)

def load_model(model_path, config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    model = HebrewOCRCorrector(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def greedy_decode(model, input_tensor, char_to_idx, device):
    with torch.no_grad():
        output = model(input_tensor)
        logits = output.squeeze(0)  # Remove batch dimension
        # Mask out special tokens (except END)
        special_tokens = ['<PAD>', '<UNK>', '<START>']
        for token in special_tokens:
            if token in char_to_idx:
                logits[:, char_to_idx[token]] = float('-inf')
        # Get the most likely token for each position
        predicted_indices = torch.argmax(logits, dim=1)
        return predicted_indices

def correct_text(model, text, char_to_idx, idx_to_char, device, max_len=100):
    # Convert text to indices, pad to max_len
    input_ids = [char_to_idx.get(c, char_to_idx["<UNK>"]) for c in text]
    input_ids = input_ids[:max_len] + [char_to_idx["<PAD>"]] * (max_len - len(input_ids))
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    
    # Get predictions
    predicted_indices = greedy_decode(model, input_tensor, char_to_idx, device)
    
    # Convert indices back to characters
    predicted_chars = [idx_to_char[idx.item()] for idx in predicted_indices]
    
    # Remove special tokens
    predicted_text = ''.join([c for c in predicted_chars if c not in ['<PAD>', '<END>', '<START>', '<UNK>']])
    
    return predicted_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--text', type=str, required=True)
    args = parser.parse_args()
    
    # Load model and configuration
    model = load_model(args.model_path, args.config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Create vocabulary
    char_to_idx, idx_to_char = create_vocab('data/processed/train.jsonl')
    
    # Correct text
    corrected_text = correct_text(model, args.text, char_to_idx, idx_to_char, device)
    
    print(f"Original text: {args.text}")
    print(f"Corrected text: {corrected_text}")

if __name__ == "__main__":
    main() 