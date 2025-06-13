import os
import json
import random
from tqdm import tqdm

class TextPreprocessor:
    def __init__(self):
        # Character confusions (bidirectional)
        self.char_confusions = {
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
        self.final_forms = {
            'ם': 'מ',  # mem
            'ן': 'נ',  # nun
            'ץ': 'צ',  # tsadi
            'ף': 'פ',  # pe
            'ך': 'כ'   # kaf
        }
        
        # Regular forms and their final counterparts
        self.regular_forms = {v: k for k, v in self.final_forms.items()}
        
        # Special tokens
        self.special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
        
        # Hebrew characters (without final forms)
        self.hebrew_chars = [
            'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת'
        ]
        
        # Talmudic text markers
        self.talmud_markers = ['מתני׳', 'גמ׳', 'תניא', 'אמר', 'אמרו', 'אמרה', 'אמרי', 'אמרי להו', 'אמרי דבי']
    
    def is_final_position(self, text, pos):
        """Check if position is at the end of a word."""
        return pos == len(text) - 1 or text[pos + 1] == ' '
    
    def create_synthetic_error(self, text):
        """Create synthetic OCR errors following the rules."""
        result = []
        words = text.split()
        
        for word in words:
            new_word = []
            for i, char in enumerate(word):
                is_final = self.is_final_position(word, i)
                
                # Handle final forms
                if is_final:
                    # In final position, can use either final or regular form
                    if char in self.final_forms:
                        # If it's a final form, might be confused with regular form
                        if random.random() < 0.3:  # 30% chance of confusion
                            char = self.final_forms[char]
                    elif char in self.regular_forms:
                        # If it's a regular form, might be confused with final form
                        if random.random() < 0.3:  # 30% chance of confusion
                            char = self.regular_forms[char]
                else:
                    # In non-final position, can only use regular forms
                    if char in self.final_forms:
                        char = self.final_forms[char]
                
                # Apply character confusions
                if char in self.char_confusions and random.random() < 0.2:  # 20% chance of confusion
                    char = random.choice(self.char_confusions[char])
                
                new_word.append(char)
            
            result.append(''.join(new_word))
        
        return ' '.join(result)
    
    def normalize_text(self, text):
        """Normalize text by removing extra spaces and empty lines."""
        # Remove extra spaces
        text = ' '.join(text.split())
        # Remove empty lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return '\n'.join(lines)
    
    def split_talmudic_text(self, text, max_length=100):
        """Split Talmudic text into appropriate chunks."""
        chunks = []
        current_chunk = []
        current_length = 0
        
        # Split into lines first
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line starts with a Talmudic marker
            is_marker = any(line.startswith(marker) for marker in self.talmud_markers)
            
            # If line is too long, split it into words
            if len(line) > max_length:
                words = line.split()
                current_line = []
                current_line_length = 0
                
                for word in words:
                    if current_line_length + len(word) + 1 <= max_length:
                        current_line.append(word)
                        current_line_length += len(word) + 1
                    else:
                        if current_line:
                            # Add the current line to the current chunk
                            line_text = ' '.join(current_line)
                            if current_length + len(line_text) <= max_length:
                                current_chunk.append(line_text)
                                current_length += len(line_text)
                            else:
                                # Start a new chunk
                                if current_chunk:
                                    chunks.append(' '.join(current_chunk))
                                current_chunk = [line_text]
                                current_length = len(line_text)
                        current_line = [word]
                        current_line_length = len(word)
                
                if current_line:
                    line_text = ' '.join(current_line)
                    if current_length + len(line_text) <= max_length:
                        current_chunk.append(line_text)
                        current_length += len(line_text)
                    else:
                        if current_chunk:
                            chunks.append(' '.join(current_chunk))
                        current_chunk = [line_text]
                        current_length = len(line_text)
            else:
                # If line starts with a marker or current chunk would be too long, start new chunk
                if is_marker or (current_length + len(line) > max_length):
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                    current_chunk = [line]
                    current_length = len(line)
                else:
                    current_chunk.append(line)
                    current_length += len(line)
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def process_file(self, file_path):
        """Process a single text file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Normalize text
        text = self.normalize_text(text)
        
        # Split into chunks
        chunks = self.split_talmudic_text(text)
        
        # Create pairs for each chunk
        pairs = []
        for chunk in chunks:
            if len(chunk) >= 10:  # Filter out very short chunks
                error_text = self.create_synthetic_error(chunk)
                pairs.append((chunk, error_text))
        
        return pairs
    
    def prepare_dataset(self, input_dir, output_dir, train_ratio=0.9):
        """Prepare the dataset by processing all text files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Process all text files
        all_pairs = []
        for filename in tqdm(os.listdir(input_dir), desc="Processing files"):
            if filename.endswith('.txt'):
                file_path = os.path.join(input_dir, filename)
                pairs = self.process_file(file_path)
                all_pairs.extend(pairs)
        
        # Remove duplicates
        unique_pairs = []
        seen = set()
        for correct, error in all_pairs:
            key = (correct, error)
            if key not in seen:
                seen.add(key)
                unique_pairs.append({
                    'input': error,
                    'target': correct
                })
        
        # Shuffle
        random.shuffle(unique_pairs)
        
        # Ensure we have at least one example in each set
        if len(unique_pairs) < 2:
            print("Warning: Not enough unique pairs for train/val split. Using all data for training.")
            train_pairs = unique_pairs
            val_pairs = []
        else:
            # Split into train and validation
            split_idx = max(1, int(len(unique_pairs) * train_ratio))  # Ensure at least 1 example in train
            train_pairs = unique_pairs[:split_idx]
            val_pairs = unique_pairs[split_idx:]
        
        # Save to files
        with open(os.path.join(output_dir, 'train.jsonl'), 'w', encoding='utf-8') as f:
            for pair in train_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')
        
        with open(os.path.join(output_dir, 'val.jsonl'), 'w', encoding='utf-8') as f:
            for pair in val_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')
        
        print(f"Processed {len(unique_pairs)} unique pairs")
        print(f"Training set: {len(train_pairs)} pairs")
        print(f"Validation set: {len(val_pairs)} pairs")

if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    preprocessor.prepare_dataset('data/txts', 'data/processed') 