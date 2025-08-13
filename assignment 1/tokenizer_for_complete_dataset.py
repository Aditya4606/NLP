import re
import os
from concurrent.futures import ProcessPoolExecutor
from collections import Counter

# --- Regex patterns (same as your code) ---
SENTENCE_PATTERN = r'https?://[^\s/$.?#]\.[^\s]{2,}'
WORD_PATTERN = r'(https?://\S+|\w+@\w+\.\w+|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d+\.\d+|\d+|[\u0A80-\u0AFF]+|[a-zA-Z]+|[^\w\s])'

# --- Tokenizer functions ---
def sentence_tokenizer(text):
    sentences = re.split(SENTENCE_PATTERN, text.strip())
    return [s.strip() for s in sentences if s.strip()]

def word_tokenizer(sentence):
    return re.findall(WORD_PATTERN, sentence)

# --- Function to process a chunk ---
def process_chunk(chunk):
    sentences = sentence_tokenizer(chunk)
    all_words = []
    reconstructed_lines = []
    for sentence in sentences:
        words = word_tokenizer(sentence)
        all_words.extend(words)
        reconstructed_lines.append(" ".join(words))
    return reconstructed_lines, all_words

# --- Read file in fixed-size chunks ---
def chunk_file(file_path, chunk_size=1024*1024*100):  # 100 MB
    with open(file_path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk

# --- Main parallel execution ---
def parallel_tokenize(file_path, output_path, workers=None):
    if workers is None:
        workers = os.cpu_count() or 4  # use all available cores

    total_sentences = 0
    total_words = 0
    total_characters = 0
    unique_words_set = set()

    with open(output_path, 'w', encoding='utf-8') as out_f:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for reconstructed_lines, all_words in executor.map(process_chunk, chunk_file(file_path)):
                for line in reconstructed_lines:
                    out_f.write(line + "\n")

                total_sentences += len(reconstructed_lines)
                total_words += len(all_words)
                total_characters += sum(len(w) for w in all_words)
                unique_words_set.update(all_words)

    ttr = len(unique_words_set) / total_words if total_words > 0 else 0
    avg_sentence_length = total_words / total_sentences if total_sentences > 0 else 0
    avg_word_length = total_characters / total_words if total_words > 0 else 0

    print("\n--- Corpus Statistics ---")
    print(f"Total number of sentences: {total_sentences}")
    print(f"Total number of words: {total_words}")
    print(f"Total number of characters (no space): {total_characters}")
    print(f"Average Sentence Length (words/sentence): {avg_sentence_length:.2f}")
    print(f"Average Word Length (chars/word): {avg_word_length:.2f}")
    print(f"Type/Token Ratio (TTR): {ttr:.4f}")

# --- Required on macOS/Windows ---
if __name__ == "__main__":
    parallel_tokenize("gu.txt", "combined_gu.txt")