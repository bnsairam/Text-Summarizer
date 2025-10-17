import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
import re
from collections import defaultdict

# Download required NLTK data (run once)
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    """Clean text using regex and convert to lowercase."""
    # Remove extra spaces, newlines, and special characters
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[^\w\s.]', '', text)
    return text.lower()

def get_sentence_scores(text, num_sentences=2):
    """Score sentences based on word frequency and return top sentences."""
    # Preprocess text
    clean_text = preprocess_text(text)
    
    # Tokenize sentences
    sentences = sent_tokenize(clean_text)
    if not sentences:
        return [], "Error: No valid sentences found in the input text."

    # Tokenize words and remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word_tokenize(sentence) for sentence in sentences]
    words = [[word for word in sentence if word not in stop_words] for sentence in words]

    # Create a dictionary for word frequency
    dictionary = Dictionary(words)
    corpus = [dictionary.doc2bow(sentence) for sentence in words]

    # Calculate word frequency-based scores (using raw frequency instead of TF-IDF for simplicity)
    word_freq = defaultdict(int)
    for sentence in words:
        for word in sentence:
            word_freq[word] += 1

    # Score sentences based on sum of word frequencies
    sentence_scores = []
    for i, sentence in enumerate(sentences):
        score = sum(word_freq[word] for word in words[i] if word in word_freq)
        sentence_scores.append((score, sentence))

    # Sort sentences by score and select top N
    sentence_scores.sort(reverse=True)
    summary_sentences = [sentence for _, sentence in sentence_scores[:min(num_sentences, len(sentences))]]
    
    # Preserve original sentence case from input text
    original_sentences = sent_tokenize(text)
    summary = []
    for sum_sent in summary_sentences:
        for orig_sent in original_sentences:
            if preprocess_text(orig_sent).strip() == sum_sent.strip():
                summary.append(orig_sent)
                break

    return summary, None

def main():
    print("📝 Text Summarizer")
    print("Enter text to summarize (press Enter twice to finish):")
    
    # Collect multi-line input
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    text = " ".join(lines)
    
    if not text.strip():
        print("Error: No text provided.")
        return

    # Get number of sentences for summary
    try:
        num_sentences = input("Enter number of sentences for summary (default 2): ").strip()
        num_sentences = int(num_sentences) if num_sentences else 2
        if num_sentences <= 0:
            raise ValueError
    except ValueError:
        print("Invalid input. Using default of 2 sentences.")
        num_sentences = 2

    # Generate summary
    summary, error = get_sentence_scores(text, num_sentences)
    
    if error:
        print(error)
    else:
        print("\nSummary:")
        for i, sentence in enumerate(summary, 1):
            print(f"- {sentence}")

if __name__ == "__main__":
    main()
