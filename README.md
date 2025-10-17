# 📝 Text Summarizer

A Python-based NLP tool that summarizes long text by extracting the most important sentences using frequency-based scoring. This project demonstrates natural language processing techniques and practical AI concepts for summarizing content.

## 📘 Overview

The Text Summarizer takes a long paragraph or document as input, tokenizes it into sentences, and generates a concise summary by ranking sentences based on word frequency. It’s an excellent showcase of NLP logic, text processing, and practical AI applications, making it appealing for recruiters.

## ✨ Features

- **Sentence Tokenization**: Breaks down text into sentences using `nltk`.
- **Frequency-Based Scoring**: Ranks sentences based on the frequency of significant words.
- **Concise Summary**: Outputs a user-defined number of key sentences as a summary.
- **Text Preprocessing**: Cleans text using `re` for consistent processing.

## 🛠️ Technologies Used

- **Python**: Core programming language.
- **nltk**: For sentence and word tokenization, stopword removal, and text processing.
- **gensim**: For word frequency analysis and text preprocessing.
- **re**: For regular expression-based text cleaning.

## 🚀 Getting Started

### Prerequisites
- Python 3.6 or higher
- `nltk` library (`pip install nltk`)
- `gensim` library (`pip install gensim`)

### Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/bnsairam/text-summarizer.git
   cd text-summarizer
