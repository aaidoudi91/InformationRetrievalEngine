# Information Retrieval Engine

A comprehensive implementation of classical and modern information retrieval techniques, including TF-IDF, BM25+, and neural embeddings for document ranking and retrieval.

## Project Overview

This project implements a complete information retrieval pipeline for ranking and retrieving relevant documents from a large corpus. It explores multiple retrieval paradigms:

- **Classical Methods**: TF-IDF and BM25+ for term-based ranking
- **Neural Methods**: Sentence embeddings using transformers for semantic search
- **Evaluation**: Comprehensive metrics including Recall, Precision, and MRR
- **Classification**: Document categorization for retrieval optimization

## Dataset

- **216,041 documents** across 5 categories (tex, unix, gaming, programmers, android)
- **327 training queries** with ground truth relevance judgments
- **~3,000 relevance judgments** for evaluation
- Average document length: ~973 characters
- Average query length: ~48 characters

## Project Structure
```
├── notebooks/
│ ├── 01_data_exploration.ipynb  # Dataset analysis and preprocessing
│ ├── 02_tfidf_bm25.ipynb        # Classical retrieval methods
│ ├── 03_embeddings.ipynb        # Neural embeddings with transformers
│ ├── 04_evaluation.ipynb        # Performance comparison
│ └── 05_classification.ipynb    # Document classification & reranking
├── src/
│ ├── preprocessing.py           # Text preprocessing utilities
│ ├── retrieval.py               # Retrieval model implementations
│ └── evaluation.py              # Evaluation metrics
├── data/
│ └── README.md
└── requirements.txt
```

## Technologies Used

- **Python 3.8+**
- **scikit-learn** - TF-IDF vectorization
- **rank-bm25** - BM25+ implementation
- **sentence-transformers** - Neural text embeddings
- **numpy**, **pandas** - Data manipulation
- **matplotlib**, **seaborn** - Visualization
- **tqdm** - Progress tracking

## Contributors

Aaron Aidoudi - M1 Artificial Intelligence, Université Paris Cité

Supervised by Prof. Themis Palpanas & Manos Chatzakis
