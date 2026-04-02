# Information Retrieval Engine

A complete information retrieval pipeline evolving from classical lexical methods (TF-IDF, BM25+) to neural retrieval with domain adaptation and contrastive fine-tuning.
Built as part of a Kaggle competition on multi-domain technical Q&A retrieval.

- **Classical Lexical Search**: TF-IDF and BM25+ for term-based baseline ranking.
- **Dense Retrieval (Bi-Encoders)**: Semantic search using Sentence Transformers (*all-MiniLM*) to solve the vocabulary mismatch problem.
- **Classification-Based Filtering**: TF-IDF + Logistic Regression with dynamic confidence thresholds (Soft vs. Hard filtering) to prune search space by domain category.
- **Cross-Encoder Reranking**: Implementation of a Two-Stage Retrieval pipeline using *ms-marco* Cross-Encoders to fine-rank top candidates.
- **Advanced Model Fine-Tuning**:
  - **Unsupervised Domain Adaptation**: Pre-training the embedding space on the specific jargon of the target dataset.
  - **Supervised Fine-Tuning**: Using *MultipleNegativesRankingLoss*.
  - **Hard Negatives Mining**: Programmatically identifying and feeding difficult negative examples to the model to drastically improve precision.

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
│ └── 05_advanced_pipeline.ipynb # Classification, Cross-Encoder Reranking & Fine-tuning
│
├── report.pdf
├── README.md
└── requirements.txt
```

## Technologies Used

- **Python 3.8+**
- **PyTorch** - Deep learning framework for model fine-tuning and domain adaptation
- **sentence-transformers** - Bi-Encoders, Cross-Encoders, and loss functions
- **scikit-learn** - TF-IDF vectorization and Logistic Regression
- **rank-bm25** - BM25+ implementation
- **umap-learn** - 2D dimensionality reduction and embedding visualization
- **numpy**, **pandas** - Data manipulation
- **matplotlib**, **seaborn** - Visualization

## Results

| Method | Recall | MRR | Kaggle Score |
|---|---|---|---|
| TF-IDF | 0.217 | 0.147 | - |
| BM25+ | 0.243 | 0.186 | - |
| all-MiniLM-L12-v2 | 0.418 | 0.285 | 0.312 |
| + Hard Category Filter | 0.941 | 0.362 | 0.576 |
| + Domain Adapt. + Fine-Tune | 0.993 | 0.614 | 0.633 |

## Contributors

Aaron Aidoudi - M1 Distributed Artificial Intelligence, Université Paris Cité

Supervised by Prof. Themis Palpanas & Manos Chatzakis
