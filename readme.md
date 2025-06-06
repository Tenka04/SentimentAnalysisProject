# Sentiment Analysis Project

This project performs sentiment analysis on Amazon food reviews using transformer-based (RoBERTa) models. It demonstrates how to preprocess data, apply sentiment models, and visualize results.

## Features

- Loads and analyzes Amazon food review data
- Uses RoBERTa (HuggingFace Transformers) for deep learning sentiment scoring
- Visualizes sentiment results

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- tqdm
- nltk
- torch
- transformers
- tabulate

Install requirements with:
```bash
pip install pandas numpy matplotlib seaborn tqdm nltk torch transformers tabulate
```

## Data

Place the [Amazon Food Reviews dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) in `input/amazon-food-review/Reviews.csv`.

## Usage

1. Download NLTK (only once):
    ```python
    import nltk
    nltk.download('vader_lexicon')
    ```

2. Run the script:
    ```bash
    python SAP.py
    ```

3. The script will:
    - Analyze the first 500 reviews with both VADER and RoBERTa
    - Print sentiment scores
    - Show pairplot visualizations of RoBERTa sentiment scores by review rating

## Example Output

```
   Id  roberta_neg  roberta_neu  roberta_pos  ...  Score
0   1     0.00123      0.04567      0.95310  ...      5
1   2     0.87654      0.12000      0.00346  ...      1
...
```

## Visualization

The script generates a seaborn pairplot comparing RoBERTa sentiment scores (`roberta_neg`, `roberta_neu`, `roberta_pos`) colored by the review's star rating.

## Notes

- If you see a warning about HuggingFace symlinks on Windows, you can ignore it.
- For best performance, ensure you have a compatible version of PyTorch installed.

## License

This project is for educational purposes.
