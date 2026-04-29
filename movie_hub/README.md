# Movie hub 

A compact Python + Streamlit project for movie analytics and recommendations.

## What this app does
- Dashboard with dataset statistics and charts
- Recommender tab based on movie similarity
- Explore tab with search, filters, and sorting
- Taste Lab for preference-based movie suggestions

## Data source
This project uses the **MovieLens latest-small** dataset from GroupLens.

## How to run

1. Open a terminal in this folder.
2. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
3. Download the dataset:
   ```bash
   python download_data.py
   ```
4. Start the app:
   ```bash
   python -m streamlit run app.py
   ```

## Folder structure
```text
cinelens_project/
├── app.py
├── movie_utils.py
├── download_data.py
├── requirements.txt
└── README.md
```

After downloading, the data will be here:

```text
data/ml-latest-small/
├── movies.csv
├── ratings.csv
├── tags.csv
└── links.csv
```
