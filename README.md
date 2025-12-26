# gaussian_copula_bachelor_thesis
Goal: Generating a synthetic data set

Python Boilerplate contains all the boilerplate you need to create a Python package.

## Inbetriebnahme

Virtuelle Python-Umgebung:

```shell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Pakete aus PIP installieren:

```shell
pip install -r requirements.txt
```


## Project Organization

--------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data/
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── notebooks/                
    │   ├── 01_test.ipynb                           
    │   └── find_best_k.ipynb                         
    │                                                 
    ├── reports/            
    │   ├── figures        <- Generated graphics and figures to be used in reporting
    │   └── summary        <- Generatd KPI reports as xlsx
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src/               <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── analyse.py           <- Scripts analyze the real and synthetic data
    │   │
    │   ├── bestellpositionen.py       <- Scripts for creating synthetic orderlines
    │   │
    │   ├── bestellungen.py         <- Scripts for generating synthtic orderheads
    │   │                   
    │   ├── maerkte.py              <- Scripts for generating synthetic markets (customer)   
    │   │  
    │   └── main.py                 <- Script for the main module
    │   
    │     
    │
    └── .gitignore                  <- File fpr git to ignore


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
