# Real Estate Price Prediction

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat?style=flat&logo=fastapi)

A **machine learning model** to predict **real estate prices** using the **[Real Estate Price Insights](https://www.kaggle.com/datasets/wardabilal/real-estate-price-insights)** dataset. Deployed as a **FastAPI service** in **Docker**, managed with **uv** for fast, reliable dependency resolution.

---

## Problem Statement & Motivation

> **"What drives the price of a home?"**

In an increasingly competitive and volatile real estate market, **accurate price prediction** is essential for:

- **Buyers**: Avoid overpaying in hot markets
- **Sellers**: Price competitively to attract offers
- **Investors**: Identify undervalued properties
- **Lenders & Insurers**: Assess risk with data-driven valuations
- **Policymakers**: Monitor housing affordability trends

Traditional valuation relies on manual appraisals — **slow, costly, and inconsistent**.

This project uses **machine learning** to:
- Predict **property sale prices** from structural and locational features
- Quantify **feature importance** (e.g., sqft, bedrooms)
- Deliver **instant, scalable predictions** via API

---

## Dataset

**Source**: [Real Estate Price Insights](https://www.kaggle.com/datasets/wardabilal/real-estate-price-insights)
**Download**:
```bash
kaggle datasets download -d wardabilal/real-estate-price-insights -p data/raw --unzip
```

---
## Deployment And Containerization

The model is deployed as a FastAPI application and as a docker container. 
The file called `predict.py` contains the core logic of web service. 
The app itself is served using uvicorn.

**Run Model training:** 

Install [UV](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer) 

```
uv sync --locked
cd src/
uv run train.py
```

**Run FastAPI application:** 

Install [UV](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer) 

```
uv sync --locked
cd src/
uv run uvicorn --bind 0.0.0.0:9696 predict:app
```


**Run application as a docker container:**
```
docker build -t real-estate-price-prediction .
docker run -it --rm -p 9696:9696 real-estate-price-prediction
```

---
## Testing

A test script is provided. Modify as needed:
```
uv sync --locked
cd src/
uv run test-service.py
```
