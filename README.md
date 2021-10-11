# Multilayer Perceptron

## Description

Ce projet a pour but de créer une librairie de machine learning pour détecter des tumeurs malignes à partir de différentes données.

Source des données: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29.

## Installation

``` bash
$> pip install -r ./requirements.txt
```

## Entrainement
```bash
$> py train.py ./datasets/data_train.csv
```

## Prédiction
```bash
$> py predict.py ./datasets/data_predict.csv
```