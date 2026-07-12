# Detecção de Fraude em Transações de Cartão de Crédito

Projeto de estudo usando Machine Learning para identificar transações fraudulentas em cartões de crédito, feito como parte do curso de Software Engineering na UTFPR.

## Sobre o dataset

Os dados usados são do dataset público de fraude em cartão de crédito (ULB), com transações de titulares europeus feitas em setembro de 2013. Ao todo são 284.807 transações, das quais apenas 492 são fraudulentas — ou seja, menos de 0,2% do total. Esse desbalanceamento é o principal desafio do projeto.

As colunas V1 a V28 já vêm transformadas por PCA por questão de privacidade dos dados originais. As únicas colunas nesse formato original são `Time` e `Amount`.

Fonte dos dados: https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv

## O que foi feito

- Análise exploratória inicial dos dados
- Normalização da coluna `Amount` com `StandardScaler`
- Separação em treino e teste, mantendo a proporção de classes (`stratify`)
- Treino de dois modelos para comparação: Regressão Logística e XGBoost
- Uso de SMOTE para balancear as classes no conjunto de treino, gerando exemplos sintéticos da classe minoritária (fraude)
- Avaliação usando `classification_report`, com foco em recall e precision da classe fraude (métricas mais relevantes que acurácia nesse tipo de problema)

## Resultado

O modelo XGBoost treinado com SMOTE conseguiu identificar 100% das fraudes presentes no conjunto de teste, com 86% de precisão nas transações marcadas como fraude.

## Tecnologias usadas

- Python
- pandas / numpy
- scikit-learn
- XGBoost
- imbalanced-learn (SMOTE)

## Como rodar

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn
python main.py
```

## Observações

Esse é um projeto de estudo, feito para praticar conceitos de classes desbalanceadas e boosting em problemas reais de fraude. Não deve ser usado como solução final para produção.
