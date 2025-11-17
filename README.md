# 📊 Salary Prediction Classification – Regressão Linear vs Regressão Logística
Este projeto tem como objetivo analisar e modelar o conjunto de dados "Salary Prediction Classification", utilizando abordagens de Regressão Linear e Regressão Logística para prever faixas salariais com base em características socioeconômicas.

---

## 🧠 Objetivo

Comparar duas abordagens de modelagem para o problema de previsão de renda:
Regressão Linear – utilizada em formato contínuo (0 ou 1), adaptada para prever probabilidades.
Regressão Logística – abordagem nativa para classificação binária.

🔍 Descrição do Dataset

O dataset contém variáveis demográficas e socioeconômicas de indivíduos.
A variável alvo é salary, com valores:

<=50K (classe 0 – majoritária);
> 50K (classe 1 – minoritária).

Exemplos de variáveis:
Numéricas: age, hours-per-week, capital-gain
Categóricas: workclass, education, occupation, sex

---

## 📊 Exploração dos Dados

- Estatísticas descritivas e análise das variáveis.
- Verificação da presença de outliers.
- Análise da proporção desbalanceada das classes.

---

## 🤖 Modelagem

### 🔹 Regressão Linear

- Usada para prever o valor contínuo da variável alvo.
- Métricas: MAE, MSE, RMSE e R².
- Resultado: desempenho limitado para classificação.

### 🔸 Regressão Logística

- Modelo adequado para classificação binária.
- Métricas: Acurácia, Precisão, Recall, F1-Score, Matriz de Confusão.
- Desempenho superior e melhores previsões para a classe >50K.

---
