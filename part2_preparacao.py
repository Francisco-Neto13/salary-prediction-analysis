import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/salary.csv")

print("\n===== VERIFICANDO VALORES FALTANTES =====")
print(df.isna().sum())  # Aqui eu já sei que não tem valores faltantes, então não tem o que tratar


print("\n===== APLICANDO ONE-HOT ENCODING =====")  
# One-hot encoding é basicamente transformar variáveis de texto em variáveis binárias. Precisa fazer isso, já que Machine Learning só funciona com números
df_encoded = pd.get_dummies(df, drop_first=True)
print("Formato após encoding:", df_encoded.shape)



print("\n===== IDENTIFICANDO COLUNA DO TARGET =====")  
# O get_dummies cria colunas como 'salary_<=50K' e 'salary_>50K', então preciso detectar qual delas é o target.
cols_salary = [col for col in df_encoded.columns if "salary" in col]
print("Colunas relacionadas ao target:", cols_salary)

# Normalmente a última é a classe positiva (>50K), então usa ela
target_col = cols_salary[-1]
print("Usando como target:", target_col)

# X = dados de entrada (todas as colunas exceto o target)
# y = rótulos (target)
X = df_encoded.drop(target_col, axis=1)
y = df_encoded[target_col]

print("Formato de X:", X.shape)
print("Formato de y:", y.shape)


print("\n===== NORMALIZANDO FEATURES =====")
# Aqui eu só normalizo os dados para que todas as features fiquem na mesma escala.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


print("\n===== DIVISÃO TREINO / TESTE =====")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y # O stratify=y garante que a proporção das classes (<=50K e >50K) seja mantida nas duas partes.
) 

# test-size=0.2 diz basicamente que 20% dos dados vão pra teste, então o resto (80%) vai pra treino

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
