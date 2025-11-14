import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("data/salary.csv") 
df.head()


df.info() # Ajuda a verificar e observar todas as variáveis disponíveis em salary.csv
df.describe() # Vai exibir as estatísticas descritivas das colunas int64
df.describe(include="object") # Também exibe as estátistas descritivas, porém das colunas texto/objeto

df["salary"].value_counts(normalize=True) # Aqui é o nosso "target"

df.isna().sum() # Mostra valor ausente por coluna 

# Aqui eu vou deixar "printado" só pra poder visualizar no log também

print("\n===== Describe =====\n")
print(df.describe())

print("\n===== Describe (object) =====\n")
print(df.describe(include="object"))

print("\n===== Salary (normalize=True) =====\n")
print(df["salary"].value_counts(normalize=True))

print("\n===== Isma.sum() =====\n")
print(df.isna().sum())

# Tudo aqui abaixo é utilizando a biblioteca seaborn, pra visualizar as tabelas, etc

plt.rcParams['figure.figsize'] = (12, 7) # Aqui é só pra ajustar o tamanho das telas, só mudar se tiver pequeno/grande 

# Distribuição da variável target
sns.countplot(data=df, x="salary")
plt.title("Distribuição da variável alvo (salary)")
plt.show()

# Histogramas das variáveis numéricas
df.select_dtypes(include="number").hist()
plt.tight_layout()
plt.show()

# Scatterplot entre variáveis numéricas
sns.scatterplot(data=df, x="age", y="hours-per-week", hue="salary")
plt.title("Relação entre idade e horas trabalhadas por semana")
plt.show()

# Distribuição de uma variável categórica
df["education"].value_counts().plot(kind="bar")
plt.title("Distribuição de 'Education'")
plt.show()
