import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Konfiguracja
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


print("\n" + "="*70)
print("REGRESJA WIELORAKA (MULTIPLE LINEAR REGRESSION)")
print("="*70)

# Generowanie danych
np.random.seed(42)
n = 100

# Zmienne niezależne
powierzchnia = np.random.uniform(30, 150, n)  # m²
liczba_pokoi = np.random.randint(1, 6, n)
wiek = np.random.uniform(0, 50, n)  # lata

# Zmienna zależna (z dodanym szumem)
cena = (100000 +
        5000 * powierzchnia +
        20000 * liczba_pokoi -
        1000 * wiek +
        np.random.normal(0, 50000, n))

# Tworzenie DataFrame
df = pd.DataFrame({
    'powierzchnia': powierzchnia,
    'liczba_pokoi': liczba_pokoi,
    'wiek': wiek,
    'cena': cena
})

print("\nPierwsze 5 wierszy danych:")
print(df.head())

print("\nStatystyki opisowe:")
print(df.describe())

# Macierz korelacji
print("\n" + "-"*70)
print("MACIERZ KORELACJI:")
print("-"*70)
correlation_matrix = df.corr()
print(correlation_matrix['cena'].sort_values(ascending=False))

# Wizualizacja korelacji
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm',
            center=0, fmt='.3f', linewidths=1)
plt.title('Macierz Korelacji', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Przygotowanie danych
X_multi = df[['powierzchnia', 'liczba_pokoi', 'wiek']]
y_multi = df['cena']

# Model regresji wielorakiej
model_multi = LinearRegression()
model_multi.fit(X_multi, y_multi)

# Współczynniki
print("\n" + "-"*70)
print("WSPÓŁCZYNNIKI MODELU:")
print("-"*70)
print(f"β₀ (intercept):     {model_multi.intercept_:,.2f} zł")
print(f"β₁ (powierzchnia):  {model_multi.coef_[0]:,.2f} zł/m²")
print(f"β₂ (liczba_pokoi):  {model_multi.coef_[1]:,.2f} zł/pokój")
print(f"β₃ (wiek):          {model_multi.coef_[2]:,.2f} zł/rok")

print(f"\nRównanie regresji:")
print(f"cena = {model_multi.intercept_:,.0f} + "
      f"{model_multi.coef_[0]:.0f}×powierzchnia + "
      f"{model_multi.coef_[1]:.0f}×pokoje + "
      f"{model_multi.coef_[2]:.0f}×wiek")

# Przewidywania
y_pred_multi = model_multi.predict(X_multi)

# Metryki
r2_multi = r2_score(y_multi, y_pred_multi)
rmse_multi = np.sqrt(mean_squared_error(y_multi, y_pred_multi))
mae_multi = mean_absolute_error(y_multi, y_pred_multi)

# Adjusted R²
n = len(y_multi)
p = X_multi.shape[1]
r2_adj = 1 - ((1 - r2_multi) * (n - 1) / (n - p - 1))

print("\n" + "-"*70)
print("METRYKI JAKOŚCI MODELU:")
print("-"*70)
print(f"R²:               {r2_multi:.4f} ({r2_multi*100:.2f}%)")
print(f"Adjusted R²:      {r2_adj:.4f} ({r2_adj*100:.2f}%)")
print(f"RMSE:             {rmse_multi:,.2f} zł")
print(f"MAE:              {mae_multi:,.2f} zł")

# Przykładowe przewidywanie
print("\n" + "-"*70)
print("PRZYKŁADOWE PRZEWIDYWANIE:")
print("-"*70)
przyklad = pd.DataFrame({
    'powierzchnia': [60, 80, 120],
    'liczba_pokoi': [3, 4, 5],
    'wiek': [10, 5, 20]
})

przewidywania = model_multi.predict(przyklad)

for i in range(len(przyklad)):
    print(f"\nMieszkanie {i+1}:")
    print(f"  Powierzchnia: {przyklad.iloc[i]['powierzchnia']} m²")
    print(f"  Pokoje: {przyklad.iloc[i]['liczba_pokoi']}")
    print(f"  Wiek: {przyklad.iloc[i]['wiek']} lat")
    print(f"  Przewidywana cena: {przewidywania[i]:,.2f} zł")

print("="*70)
