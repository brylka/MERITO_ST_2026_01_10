from scipy import stats
import numpy as np

print("="*60)
print("TEST T-STUDENTA DLA JEDNEJ PRÓBKI")
print("="*60)

# Dane z Zadania 5
baterie = np.array([98, 102, 101, 97, 103, 99, 100, 101, 98, 102])
mu_0 = 100  # wartość z hipotezy H₀

# Statystyki opisowe
print("\n1. STATYSTYKI OPISOWE:")
print("-" * 60)
print(f"Liczba obserwacji (n):        {len(baterie)}")
print(f"Średnia próbki (x̄):           {np.mean(baterie):.4f} godz.")
print(f"Odchylenie std próbki (s):    {np.std(baterie, ddof=1):.4f} godz.")
print(f"Wartość testowana (μ₀):       {mu_0} godz.")

# Test t-Studenta (1 próbka)
t_statistic, p_value = stats.ttest_1samp(baterie, mu_0)

# Wyniki testu
print("\n2. WYNIKI TESTU:")
print("-" * 60)
print(f"Statystyka t:                 {t_statistic:.4f}")
print(f"P-value (dwustronny):         {p_value:.4f}")
print(f"Stopnie swobody (df):         {len(baterie)-1}")

# Wartość krytyczna
alpha = 0.05
df = len(baterie) - 1
t_critical = stats.t.ppf(1 - alpha/2, df)
print(f"Wartość krytyczna t_α/2:      ±{t_critical:.4f}")

# Decyzja
print("\n3. DECYZJA:")
print("-" * 60)
print(f"Poziom istotności (α):        {alpha}")

if p_value < alpha:
    print(f"p-value ({p_value:.4f}) < α ({alpha})")
    print("  - ODRZUCAMY H₀")
    print("  - Średnia jest istotnie różna od 100 godz.")
else:
    print(f"p-value ({p_value:.4f}) ≥ α ({alpha})")
    print("  - NIE ODRZUCAMY H₀")
    print("  - Brak podstaw do odrzucenia twierdzenia producenta")

# Przedział ufności
confidence_level = 0.95
ci = stats.t.interval(confidence_level, df,
                      loc=np.mean(baterie),
                      scale=stats.sem(baterie))

print("\n4. PRZEDZIAŁ UFNOŚCI:")
print("-" * 60)
print(f"95% przedział ufności:        [{ci[0]:.4f}, {ci[1]:.4f}] godz.")
print(f"Czy {mu_0} należy do przedziału? ", end="")
if ci[0] <= mu_0 <= ci[1]:
    print("TAK")
else:
    print("NIE")

print("="*60)
