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



print("\n" + "="*60)
print("TEST T-STUDENTA DLA DWÓCH GRUP NIEZALEŻNYCH")
print("="*60)

# Dane z Zadania 6
grupa_a = np.array([78, 82, 85, 80, 88, 75, 90])
grupa_b = np.array([85, 88, 92, 87, 90, 85, 95])

# Statystyki opisowe
print("\n1. STATYSTYKI OPISOWE:")
print("-" * 60)
print(f"{'':20} {'Grupa A':>15} {'Grupa B':>15} {'Różnica':>15}")
print("-" * 60)
print(f"{'Liczba (n)':20} {len(grupa_a):>15} {len(grupa_b):>15} {''}")
print(f"{'Średnia (x̄)':20} {np.mean(grupa_a):>15.2f} {np.mean(grupa_b):>15.2f} {np.mean(grupa_b)-np.mean(grupa_a):>15.2f}")
print(f"{'Odchylenie std (s)':20} {np.std(grupa_a, ddof=1):>15.2f} {np.std(grupa_b, ddof=1):>15.2f} {''}")
print(f"{'Min':20} {np.min(grupa_a):>15.2f} {np.min(grupa_b):>15.2f} {''}")
print(f"{'Max':20} {np.max(grupa_a):>15.2f} {np.max(grupa_b):>15.2f} {''}")

# Test normalności (Shapiro-Wilk)
print("\n2. SPRAWDZENIE ZAŁOŻEŃ:")
print("-" * 60)
_, p_shapiro_a = stats.shapiro(grupa_a)
_, p_shapiro_b = stats.shapiro(grupa_b)
print(f"Test normalności (Shapiro-Wilk):")
print(f"  Grupa A: p-value = {p_shapiro_a:.4f}", end="")
print("   (normalny)" if p_shapiro_a > 0.05 else " ✗ (nienormalny)")
print(f"  Grupa B: p-value = {p_shapiro_b:.4f}", end="")
print("   (normalny)" if p_shapiro_b > 0.05 else " ✗ (nienormalny)")

# Test jednorodności wariancji (Levene)
_, p_levene = stats.levene(grupa_a, grupa_b)
print(f"\nTest jednorodności wariancji (Levene):")
print(f"  p-value = {p_levene:.4f}", end="")
print("   (wariancje jednorodne)" if p_levene > 0.05 else " ✗ (wariancje niejednorodne)")

# Test t-Studenta (2 próbki niezależne)
t_statistic, p_value = stats.ttest_ind(grupa_a, grupa_b)

# Dla testu jednostronnego dzielimy p-value przez 2
p_value_one_sided = p_value / 2

print("\n3. WYNIKI TESTU:")
print("-" * 60)
print(f"Statystyka t:                 {t_statistic:.4f}")
print(f"P-value (dwustronny):         {p_value:.4f}")
print(f"P-value (jednostronny):       {p_value_one_sided:.4f}")

# Decyzja
alpha = 0.05
print("\n4. DECYZJA (test jednostronny H₁: μ_B > μ_A):")
print("-" * 60)
print(f"Poziom istotności (α):        {alpha}")

if p_value_one_sided < alpha and t_statistic < 0:
    print(f"✓ p-value ({p_value_one_sided:.4f}) < α ({alpha})")
    print("  → ODRZUCAMY H₀")
    print("  → Metoda B jest ISTOTNIE LEPSZA niż metoda A")
else:
    print(f"✗ p-value ({p_value_one_sided:.4f}) ≥ α ({alpha})")
    print("  → NIE ODRZUCAMY H₀")
    print("  → Brak istotnej różnicy między metodami")

# Wielkość efektu (Cohen's d)
pooled_std = np.sqrt(((len(grupa_a)-1)*np.var(grupa_a, ddof=1) +
                       (len(grupa_b)-1)*np.var(grupa_b, ddof=1)) /
                      (len(grupa_a) + len(grupa_b) - 2))
cohens_d = (np.mean(grupa_b) - np.mean(grupa_a)) / pooled_std

print("\n5. WIELKOŚĆ EFEKTU:")
print("-" * 60)
print(f"Cohen's d:                    {cohens_d:.4f}")
if abs(cohens_d) < 0.2:
    effect_size = "mały"
elif abs(cohens_d) < 0.5:
    effect_size = "średni"
elif abs(cohens_d) < 0.8:
    effect_size = "duży"
else:
    effect_size = "bardzo duży"
print(f"Interpretacja:                efekt {effect_size}")

print("="*60)




import matplotlib.pyplot as plt
import seaborn as sns

# Konfiguracja
sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Analiza Testów Statystycznych', fontsize=16, fontweight='bold')

# 1. Test t dla jednej próbki - rozkład próbki
axes[0, 0].hist(baterie, bins=8, edgecolor='black', alpha=0.7, color='skyblue')
axes[0, 0].axvline(np.mean(baterie), color='red', linestyle='--', linewidth=2,
                   label=f'Średnia próbki: {np.mean(baterie):.1f}')
axes[0, 0].axvline(mu_0, color='green', linestyle='--', linewidth=2,
                   label=f'H_0: μ = {mu_0}')
axes[0, 0].set_xlabel('Czas działania [godz.]')
axes[0, 0].set_ylabel('Częstość')
axes[0, 0].set_title('Test t (1 próbka): Rozkład Danych')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Rozkład t-Studenta z zaznaczonymi wartościami krytycznymi
x = np.linspace(-4, 4, 1000)
y = stats.t.pdf(x, df=len(baterie)-1)

axes[0, 1].plot(x, y, 'b-', linewidth=2, label='Rozkład t-Studenta')
axes[0, 1].axvline(t_statistic, color='red', linestyle='--', linewidth=2,
                   label=f't = {t_statistic:.3f}')
axes[0, 1].axvline(-t_critical, color='green', linestyle=':', linewidth=2,
                   label=f't_kryt = ±{t_critical:.3f}')
axes[0, 1].axvline(t_critical, color='green', linestyle=':', linewidth=2)

# Obszar odrzucenia
x_left = np.linspace(-4, -t_critical, 100)
x_right = np.linspace(t_critical, 4, 100)
axes[0, 1].fill_between(x_left, stats.t.pdf(x_left, df=len(baterie)-1),
                        alpha=0.3, color='red', label='Obszar odrzucenia')
axes[0, 1].fill_between(x_right, stats.t.pdf(x_right, df=len(baterie)-1),
                        alpha=0.3, color='red')

axes[0, 1].set_xlabel('Wartość t')
axes[0, 1].set_ylabel('Gęstość')
axes[0, 1].set_title(f'Rozkład t-Studenta (df={len(baterie)-1})')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Porównanie dwóch grup - boxploty
bp = axes[1, 0].boxplot([grupa_a, grupa_b], tick_labels=['Grupa A', 'Grupa B'],
                        patch_artist=True,
                        boxprops=dict(facecolor='lightblue'),
                        medianprops=dict(color='red', linewidth=2))
axes[1, 0].set_ylabel('Wyniki [pkt]')
axes[1, 0].set_title('Porównanie Dwóch Grup')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Adnotacje
axes[1, 0].text(1, np.mean(grupa_a), f'x̄={np.mean(grupa_a):.1f}',
               fontsize=10, ha='right')
axes[1, 0].text(2, np.mean(grupa_b), f'x̄={np.mean(grupa_b):.1f}',
               fontsize=10, ha='left')

# 4. Porównanie rozkładów
axes[1, 1].hist(grupa_a, bins=6, alpha=0.5, label='Grupa A',
               edgecolor='black', color='blue', density=True)
axes[1, 1].hist(grupa_b, bins=6, alpha=0.5, label='Grupa B',
               edgecolor='black', color='orange', density=True)
axes[1, 1].axvline(np.mean(grupa_a), color='blue', linestyle='--', linewidth=2)
axes[1, 1].axvline(np.mean(grupa_b), color='orange', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('Wyniki [pkt]')
axes[1, 1].set_ylabel('Gęstość')
axes[1, 1].set_title('Rozkłady Dwóch Grup')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('testy_statystyczne_wizualizacje.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nWizualizacja została zapisana jako 'testy_statystyczne_wizualizacje.png'")
