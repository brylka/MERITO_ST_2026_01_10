import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd

wyniki = np.array([65, 70, 75, 80, 80, 85, 85, 85, 90, 95, 100, 150])

# print("STATYSTYKA OPISOWA - PODSTAWOWE MIARY:")
#
# # print("Średnia arytmetyczna: ", end="")
# # print(np.mean(wyniki))
# # print("Średnia arytmetyczna: " + str(np.mean(wyniki)))
# # print("Średnia arytmetyczna:", np.mean(wyniki))
# print(f"Średnia arytmetyczna: {np.mean(wyniki):.2f}")
# print(f"Mediana:              {np.median(wyniki):.2f}")
# print(f"Moda:                 {stats.mode(wyniki, keepdims=True).mode[0]}")
#
# print("MIARY ROZPROSZENIA:")
# print(f"Minimum:              {np.min(wyniki):.2f}")
# print(f"Maksimum:             {np.max(wyniki):.2f}")
# # print(f"Rozstęp:              {np.max(wyniki)-np.min(wyniki):.2f}")
# print(f"Rozstęp:              {np.ptp(wyniki):.2f}")
# print(f"Wariancja:            {np.var(wyniki, ddof=1):.2f}")
# print(f"Odchylenie std:       {np.std(wyniki, ddof=1):.2f}")
#
# print("KWARTYLE:")
# q1 = np.percentile(wyniki, 25, method="midpoint")
# q2 = np.percentile(wyniki, 50, method="midpoint")
# q3 = np.percentile(wyniki, 75, method="midpoint")
# print(f"Q1:                   {q1:.2f}")
# print(f"Q2:                   {q2:.2f}")
# print(f"Q3:                   {q3:.2f}")
# print(f"IQR:                  {q3-q1:.2f}")
#
# d_g = q1 - 1.5 * (q3-q1)
# g_g = q3 + 1.5 * (q3-q1)
# print(f"Dolna granica:        {d_g}")
# print(f"Górna granica:        {g_g}")
#
# outliery = wyniki[(wyniki < d_g) | (wyniki > g_g)]
# if len(outliery) > 0:
#     print(f"Wykryte wartości odstające (outliery): {outliery}")
# else:
#     print("Brak wartości odstających (outlierów).")

# tabela1 = [f"Strudent {i}" for i in range(1,13)]
# print(tabela1)
#
# tabela2 = []
# for i in range(1,13):
#     tabela2.append(f"Strudent {i}")
# print(tabela2)


# df = pd.DataFrame({
#     'student': [f"Strudent {i}" for i in range(1,13)],
#     'wynik': wyniki
# })
#
# print("ANALIZA Z UŻYCIEM PANDASA")
#
# print("Pięć pierwszych rekordów próbki:")
# print(df.head())
#
# print("Informacje o DataFrame:")
# print(df.info())
#
# print("Statystyki:")
# print(df["wynik"].describe().round(2))
#
# print(f"Średnia arytmetyczna: {df["wynik"].mean().round(2)}")
# print(f"Mediana:              {df["wynik"].median().round(2)}")
# print(f"Q1:                   {df["wynik"].quantile(0.25, interpolation = 'midpoint')}")


plt.figure(figsize=(10,5))
plt.hist(wyniki, bins=9, color="red", edgecolor="black", alpha=0.5, density=True)
plt.axvline(np.mean(wyniki), color="blue", linestyle="--", label=f"Średnia: {np.mean(wyniki):.2f}")
plt.axvline(np.median(wyniki), color="black", linestyle="--", label=f"Mediana: {np.median(wyniki):.2f}")
plt.xlabel("Wyniki")
plt.ylabel("Gęstość")
plt.title("Histogram")
plt.legend()
plt.show()
