import numpy as np
from scipy import stats

wyniki = np.array([65, 70, 75, 80, 80, 85, 85, 85, 90, 95, 100, 150])

print("STATYSTYKA OPISOWA - PODSTAWOWE MIARY:")

# print("Średnia arytmetyczna: ", end="")
# print(np.mean(wyniki))
# print("Średnia arytmetyczna: " + str(np.mean(wyniki)))
# print("Średnia arytmetyczna:", np.mean(wyniki))
print(f"Średnia arytmetyczna: {np.mean(wyniki):.2f}")
print(f"Mediana:              {np.median(wyniki):.2f}")
print(f"Moda:                 {stats.mode(wyniki, keepdims=True).mode[0]}")

print("MIARY ROZPROSZENIA:")
print(f"Minimum:              {np.min(wyniki):.2f}")
print(f"Maksimum:             {np.max(wyniki):.2f}")
# print(f"Rozstęp:              {np.max(wyniki)-np.min(wyniki):.2f}")
print(f"Rozstęp:              {np.ptp(wyniki):.2f}")
print(f"Wariancja:            {np.var(wyniki, ddof=1):.2f}")
print(f"Odchylenie std:       {np.std(wyniki, ddof=1):.2f}")