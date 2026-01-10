import numpy as np

wyniki = np.array([65, 70, 75, 80, 80, 85, 85, 85, 90, 95, 100, 150])

print("STATYSTYKA OPISOWA - PODSTAWOWE MIARY:")

# print("Średnia arytmetyczna: ", end="")
# print(np.mean(wyniki))
# print("Średnia arytmetyczna: " + str(np.mean(wyniki)))
# print("Średnia arytmetyczna:", np.mean(wyniki))
print(f"Średnia arytmetyczna: {np.mean(wyniki):.2f}")
