import matplotlib.pyplot as plt

x = [1,2,3,4,5,1,3]
y = [2,4,3,6,1]

# plt.plot(x,y)
# plt.show()
#
# plt.scatter(x,y)
# plt.show()

# plt.bar(x,y)
# plt.show()

plt.figure(figsize=(10,5))
plt.hist(x, bins=9, color="red", edgecolor="black", alpha=0.6, density=True)
plt.title("Tytuł wykresu")
plt.xlabel("Oś x")
plt.ylabel("Oś y")
plt.savefig("wykres.png", dpi=300)
plt.show()


# plt.pie(x)
# plt.show()