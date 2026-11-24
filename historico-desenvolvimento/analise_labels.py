import scipy.io
import numpy as np

arquivo = "data/raw/amostra/labels.mat"

dados = scipy.io.loadmat(arquivo)

dreem = dados["dreem_label"].flatten()
expert = dados["expert_label"].flatten()

print("Quantidade de épocas:")
print("Dreem:", len(dreem))
print("Expert:", len(expert))

print("\nValores existentes em dreem_label:")
print(np.unique(dreem))

print("\nValores existentes em expert_label:")
print(np.unique(expert))

print("\nContagem dos valores - dreem_label:")
valores, contagens = np.unique(dreem, return_counts=True)

for valor, quantidade in zip(valores, contagens):
    porcentagem = quantidade / len(dreem) * 100
    print(
        f"Label {valor}: "
        f"{quantidade} épocas "
        f"({porcentagem:.2f}%)"
    )

print("\nContagem dos valores - expert_label:")
valores, contagens = np.unique(expert, return_counts=True)

for valor, quantidade in zip(valores, contagens):
    porcentagem = quantidade / len(expert) * 100
    print(
        f"Label {valor}: "
        f"{quantidade} épocas "
        f"({porcentagem:.2f}%)"
    )