import urllib.request
import ssl
import certifi

url = "https://physionet.org/files/bidsleep-dataset/1.0.0/Bidslab00/1/labels.mat"

context = ssl.create_default_context(cafile=certifi.where())

request = urllib.request.Request(
    url,
    headers={"User-Agent": "Mozilla/5.0"}
)

response = urllib.request.urlopen(request, context=context)

total = int(response.headers.get("Content-Length", 0))
baixado = 0

print(f"Tamanho do arquivo: {total / (1024 * 1024):.2f} MB")

with open("data/raw/amostra/labels.mat", "wb") as arquivo:

    while True:
        bloco = response.read(1024 * 1024)

        if not bloco:
            break

        arquivo.write(bloco)
        baixado += len(bloco)

        if total:
            porcentagem = (baixado / total) * 100
            print(
                f"\rBaixado: {porcentagem:.1f}% "
                f"({baixado / (1024 * 1024):.2f} MB)",
                end=""
            )

print("\nDownload concluído!")