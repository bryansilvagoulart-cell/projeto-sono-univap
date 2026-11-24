import os
import urllib.request
import ssl
import certifi

PARTICIPANTE = "Bidslab01"

BASE_URL = (
    f"https://physionet.org/files/"
    f"bidsleep-dataset/1.0.0/{PARTICIPANTE}/1/"
)

PASTA_DESTINO = f"data/raw/participantes/{PARTICIPANTE}"

os.makedirs(PASTA_DESTINO, exist_ok=True)

arquivos = [
    "hr.csv",
    "motion.csv",
    "labels.mat"
]

context = ssl.create_default_context(cafile=certifi.where())

for nome_arquivo in arquivos:

    url = BASE_URL + nome_arquivo
    caminho = os.path.join(PASTA_DESTINO, nome_arquivo)

    print("=" * 60)
    print(f"Baixando: {PARTICIPANTE}/{nome_arquivo}")
    print(f"Destino: {caminho}")

    try:
        with urllib.request.urlopen(url, context=context) as resposta:

            tamanho = resposta.headers.get("Content-Length")

            if tamanho:
                tamanho = int(tamanho)
                print(f"Tamanho: {tamanho / (1024 * 1024):.2f} MB")

            with open(caminho, "wb") as arquivo:

                total = 0

                while True:
                    bloco = resposta.read(1024 * 1024)

                    if not bloco:
                        break

                    arquivo.write(bloco)
                    total += len(bloco)

                    if tamanho:
                        porcentagem = total / tamanho * 100
                        print(
                            f"\rBaixado: {porcentagem:.1f}%",
                            end=""
                        )

        print("\nDownload concluído!")

    except Exception as erro:
        print(f"\nERRO: {erro}")

print("\n" + "=" * 60)
print("PARTICIPANTE BAIXADO")
print("=" * 60)
print(f"Arquivos salvos em: {PASTA_DESTINO}")