import requests
import pandas as pd
from tqdm import tqdm
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# URL direta do arquivo no S3 (visível na página do recurso)
URL_S3 = "https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SRAG/2026/INFLUD26-16-03-2026.csv"
NOME_ARQUIVO = "srag_2026.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
}

def baixar_arquivo(url: str, nome_arquivo: str) -> bool:
    print(f"Iniciando download de:\n  {url}\n")
    try:
        resp = requests.get(url, stream=True, headers=HEADERS, verify=False, timeout=300)
        resp.raise_for_status()

        tamanho_total = int(resp.headers.get("content-length", 0))

        with open(nome_arquivo, "wb") as f, tqdm(
            desc=nome_arquivo,
            total=tamanho_total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in resp.iter_content(1024 * 1024):
                bar.update(len(chunk))
                f.write(chunk)

        print(f"\n✓ Download concluído: {nome_arquivo}")
        return True

    except Exception as e:
        print(f"Erro durante o download: {e}")
        return False


# ── Execução ──────────────────────────────────────────────────────────────────
if baixar_arquivo(URL_S3, NOME_ARQUIVO):
    print("\nCarregando os dados no Pandas...")
    try:
        df = pd.read_csv(NOME_ARQUIVO, sep=";", encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(NOME_ARQUIVO, sep=";", encoding="latin1", low_memory=False)

    print(f"✓ Base carregada! Linhas: {df.shape[0]:,} | Colunas: {df.shape[1]}")
    print(df.head())