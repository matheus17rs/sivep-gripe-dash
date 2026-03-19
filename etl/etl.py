import os
import pandas as pd
import numpy as np
from scipy import stats as sp_stats

BASE_DIR    = r"D:\Users\mathe\Arquivos\GSK\SIVEP_Gripe"
CAMINHO_CSV = os.path.join(BASE_DIR, "data", "raw", "srag_2026.csv")
DIR_OUT     = os.path.join(BASE_DIR, "data", "processed")


def carregar_raw(caminho: str) -> pd.DataFrame:
    try:
        return pd.read_csv(caminho, sep=";", encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(caminho, sep=";", encoding="latin1", low_memory=False)


def calcular_idade_anos(df: pd.DataFrame) -> pd.Series:
    idade = pd.to_numeric(df["NU_IDADE_N"], errors="coerce")
    tipo  = pd.to_numeric(df["TP_IDADE"],   errors="coerce")
    return pd.Series(
        np.where(tipo == 1, idade / 365,
        np.where(tipo == 2, idade / 12,
        np.where(tipo == 3, idade, np.nan))),
        index=df.index
    )


def faixa_etaria(idade_anos: pd.Series) -> pd.Series:
    bins   = [0, 10, 20, 30, 40, 50, 60, 70, 80, np.inf]
    labels = ["0-9","10-19","20-29","30-39","40-49","50-59","60-69","70-79","80+"]
    return pd.cut(idade_anos, bins=bins, labels=labels, right=False)


def col_sim(serie: pd.Series) -> pd.Series:
    """Retorna True onde o valor é exatamente 1 (inteiro ou float), ignora NaN, 2, 9."""
    return pd.to_numeric(serie, errors="coerce").fillna(0).astype(int) == 1


def confirmar_vsr(df: pd.DataFrame) -> pd.Series:
    return (col_sim(df["AN_VSR"]) | col_sim(df["PCR_VSR"])).astype(int)


def confirmar_influenza(df: pd.DataFrame) -> pd.Series:
    classi = pd.to_numeric(df["CLASSI_FIN"], errors="coerce").fillna(0).astype(int)
    return (classi == 1).astype(int)


def confirmar_outro(df: pd.DataFrame) -> pd.Series:
    """
    Outros patógenos confirmados:
    - CLASSI_FIN preenchido (não vazio / não zero)
    - E não é Influenza (CLASSI_FIN != 1)
    - E não é VSR (AN_VSR != 1 e PCR_VSR != 1)
    """
    classi         = pd.to_numeric(df["CLASSI_FIN"], errors="coerce")
    tem_classi_fin = classi.notna() & (classi > 0)
    nao_influenza  = classi != 1
    nao_vsr        = confirmar_vsr(df).astype(bool) == False
    return (tem_classi_fin & nao_influenza & nao_vsr).astype(int)


def processar_colunas_binarias(df: pd.DataFrame, mapa_colunas: dict) -> pd.Series:
    """
    Recebe o df e um dicionário {coluna_raw: label}.
    Retorna uma Series com a contagem de casos por label (valor == 1).
    """
    existentes = {k: v for k, v in mapa_colunas.items() if k in df.columns}
    contagens = {}
    for col, label in existentes.items():
        contagens[label] = int(col_sim(df[col]).sum())
    return pd.Series(contagens).sort_values(ascending=False)


def top7_com_outros(serie_contagem: pd.Series, col_nome: str) -> pd.DataFrame:
    top7  = serie_contagem.nlargest(7)
    resto = serie_contagem.drop(top7.index).sum()
    if resto > 0:
        top7["Outros"] = top7.get("Outros", 0) + resto
    return (
        top7.reset_index()
            .rename(columns={"index": col_nome, 0: "total"})
    )

def gerar_perfis(df: pd.DataFrame, mascara: pd.Series = None) -> dict:
    """
    Gera os 4 dataframes de perfil (faixa etária, sexo, sintomas, fatores)
    para um subconjunto do df definido pela máscara booleana.
    Se mascara for None, usa todos os registros.
    """
    sub = df[mascara] if mascara is not None else df

    # Faixa etária
    fe = (
        sub["faixa_etaria"]
        .value_counts()
        .reindex(["0-9","10-19","20-29","30-39","40-49","50-59","60-69","70-79","80+"])
        .fillna(0).astype(int)
        .reset_index()
        .rename(columns={"index": "faixa_etaria", "faixa_etaria": "total"})
    )

    # Sexo
    sx = (
        sub["sexo_label"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "sexo", "sexo_label": "total"})
    )

    # Sintomas
    mapa_sintomas = {
        "FEBRE":"Febre","TOSSE":"Tosse","GARGANTA":"Dor de Garganta",
        "DISPNEIA":"Dispneia","DESC_RESP":"Desconforto Resp.",
        "SATURACAO":"Saturação O2 <95%","DIARREIA":"Diarreia",
        "VOMITO":"Vômito","DOR_ABD":"Dor Abdominal","FADIGA":"Fadiga",
        "PERD_OLFT":"Perda de Olfato","PERD_PALA":"Perda de Paladar",
        "OUTRO_SIN":"Outros Sintomas",
    }
    sint_counts = processar_colunas_binarias(sub, mapa_sintomas)
    sint        = top7_com_outros(sint_counts, "sintoma")

    # Fatores de risco
    mapa_fr = {
        "CARDIOPATI":"Doença Cardiovascular","DIABETES":"Diabetes",
        "OBESIDADE":"Obesidade","IMUNODEPRE":"Imunodeficiência",
        "RENAL":"Doença Renal","NEUROLOGIC":"Doença Neurológica",
        "PNEUMOPATI":"Pneumopatia","ASMA":"Asma","HEPATICA":"Doença Hepática",
        "HEMATOLOGI":"Doença Hematológica","SIND_DOWN":"Síndrome de Down",
        "PUERPERA":"Puérpera","TABAG":"Tabagismo","OUT_MORBI":"Outros",
    }
    fr_counts = processar_colunas_binarias(sub, mapa_fr)
    fr        = top7_com_outros(fr_counts, "fator_risco")

    return {"faixa_etaria": fe, "sexo": sx, "sintomas": sint, "fatores_risco": fr}

def calcular_tendencia_uf(evolucao: pd.DataFrame, coluna: str = "total_srag") -> pd.DataFrame:
    """
    Regressão linear das últimas 6 semanas por UF.
    Retorna DataFrame com colunas: uf_notificacao, slope
    """

    semanas  = sorted(evolucao["sem_label"].dropna().unique())
    ultimas6 = semanas[-6:] if len(semanas) >= 6 else semanas
    df6      = evolucao[evolucao["sem_label"].isin(ultimas6)].copy()
    sem_idx  = {s: i for i, s in enumerate(ultimas6)}
    df6["sem_idx"] = df6["sem_label"].map(sem_idx)

    resultados = []
    for uf, grupo in df6.groupby("uf_notificacao"):
        grupo = grupo.sort_values("sem_idx")
        if len(grupo) >= 3:
            slope, _, _, _, _ = sp_stats.linregress(
                grupo["sem_idx"], grupo[coluna].fillna(0)
            )
        else:
            slope = 0.0
        resultados.append({"uf_notificacao": uf, "slope": round(slope, 2)})

    return pd.DataFrame(resultados)

def executar_etl(caminho_csv: str):
    print("1/7 Carregando dados brutos...")
    df_raw = carregar_raw(caminho_csv)
    print(f"   {df_raw.shape[0]:,} linhas | {df_raw.shape[1]} colunas")

    print("2/7 Calculando campos derivados...")
    # Consolida todas as colunas novas de uma vez para evitar fragmentação
    novos = pd.DataFrame({
        "idade_anos":   calcular_idade_anos(df_raw),
        "is_vsr":       confirmar_vsr(df_raw),
        "is_influenza": confirmar_influenza(df_raw),
        "is_outro":     confirmar_outro(df_raw),
        "sexo_label":   df_raw["CS_SEXO"].astype(str).str.strip()
                            .map({"M": "Masculino", "F": "Feminino", "I": "Ignorado"})
                            .fillna("Ignorado"),
        # SEM_PRI é inteiro (ex: 3 = SE03 de 2026) — formata como "2026-SE03"
        "sem_label":    df_raw["SEM_PRI"].apply(
                            lambda x: f"2026-SE{int(x):02d}"
                            if pd.notna(x) and str(x).strip() not in ["", "nan"]
                            else None
                        ),
    }, index=df_raw.index)

    novos["faixa_etaria"] = faixa_etaria(novos["idade_anos"])
    df = pd.concat([df_raw, novos], axis=1).copy()  # .copy() desfragmenta

    # ── Tabela 1: Visão Geral por UF ─────────────────────────────────────────
    print("3/7 Gerando tabela Visão Geral por UF...")
    visao_geral = (
        df.groupby("SG_UF_NOT", dropna=False)
        .agg(
            total_srag      =("NU_NOTIFIC",   "count"),
            casos_vsr       =("is_vsr",        "sum"),
            casos_influenza =("is_influenza",  "sum"),
            casos_outro     =("is_outro",      "sum"),
        )
        .reset_index()
        .rename(columns={"SG_UF_NOT": "uf_notificacao"})
        .sort_values("total_srag", ascending=False)
    )

    # ── Tabela 2: Evolução semanal ────────────────────────────────────────────
    print("4/7 Gerando série temporal semanal...")
    evolucao_semanal = (
        df.groupby(["SG_UF_NOT", "sem_label"], dropna=False)
        .agg(
            total_srag=("NU_NOTIFIC", "count"),
            casos_vsr =("is_vsr",     "sum"),
            casos_influenza =("is_influenza", "sum"),
        )
        .reset_index()
        .rename(columns={"SG_UF_NOT": "uf_notificacao"})
        .sort_values(["uf_notificacao", "sem_label"])
    )

    # ── Tabela 3: Faixa etária ────────────────────────────────────────────────
    print("5/7 Gerando perfil etário...")
    faixa_etaria_df = (
        df["faixa_etaria"]
        .value_counts()
        .reindex(["0-9","10-19","20-29","30-39","40-49","50-59","60-69","70-79","80+"])
        .fillna(0).astype(int)
        .reset_index()
        .rename(columns={"index": "faixa_etaria", "faixa_etaria": "total"})
    )

    # ── Tabela 4: Sexo ────────────────────────────────────────────────────────
    sexo_df = (
        df["sexo_label"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "sexo", "sexo_label": "total"})
    )

    # ── Tabela 5: Sintomas ────────────────────────────────────────────────────
    print("6/7 Gerando perfis de sintomas e fatores de risco...")
    mapa_sintomas = {
        "FEBRE":     "Febre",         "TOSSE":     "Tosse",
        "GARGANTA":  "Dor de Garganta","DISPNEIA":  "Dispneia",
        "DESC_RESP": "Desconforto Resp.","SATURACAO":"Saturação O2 <95%",
        "DIARREIA":  "Diarreia",      "VOMITO":    "Vômito",
        "DOR_ABD":   "Dor Abdominal", "FADIGA":    "Fadiga",
        "PERD_OLFT": "Perda de Olfato","PERD_PALA":"Perda de Paladar",
        "OUTRO_SIN": "Outros Sintomas",
    }
    sint_counts = processar_colunas_binarias(df, mapa_sintomas)
    sintomas_df = top7_com_outros(sint_counts, "sintoma")

    # ── Tabela 6: Fatores de risco ────────────────────────────────────────────
    mapa_fr = {
        "CARDIOPATI": "Doença Cardiovascular","DIABETES":  "Diabetes",
        "OBESIDADE":  "Obesidade",            "IMUNODEPRE":"Imunodeficiência",
        "RENAL":      "Doença Renal",         "NEUROLOGIC":"Doença Neurológica",
        "PNEUMOPATI": "Pneumopatia",           "ASMA":      "Asma",
        "HEPATICA":   "Doença Hepática",      "HEMATOLOGI":"Doença Hematológica",
        "SIND_DOWN":  "Síndrome de Down",     "PUERPERA":  "Puérpera",
        "TABAG":      "Tabagismo",            "OUT_MORBI": "Outros",
    }
    fr_counts  = processar_colunas_binarias(df, mapa_fr)
    fatores_df = top7_com_outros(fr_counts, "fator_risco")

    # ── Salva Parquets ────────────────────────────────────────────────────────
    print("7/7 Salvando Parquets...")
    os.makedirs(DIR_OUT, exist_ok=True)
    visao_geral.to_parquet(     os.path.join(DIR_OUT, "visao_geral.parquet"),      index=False)
    evolucao_semanal.to_parquet(os.path.join(DIR_OUT, "evolucao_semanal.parquet"), index=False)
    for sufixo, mascara in [
        ("srag",      None),                          # todos os casos
        ("vsr",       df["is_vsr"]       == 1),       # só VSR
        ("influenza", df["is_influenza"] == 1),       # só Influenza
    ]:
        perfis = gerar_perfis(df, mascara)
        perfis["faixa_etaria"].to_parquet( os.path.join(DIR_OUT, f"faixa_etaria_{sufixo}.parquet"), index=False)
        perfis["sexo"].to_parquet(         os.path.join(DIR_OUT, f"sexo_{sufixo}.parquet"),         index=False)
        perfis["sintomas"].to_parquet(     os.path.join(DIR_OUT, f"sintomas_{sufixo}.parquet"),     index=False)
        perfis["fatores_risco"].to_parquet(os.path.join(DIR_OUT, f"fatores_risco_{sufixo}.parquet"),index=False)
    calcular_tendencia_uf(evolucao_semanal, "total_srag").to_parquet(
        os.path.join(DIR_OUT, "tendencia_srag.parquet"), index=False)
    calcular_tendencia_uf(
        evolucao_semanal[["uf_notificacao","sem_label","casos_vsr"]]
        .rename(columns={"casos_vsr":"total_srag"}), "total_srag"
    ).to_parquet(os.path.join(DIR_OUT, "tendencia_vsr.parquet"), index=False)
    calcular_tendencia_uf(
        evolucao_semanal[["uf_notificacao","sem_label","casos_influenza"]]
        .rename(columns={"casos_influenza":"total_srag"}), "total_srag"
    ).to_parquet(os.path.join(DIR_OUT, "tendencia_influenza.parquet"), index=False)

    print("✓ ETL concluído!")
    return {
        "visao_geral": visao_geral, "evolucao_semanal": evolucao_semanal,
        "faixa_etaria": faixa_etaria_df, "sexo": sexo_df,
        "sintomas": sintomas_df, "fatores_risco": fatores_df,
    }


# ── Execução ──────────────────────────────────────────────────────────────────
tabelas = executar_etl(CAMINHO_CSV)

for nome, df_out in tabelas.items():
    print(f"{nome:20s} → {df_out.shape[0]} linhas | {df_out.shape[1]} colunas")
    print(df_out.head(3).to_string(), "\n")