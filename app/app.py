import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import requests
from scipy import stats

# ── Configuração da página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="SRAG & VSR | Vigilância Epidemiológica",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paleta ────────────────────────────────────────────────────────────────────
LARANJA        = "#E8600A"
LARANJA_CLARO  = "#F28C3A"
LARANJA_ESCURO = "#B84A06"
FUNDO_ESCURO   = "#1A1208"
FUNDO_CARD     = "#231A0E"
FUNDO_SIDEBAR  = "#1E1309"
TEXTO_CLARO    = "#F5E6D3"
TEXTO_MUTED    = "#A8876A"
CINZA_LINHA    = "#3D2E1E"
AMARELO_ACC    = "#F5B841"
AZUL_LINHA     = "#4A9EBF"
VERDE_LINHA    = "#5DBF7A"

# ── CSS global ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background-color: {FUNDO_ESCURO};
    color: {TEXTO_CLARO};
}}
section[data-testid="stSidebar"] {{
    background-color: {FUNDO_SIDEBAR} !important;
    border-right: 1px solid {CINZA_LINHA};
}}
section[data-testid="stSidebar"] * {{ color: {TEXTO_CLARO} !important; }}
.stApp {{ background-color: {FUNDO_ESCURO}; }}
.main .block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; }}

.metric-card {{
    background: {FUNDO_CARD};
    border: 1px solid {CINZA_LINHA};
    border-top: 3px solid {LARANJA};
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.5rem;
}}
.metric-card .label {{
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: {TEXTO_MUTED};
    margin-bottom: 0.3rem;
}}
.metric-card .value {{
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: {LARANJA};
    line-height: 1;
}}
.metric-card .sub {{
    font-size: 0.72rem;
    color: {TEXTO_MUTED};
    margin-top: 0.2rem;
}}
.section-title {{
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: {TEXTO_CLARO};
    letter-spacing: 0.03em;
    margin-bottom: 0.2rem;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid {LARANJA};
    display: inline-block;
}}
.section-subtitle {{
    font-size: 0.78rem;
    color: {TEXTO_MUTED};
    margin-bottom: 1rem;
}}
.app-header {{
    background: linear-gradient(135deg, {LARANJA_ESCURO} 0%, {LARANJA} 60%, {LARANJA_CLARO} 100%);
    border-radius: 10px;
    padding: 1.4rem 2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}}
.app-header h1 {{
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 600;
    color: #fff;
    margin: 0;
    line-height: 1.1;
    letter-spacing: normal;
    font-stretch: normal;
    transform: none;
}}
.app-header p {{
    font-size: 0.8rem;
    color: rgba(255,255,255,0.75);
    margin: 0.2rem 0 0 0;
}}
.divider {{
    height: 1px;
    background: {CINZA_LINHA};
    margin: 1.8rem 0 1.4rem 0;
}}
.styled-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
}}
.styled-table th {{
    background: {LARANJA_ESCURO};
    color: #fff;
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 0.6rem 0.9rem;
    text-align: left;
}}
.styled-table td {{
    padding: 0.5rem 0.9rem;
    border-bottom: 1px solid {CINZA_LINHA};
    color: {TEXTO_CLARO};
}}
.styled-table tr:nth-child(even) td {{ background: rgba(255,255,255,0.03); }}
.styled-table tr:hover td {{ background: rgba(232,96,10,0.08); }}
.styled-table td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.styled-table td.pct {{
    text-align: left;
    font-size: 0.72rem;
    color: {TEXTO_MUTED};
    font-variant-numeric: tabular-nums;
    padding-left: 0.3rem;
}}
.styled-table td.uf {{
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    color: {LARANJA_CLARO};
    letter-spacing: normal;
    font-stretch: normal;
    transform: none;
}}
div[data-baseweb="select"] > div {{
    background-color: {FUNDO_CARD} !important;
    border-color: {CINZA_LINHA} !important;
    color: {TEXTO_CLARO} !important;
}}
.legend-box {{
    background: {FUNDO_CARD};
    border: 1px solid {CINZA_LINHA};
    border-radius: 8px;
    padding: 1rem;
    margin-top: 0.5rem;
}}
.legend-item {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.4rem;
    font-size: 0.78rem;
    color: {TEXTO_CLARO};
}}
.legend-dot {{
    width: 14px;
    height: 14px;
    border-radius: 3px;
    flex-shrink: 0;
}}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def layout_plotly(fig, height=340):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=TEXTO_CLARO, size=11),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXTO_CLARO, size=10)),
        xaxis=dict(gridcolor=CINZA_LINHA, linecolor=CINZA_LINHA, tickfont=dict(color=TEXTO_MUTED)),
        yaxis=dict(gridcolor=CINZA_LINHA, linecolor=CINZA_LINHA, tickfont=dict(color=TEXTO_MUTED)),
    )
    return fig


@st.cache_data
def carregar_dados(base_dir: str) -> dict:
    proc = os.path.join(base_dir, "data", "processed")
    d = {
        "visao_geral":      pd.read_parquet(os.path.join(proc, "visao_geral.parquet")),
        "evolucao_semanal": pd.read_parquet(os.path.join(proc, "evolucao_semanal.parquet")),
    }
    # Perfis por subgrupo
    for sufixo in ["srag", "vsr", "influenza"]:
        for tabela in ["faixa_etaria", "sexo", "sintomas", "fatores_risco"]:
            chave = f"{tabela}_{sufixo}"
            p     = os.path.join(proc, f"{chave}.parquet")
            if os.path.exists(p):
                d[chave] = pd.read_parquet(p)
    # Tendências
    for nome in ["tendencia_srag", "tendencia_vsr", "tendencia_influenza"]:
        p = os.path.join(proc, f"{nome}.parquet")
        if os.path.exists(p):
            d[nome] = pd.read_parquet(p)
    return d


@st.cache_data
def carregar_geojson():
    url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson"
    try:
        r = requests.get(url, timeout=15)
        return r.json()
    except Exception:
        return None


def calcular_tendencia_on_the_fly(evolucao: pd.DataFrame, coluna: str = "total_srag") -> pd.DataFrame:
    semanas   = sorted(evolucao["sem_label"].dropna().unique())
    ultimas6  = semanas[-6:] if len(semanas) >= 6 else semanas
    df6       = evolucao[evolucao["sem_label"].isin(ultimas6)].copy()
    sem_idx   = {s: i for i, s in enumerate(ultimas6)}
    df6["sem_idx"] = df6["sem_label"].map(sem_idx)

    resultados = []
    for uf, grupo in df6.groupby("uf_notificacao"):
        grupo = grupo.sort_values("sem_idx")
        if len(grupo) >= 3:
            slope, _, _, _, _ = stats.linregress(grupo["sem_idx"], grupo[coluna].fillna(0))
        else:
            slope = 0.0
        resultados.append({"uf_notificacao": uf, "slope": round(slope, 2)})
    return pd.DataFrame(resultados)


SIGLA_PARA_NOME = {
    "AC":"Acre","AL":"Alagoas","AP":"Amapá","AM":"Amazonas","BA":"Bahia",
    "CE":"Ceará","DF":"Distrito Federal","ES":"Espírito Santo","GO":"Goiás",
    "MA":"Maranhão","MT":"Mato Grosso","MS":"Mato Grosso do Sul","MG":"Minas Gerais",
    "PA":"Pará","PB":"Paraíba","PR":"Paraná","PE":"Pernambuco","PI":"Piauí",
    "RJ":"Rio de Janeiro","RN":"Rio Grande do Norte","RS":"Rio Grande do Sul",
    "RO":"Rondônia","RR":"Roraima","SC":"Santa Catarina","SP":"São Paulo",
    "SE":"Sergipe","TO":"Tocantins",
}

# ── Carrega dados ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dados    = carregar_dados(BASE_DIR)
geojson  = carregar_geojson()

vg  = dados["visao_geral"]
es  = dados["evolucao_semanal"]

ultima_semana   = es["sem_label"].dropna().max() if not es.empty else "N/D"
total_srag      = int(vg["total_srag"].sum())
total_vsr       = int(vg["casos_vsr"].sum())
total_influenza = int(vg["casos_influenza"].sum())
total_outro     = int(vg["casos_outro"].sum())
pct_vsr         = total_vsr       / total_srag * 100 if total_srag else 0
pct_inf         = total_influenza / total_srag * 100 if total_srag else 0
pct_out         = total_outro     / total_srag * 100 if total_srag else 0


# ── Sidebar ───────────────────────────────────────────────────────────────────
# DEPOIS
with st.sidebar:
    st.markdown(f"""
    <div style='margin-bottom:1.5rem'>
        <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:600;color:{LARANJA}'>
            🫁 SIVEP-Gripe
        </div>
        <div style='font-size:0.74rem;color:{TEXTO_MUTED};margin-top:0.2rem'>
            Vigilância Epidemiológica SRAG 2026
        </div>
    </div>
    <div style='margin-top:1rem;padding-top:1rem;border-top:1px solid {CINZA_LINHA};
                font-size:0.70rem;color:{TEXTO_MUTED};line-height:1.8'>
        Fonte: SIVEP-Gripe / OpenDataSUS<br>
        Última Semana Epidemiológica disponível: <b style='color:{LARANJA_CLARO}'>{ultima_semana}</b>
    </div>
    <div style='margin-top:1rem;padding-top:1rem;border-top:1px solid {CINZA_LINHA};
                font-size:0.70rem;color:{TEXTO_MUTED};line-height:1.8'>
        <b style='color:{TEXTO_CLARO}'>Projeto de Exploração de Dados Públicos</b><br>
        Objetivo: Prova de Conceito<br>
        Autor: Matheus Rodrigues
    </div>
    <div style='margin-top:1rem;padding:0.8rem;background:{FUNDO_CARD};
                border-left:3px solid {LARANJA};border-radius:4px;
                font-size:0.65rem;color:{TEXTO_MUTED};line-height:1.6'>
        ⚠ Os dados contidos nesse dash estão sujeitos à revisão, considere esse ponto ao fazer análises a partir dos mesmos.
    </div>
    """, unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="app-header">
    <div>🫁</div>
    <div>
        <h1>Vigilância de SRAG e VSR — Brasil 2026</h1>
        <p>Síndrome Respiratória Aguda Grave · Vírus Sincicial Respiratório · SIVEP-Gripe</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ── KPIs ──────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-card">
        <div class="label">Total de Casos SRAG Notificados</div>
        <div class="value">{total_srag:,}</div>
        <div class="sub">Notificados em 2026</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card">
        <div class="label">Confirmados VSR</div>
        <div class="value" style="color:{LARANJA_CLARO}">{total_vsr:,}</div>
        <div class="sub">{pct_vsr:.1f}% do total de SRAG notificados</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card">
        <div class="label">Confirmados Influenza</div>
        <div class="value" style="color:{AMARELO_ACC}">{total_influenza:,}</div>
        <div class="sub">{pct_inf:.1f}% do total de SRAG notificados</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card">
        <div class="label">Outros Patógenos Confirmados</div>
        <div class="value" style="color:{TEXTO_MUTED}">{total_outro:,}</div>
        <div class="sub">{pct_out:.1f}% — COVID-19, outros vírus e agentes</div>
    </div>""", unsafe_allow_html=True)


# ── Seção 1: Tabela Visão Geral por UF ───────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Visão Geral VSR e SRAG</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Casos notificados por Unidade Federativa em 2026</div>', unsafe_allow_html=True)

totais = pd.DataFrame([{
    "uf_notificacao": "BRASIL",
    "total_srag":      total_srag,
    "casos_vsr":       total_vsr,
    "casos_influenza": total_influenza,
    "casos_outro":     total_outro,
}])
tabela = pd.concat([vg, totais], ignore_index=True)

linhas_html = ""
for _, row in tabela.iterrows():
    uf     = str(row["uf_notificacao"])
    t_srag = int(row["total_srag"])
    t_vsr  = int(row["casos_vsr"])
    t_inf  = int(row["casos_influenza"])
    t_out  = int(row["casos_outro"])
    pv     = t_vsr / t_srag * 100 if t_srag else 0
    pi     = t_inf / t_srag * 100 if t_srag else 0
    po     = t_out / t_srag * 100 if t_srag else 0
    brasil = uf == "BRASIL"
    est_tr = 'style="background:rgba(232,96,10,0.12)"' if brasil else ""
    est_uf = f'style="font-weight:800;color:{"#fff" if brasil else LARANJA_CLARO}"'
    linhas_html += f"""
    <tr {est_tr}>
        <td class="uf" {est_uf}>{uf}</td>
        <td class="num">{t_srag:,}</td>
        <td class="num">{t_vsr:,}</td><td class="pct">({pv:.1f}%)</td>
        <td class="num">{t_inf:,}</td><td class="pct">({pi:.1f}%)</td>
        <td class="num">{t_out:,}</td><td class="pct">({po:.1f}%)</td>
    </tr>"""

st.markdown(f"""
<div style="max-height:420px;overflow-y:auto;border:1px solid {CINZA_LINHA};border-radius:8px">
<table class="styled-table">
    <thead><tr>
        <th>UF</th>
        <th style="text-align:right">Total de Casos SRAG Notificados</th>
        <th style="text-align:right">VSR</th><th>%</th>
        <th style="text-align:right">Influenza</th><th>%</th>
        <th style="text-align:right">Outro Patógeno</th><th>%</th>
    </tr></thead>
    <tbody>{linhas_html}</tbody>
</table>
</div>
""", unsafe_allow_html=True)


# ── Seção 2: Evolução Semanal ─────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Evolução de Casos por Semana Epidemiológica</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Tendência semanal de casos notificados e confirmados</div>', unsafe_allow_html=True)

col_filtro_ev, _ = st.columns([2, 3])
with col_filtro_ev:
    st.markdown(f"<div style='font-size:0.72rem;color:{TEXTO_MUTED};text-transform:uppercase;"
                f"letter-spacing:0.08em;margin-bottom:0.3rem'>Estado de Notificação</div>",
                unsafe_allow_html=True)
    ufs_disponiveis = sorted(es["uf_notificacao"].dropna().unique().tolist())
    uf_selecionada  = st.selectbox(
        "Estado de Notificação",
        options=["Todos"] + ufs_disponiveis,
        index=0,
        label_visibility="collapsed",
        key="uf_evolucao",
    )

if uf_selecionada == "Todos":
    df_ev = es.groupby("sem_label", as_index=False).agg(
        total_srag      =("total_srag",      "sum"),
        casos_vsr       =("casos_vsr",       "sum"),
        casos_influenza =("casos_influenza", "sum"),
    )
else:
    df_ev = es[es["uf_notificacao"] == uf_selecionada].copy()

df_ev = df_ev.sort_values("sem_label")

fig_linha = go.Figure()
fig_linha.add_trace(go.Scatter(
    x=df_ev["sem_label"], y=df_ev["total_srag"],
    name="Total SRAG Notificados", mode="lines+markers",
    line=dict(color=LARANJA, width=2.5),
    marker=dict(size=5, color=LARANJA),
    fill="tozeroy", fillcolor="rgba(232,96,10,0.07)",
))
fig_linha.add_trace(go.Scatter(
    x=df_ev["sem_label"], y=df_ev["casos_vsr"],
    name="VSR confirmado", mode="lines+markers",
    line=dict(color=AZUL_LINHA, width=2.5, dash="dot"),
    marker=dict(size=5, color=AZUL_LINHA),
))
if "casos_influenza" in df_ev.columns:
    fig_linha.add_trace(go.Scatter(
        x=df_ev["sem_label"], y=df_ev["casos_influenza"],
        name="Influenza confirmado", mode="lines+markers",
        line=dict(color=VERDE_LINHA, width=2.5, dash="dash"),
        marker=dict(size=5, color=VERDE_LINHA),
    ))
fig_linha = layout_plotly(fig_linha, height=380)
fig_linha.update_xaxes(tickangle=-45)
st.plotly_chart(fig_linha, width="stretch")


# ── Seção 3: Mapa de Casos ────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Mapa de Casos</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">'
            'Tendência de crescimento nas últimas 6 semanas epidemiológicas por UF · '
            'Lógica InfoGripe/Fiocruz</div>', unsafe_allow_html=True)

col_fm, _ = st.columns([2, 3])
with col_fm:
    st.markdown(f"<div style='font-size:0.72rem;color:{TEXTO_MUTED};text-transform:uppercase;"
                f"letter-spacing:0.08em;margin-bottom:0.3rem'>Tipo de Caso</div>",
                unsafe_allow_html=True)
    tipo_mapa = st.selectbox(
        "Tipo de Caso Mapa",
        options=["Casos Notificados SRAG", "Casos Confirmados VSR", "Casos Confirmados Influenza"],
        index=0,
        label_visibility="collapsed",
        key="tipo_mapa",
    )

col_tendencia = {
    "Casos Notificados SRAG":      ("tendencia_srag",      "total_srag"),
    "Casos Confirmados VSR":       ("tendencia_vsr",       "casos_vsr"),
    "Casos Confirmados Influenza": ("tendencia_influenza", "casos_influenza"),
}
chave_tend, coluna_tend = col_tendencia[tipo_mapa]

if chave_tend in dados:
    tend_df = dados[chave_tend].copy()
else:
    es_tend = es[["uf_notificacao", "sem_label", coluna_tend]].rename(
        columns={coluna_tend: "total_srag"}
    )
    tend_df = calcular_tendencia_on_the_fly(es_tend, "total_srag")

tend_df["nome_uf"] = tend_df["uf_notificacao"].map(SIGLA_PARA_NOME)

col_mapa, col_legenda = st.columns([4, 1])
with col_mapa:
    if geojson:
        fig_mapa = px.choropleth(
            tend_df,
            geojson=geojson,
            locations="nome_uf",
            featureidkey="properties.name",
            color="slope",
            color_continuous_scale=[
                [0.0,  "#1F0F08"],
                [0.25, "#3D2010"],
                [0.45, "#6B3A1F"],
                [0.6,  "#C45A1A"],
                [0.8,  "#E8600A"],
                [1.0,  "#FF4500"],
            ],
            hover_name="nome_uf",
            hover_data={"slope": ":.1f", "nome_uf": False},
            labels={"slope": "Tendência (slope)"},
        )
        fig_mapa.update_geos(fitbounds="locations", visible=False, bgcolor="rgba(0,0,0,0)")
        fig_mapa.update_layout(
            height=500,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color=TEXTO_CLARO),
            margin=dict(l=0, r=0, t=10, b=0),
            coloraxis_colorbar=dict(
                title="Tendência",
                tickfont=dict(color=TEXTO_CLARO),
                title_font=dict(color=TEXTO_CLARO),
                bgcolor="rgba(0,0,0,0)",
                bordercolor=CINZA_LINHA,
            ),
        )
        st.plotly_chart(fig_mapa, width="stretch")
    else:
        st.warning("Não foi possível carregar o GeoJSON. Verifique a conexão com a internet.")

with col_legenda:
    st.markdown(f"""
    <div class="legend-box" style="margin-top:3rem">
        <div style="font-family:Syne,sans-serif;font-size:0.8rem;font-weight:700;
                    color:{TEXTO_CLARO};margin-bottom:0.8rem">
            Tendência<br>
            <span style="font-size:0.68rem;color:{TEXTO_MUTED}">Últimas 6 semanas</span>
        </div>
        <div class="legend-item"><div class="legend-dot" style="background:#FF4500"></div>Alto crescimento</div>
        <div class="legend-item"><div class="legend-dot" style="background:#E8600A"></div>Crescimento</div>
        <div class="legend-item"><div class="legend-dot" style="background:#C45A1A"></div>Crescimento leve</div>
        <div class="legend-item"><div class="legend-dot" style="background:#6B3A1F"></div>Estável</div>
        <div class="legend-item"><div class="legend-dot" style="background:#3D2010"></div>Queda</div>
        <div class="legend-item"><div class="legend-dot" style="background:#1F0F08"></div>Queda acentuada</div>
        <div style="margin-top:0.8rem;font-size:0.65rem;color:{TEXTO_MUTED};line-height:1.5">
            Regressão linear simples (coef. angular) sobre as últimas 6 semanas
            epidemiológicas. Metodologia InfoGripe/Fiocruz.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Seção 4: Perfil do Paciente ───────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Perfil do Paciente</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Distribuição demográfica e clínica dos casos</div>',
            unsafe_allow_html=True)

col_fp, _ = st.columns([2, 3])
with col_fp:
    st.markdown(f"<div style='font-size:0.72rem;color:{TEXTO_MUTED};text-transform:uppercase;"
                f"letter-spacing:0.08em;margin-bottom:0.3rem'>Tipo de Caso</div>",
                unsafe_allow_html=True)
    tipo_perfil = st.selectbox(
        "Tipo de Caso Perfil",
        options=["Casos Notificados SRAG", "Casos Confirmados VSR", "Casos Confirmados Influenza"],
        index=0,
        label_visibility="collapsed",
        key="tipo_perfil",
    )

# Mapeia seleção → sufixo dos parquets
sufixo_perfil = {
    "Casos Notificados SRAG":      "srag",
    "Casos Confirmados VSR":       "vsr",
    "Casos Confirmados Influenza": "influenza",
}[tipo_perfil]

# Carrega os 4 dataframes do subgrupo selecionado
fe  = dados.get(f"faixa_etaria_{sufixo_perfil}",  dados.get("faixa_etaria_srag",  pd.DataFrame()))
sx  = dados.get(f"sexo_{sufixo_perfil}",           dados.get("sexo_srag",           pd.DataFrame()))
sin = dados.get(f"sintomas_{sufixo_perfil}",       dados.get("sintomas_srag",       pd.DataFrame()))
fr  = dados.get(f"fatores_risco_{sufixo_perfil}",  dados.get("fatores_risco_srag",  pd.DataFrame()))

col_idade, col_sexo = st.columns([3, 2])

with col_idade:
    st.markdown(f"<div style='font-size:0.8rem;color:{TEXTO_MUTED};margin-bottom:0.5rem'>"
                "Casos por Faixa Etária</div>", unsafe_allow_html=True)
    fe_plot = fe.copy()
    if not fe_plot.empty:
        if "faixa_etaria" not in fe_plot.columns:
            fe_plot = fe_plot.rename(columns={fe_plot.columns[0]: "faixa_etaria",
                                              fe_plot.columns[1]: "total"})
        fig_idade = px.bar(
            fe_plot, x="faixa_etaria", y="total",
            color_discrete_sequence=[LARANJA],
            labels={"faixa_etaria": "Faixa Etária", "total": "Casos"},
        )
        fig_idade.update_traces(marker_line_width=0)
        fig_idade = layout_plotly(fig_idade, height=300)
        st.plotly_chart(fig_idade, width="stretch")

with col_sexo:
    st.markdown(f"<div style='font-size:0.8rem;color:{TEXTO_MUTED};margin-bottom:0.5rem'>"
                "Casos por Sexo</div>", unsafe_allow_html=True)
    sx_plot = sx.copy()
    if not sx_plot.empty:
        if "sexo" not in sx_plot.columns:
            sx_plot = sx_plot.rename(columns={sx_plot.columns[0]: "sexo",
                                              sx_plot.columns[1]: "total"})
        sx_plot      = sx_plot[sx_plot["sexo"] != "Ignorado"]
        total_genero = int(sx_plot["total"].sum())
        fig_sexo = go.Figure(go.Pie(
            labels=sx_plot["sexo"], values=sx_plot["total"],
            hole=0.62,
            marker=dict(colors=[LARANJA, LARANJA_CLARO],
                        line=dict(color=FUNDO_ESCURO, width=2)),
            textfont=dict(color="#fff", size=11),
            hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
        ))
        fig_sexo.update_layout(
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color=TEXTO_CLARO),
            margin=dict(l=0, r=0, t=10, b=10),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXTO_CLARO, size=11)),
            annotations=[dict(
                text=f"<b>{total_genero:,}</b>",
                x=0.5, y=0.5,
                font=dict(size=16, color=TEXTO_CLARO, family="Syne"),
                showarrow=False,
            )],
        )
        st.plotly_chart(fig_sexo, width="stretch")

col_sint, col_fr = st.columns(2)

with col_sint:
    st.markdown(f"<div style='font-size:0.8rem;color:{TEXTO_MUTED};margin-bottom:0.5rem'>"
                "Top Sintomas Relatados</div>", unsafe_allow_html=True)
    sin_plot = sin.copy()
    if not sin_plot.empty:
        if "sintoma" not in sin_plot.columns:
            sin_plot = sin_plot.rename(columns={sin_plot.columns[0]: "sintoma",
                                                sin_plot.columns[1]: "total"})
        sin_plot = sin_plot.sort_values("total", ascending=True)
        fig_sint = px.bar(
            sin_plot, x="total", y="sintoma", orientation="h",
            color_discrete_sequence=[LARANJA],
            labels={"sintoma": "", "total": "Casos"},
        )
        fig_sint.update_traces(marker_line_width=0)
        fig_sint = layout_plotly(fig_sint, height=320)
        st.plotly_chart(fig_sint, width="stretch")

with col_fr:
    st.markdown(f"<div style='font-size:0.8rem;color:{TEXTO_MUTED};margin-bottom:0.5rem'>"
                "Top Fatores de Risco</div>", unsafe_allow_html=True)
    fr_plot = fr.copy()
    if not fr_plot.empty:
        if "fator_risco" not in fr_plot.columns:
            fr_plot = fr_plot.rename(columns={fr_plot.columns[0]: "fator_risco",
                                              fr_plot.columns[1]: "total"})
        fr_plot = fr_plot.sort_values("total", ascending=True)
        fig_fr = px.bar(
            fr_plot, x="total", y="fator_risco", orientation="h",
            color_discrete_sequence=[LARANJA_CLARO],
            labels={"fator_risco": "", "total": "Casos"},
        )
        fig_fr.update_traces(marker_line_width=0)
        fig_fr = layout_plotly(fig_fr, height=320)
        st.plotly_chart(fig_fr, width="stretch")


# ── Rodapé ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-top:2rem;padding-top:1rem;border-top:1px solid {CINZA_LINHA};
            text-align:center;font-size:0.7rem;color:{TEXTO_MUTED}">
    Dados: SIVEP-Gripe / Ministério da Saúde · OpenDataSUS ·
    Última atualização disponível: <b style="color:{LARANJA_CLARO}">{ultima_semana}</b> ·
    Banco vivo atualizado semanalmente · Valores sujeitos a revisão
</div>
""", unsafe_allow_html=True)
