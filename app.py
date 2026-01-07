import os
import json
from typing import Dict, Any, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import streamlit as st
from urllib.parse import urlparse

# =====================
# CONFIG
# =====================

ARQUIVOS = {
    "Veículos": "mapeamento_veiculos_atualizados.json",
    "Agro": "Agro_infos.json",
}

TOP_N_SUBCATEGORIAS = 20
LIMIAR_GAP_QTD = 10

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()


# =====================
# DATA
# =====================

def carregar_json(caminho: str) -> Dict[str, Any]:
    if not os.path.isfile(caminho):
        raise FileNotFoundError(f"Arquivo JSON não encontrado: {caminho}")
    with open(caminho, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "detalhes" not in data or not isinstance(data["detalhes"], list):
        raise ValueError("JSON não possui a chave 'detalhes' como lista.")
    return data


def extrair_slug_corrigido(link: str) -> str:
    if not isinstance(link, str):
        return "Geral"
    path = urlparse(link).path or ""
    if "_NoIndex" in path:
        path = path.split("_NoIndex")[0]
    parts = [p for p in path.split("/") if p]
    if not parts:
        return "Geral"
    slug = parts[-1]
    slug_limpo = slug.replace("-", " ").title()
    return slug_limpo or "Geral"


def preparar_dataframe(data: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    detalhes = data["detalhes"]
    df = pd.DataFrame(detalhes)

    for col in ["categoria", "quantidade", "link"]:
        if col not in df.columns:
            raise ValueError(f"Coluna obrigatória ausente: {col}")

    categorias_split = df["categoria"].fillna("").str.split(" > ", expand=True)
    col_names = [f"nivel_{i+1}" for i in range(categorias_split.shape[1])]
    categorias_split.columns = col_names
    df = pd.concat([df, categorias_split], axis=1)
    df[col_names] = df[col_names].replace("", pd.NA)

    df["categoria_real"] = df["link"].apply(extrair_slug_corrigido)
    df["quantidade"] = pd.to_numeric(df["quantidade"], errors="coerce").fillna(0).astype(int)

    def montar_categoria_completa(row) -> str:
        partes_validas = []
        for col in col_names:
            val = row.get(col)
            if isinstance(val, str) and val.strip():
                partes_validas.append(val.strip())
        slug = row.get("categoria_real")
        if isinstance(slug, str) and slug.strip():
            partes_validas.append(slug.strip())
        return " / ".join(partes_validas) if partes_validas else "Geral"

    df["categoria_completa"] = df.apply(montar_categoria_completa, axis=1)

    total_quantidade = int(df["quantidade"].sum())

    meta = {
        "total_acumulado_json": data.get("total_acumulado_veiculos"),
        "ultima_atualizacao": data.get("ultima_atualizacao"),
        "categorias_concluidas_count": data.get("categorias_concluidas_count"),
        "total_quantidade_calculado": total_quantidade,
        "total_linhas": int(len(df)),
    }
    return df, meta


def agregar_dados(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    resultados: Dict[str, pd.DataFrame] = {}

    agg_n1 = (
        df.groupby("nivel_1", dropna=False)["quantidade"]
        .sum()
        .reset_index()
        .rename(columns={"quantidade": "quantidade_total"})
        .sort_values("quantidade_total", ascending=False)
    )
    total = agg_n1["quantidade_total"].sum()
    if total > 0:
        agg_n1["participacao_%"] = (agg_n1["quantidade_total"] / total * 100).round(2)
    else:
        agg_n1["participacao_%"] = 0
    resultados["agg_nivel_1"] = agg_n1

    agg_n1_n2 = (
        df.groupby(["nivel_1", "nivel_2"], dropna=False)["quantidade"]
        .sum()
        .reset_index()
        .rename(columns={"quantidade": "quantidade_total"})
        .sort_values("quantidade_total", ascending=False)
    )
    resultados["agg_nivel_1_2"] = agg_n1_n2

    top_n = df.sort_values("quantidade", ascending=False).head(TOP_N_SUBCATEGORIAS).copy()
    resultados["top_n_subcategorias"] = top_n

    gaps = df[df["quantidade"] <= LIMIAR_GAP_QTD].copy()
    resultados["gaps_subcategorias"] = gaps

    maiores_zero = df[df["quantidade"] > 0].copy()
    resultados["subcategorias_maiores_zero"] = maiores_zero

    iguais_zero = df[df["quantidade"] == 0].copy()
    resultados["subcategorias_zero"] = iguais_zero

    return resultados


def gerar_resumo_executivo(meta: Dict[str, Any], agregados: Dict[str, pd.DataFrame]) -> str:
    total_produtos = meta["total_quantidade_calculado"]
    total_subcats = meta["total_linhas"]
    zeros = agregados["subcategorias_zero"]
    gaps = agregados["gaps_subcategorias"]

    linhas = []
    linhas.append(f"Total de produtos mapeados: {total_produtos:,}".replace(",", "."))
    linhas.append(f"Total de subcategorias analisadas: {total_subcats}")
    linhas.append(f"Subcategorias sem nenhum produto: {len(zeros)}")
    linhas.append(
        f"Subcategorias com até {LIMIAR_GAP_QTD} produtos (possíveis gaps): {len(gaps)}"
    )

    top_n1 = agregados["agg_nivel_1"].head(5)
    linhas.append("")
    linhas.append("Top 5 categorias de nível 1 por volume:")
    for _, row in top_n1.iterrows():
        linha = (
            f"- {row['nivel_1']}: {row['quantidade_total']:,} produtos "
            f"({row['participacao_%']:.2f}% do total)"
        ).replace(",", ".")
        linhas.append(linha)

    return "\n".join(linhas)


# =====================
# GRÁFICOS AUXILIARES
# =====================

def grafico_top_subcategorias(df_top: pd.DataFrame):
    df_plot = df_top.sort_values("quantidade", ascending=True)
    fig, ax = plt.subplots(figsize=(11, max(6, len(df_plot) * 0.4)))
    
    ax.barh(df_plot["categoria_completa"], df_plot["quantidade"])
    
    ax.set_xlabel("Quantidade de produtos")
    ax.set_ylabel("Subcategoria")
    ax.set_title(f"Top {len(df_plot)} subcategorias por quantidade")
    
    ax.tick_params(labelsize=8)

    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, pos: f"{int(x):,}".replace(",", "."))
    )#formatação com o matplotlib.ticker para o eixo x

    fig.subplots_adjust(left=0.38, right=0.98, top=0.9, bottom=0.2)

    return fig


def grafico_histograma(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["quantidade"], bins=50)
    ax.set_xlabel("Quantidade de produtos por subcategoria")
    ax.set_ylabel("Número de subcategorias")
    ax.set_title("Distribuição de quantidade por subcategoria")
    plt.tight_layout()
    return fig


def grafico_cauda_longa(df_maiores_zero: pd.DataFrame):
    df_plot = df_maiores_zero.sort_values("quantidade", ascending=True).head(30)
    if df_plot.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, max(6, len(df_plot) * 0.35)))
    ax.barh(df_plot["categoria_completa"], df_plot["quantidade"])
    ax.set_xlabel("Quantidade de produtos (>0)")
    ax.set_ylabel("Subcategorias com menor quantidade")
    ax.set_title("Menores subcategorias com produtos (cauda longa)")
    plt.tight_layout()
    return fig


def grafico_comparativo_totais(df_metric: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df_metric["Segmento"], df_metric["Total_Produtos"])
    ax.set_ylabel("Total de produtos mapeados")
    ax.set_title("Comparativo de volume por segmento")
    for i, v in enumerate(df_metric["Total_Produtos"]):
        ax.text(i, v, f"{v:,}".replace(",", "."), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    return fig


# =====================
# DASHBOARDS
# =====================

def montar_dashboard_segmento(nome: str, df: pd.DataFrame, meta: Dict[str, Any], agregados: Dict[str, pd.DataFrame]):
    st.subheader(f"Visão geral - {nome}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total produtos", f"{meta['total_quantidade_calculado']:,}".replace(",", "."))
    col2.metric("Subcategorias", meta["total_linhas"])
    col3.metric("Sem produtos", len(agregados["subcategorias_zero"]))
    col4.metric(f"Até {LIMIAR_GAP_QTD} produtos", len(agregados["gaps_subcategorias"]))

    st.caption(f"Última atualização (JSON): {meta.get('ultima_atualizacao')}")
    st.markdown("---")

    agg_n1 = agregados["agg_nivel_1"]
    top_n1 = agg_n1.iloc[0] if not agg_n1.empty else None

    top_sub = agregados["top_n_subcategorias"].iloc[0] if not agregados["top_n_subcategorias"].empty else None
    pct_top = 0
    if meta["total_quantidade_calculado"] > 0 and not agregados["top_n_subcategorias"].empty:
        pct_top = (
            agregados["top_n_subcategorias"]["quantidade"].sum()
            / meta["total_quantidade_calculado"]
            * 100
        )

    st.markdown("#### Insights rápidos")

    insights = []
    if top_n1 is not None:
        insights.append(
            f"- A maior macro categoria é **{top_n1['nivel_1']}** "
            f"com **{int(top_n1['quantidade_total']):,}** produtos "
            f"({top_n1['participacao_%']:.1f}% do total).".replace(",", ".")
        )

    if top_sub is not None:
        insights.append(
            f"- A subcategoria com mais produtos é **{top_sub['categoria_completa']}** "
            f"com **{int(top_sub['quantidade']):,}** itens.".replace(",", ".")
        )

    insights.append(
        f"- As top {len(agregados['top_n_subcategorias'])} subcategorias concentram "
        f"cerca de **{pct_top:.1f}%** de todos os produtos do segmento."
    )

    st.markdown("\n".join(insights))
    st.markdown("---")


    col_a, col_b = st.columns((2, 1))

    with col_a:
        st.markdown("#### Top subcategorias")
        fig_top = grafico_top_subcategorias(agregados["top_n_subcategorias"])
        st.pyplot(fig_top)

    with col_b:
        st.markdown("#### Distribuição geral")
        fig_hist = grafico_histograma(df)
        st.pyplot(fig_hist)

    st.markdown("#### Menores subcategorias com produtos")
    fig_cauda = grafico_cauda_longa(agregados["subcategorias_maiores_zero"])
    if fig_cauda:
        st.pyplot(fig_cauda)
    else:
        st.info("Não há subcategorias com quantidade > 0 para exibir cauda longa.")

    st.markdown("#### Gaps (subcategorias com baixa oferta)")
    gaps_view = agregados["gaps_subcategorias"][
        ["categoria_completa", "quantidade", "link"]
    ].sort_values("quantidade", ascending=True)
    st.dataframe(gaps_view, use_container_width=True, hide_index=True)

    with st.expander("Resumo executivo em texto"):
        resumo = gerar_resumo_executivo(meta, agregados)
        st.text(resumo)


def montar_dashboard_comparativo(dados_segmentos: Dict[str, Dict[str, Any]]):
    st.subheader("Comparativo Agro x Veículos")

    metric_rows = []
    for nome, dados in dados_segmentos.items():
        df = dados["df"]
        meta = dados["meta"]
        agregados = dados["agregados"]
        metric_rows.append(
            {
                "Segmento": nome,
                "Total_Produtos": meta["total_quantidade_calculado"],
                "Subcategorias": meta["total_linhas"],
                "Subcategorias_Zero": len(agregados["subcategorias_zero"]),
                "Gaps_Até_Limiar": len(agregados["gaps_subcategorias"]),
            }
        )
    df_metric = pd.DataFrame(metric_rows)

    col1, col2, col3, col4 = st.columns(4)
    for _, row in df_metric.iterrows():
        label = row["Segmento"]
        if label == "Veículos":
            c = col1
        else:
            c = col2
        c.metric(label, f"{row['Total_Produtos']:,}".replace(",", "."))

    col3.metric("Total subcategorias (soma)", int(df_metric["Subcategorias"].sum()))
    col4.metric("Total segmentos", len(df_metric))

    st.markdown("#### Volume total por segmento")
    fig_totais = grafico_comparativo_totais(df_metric)
    st.pyplot(fig_totais)

    st.markdown("#### Comparativo Top categorias de nível 1")
    for nome, dados in dados_segmentos.items():
        st.markdown(f"**{nome} – Top nível 1**")
        top_n1 = dados["agregados"]["agg_nivel_1"].head(10)
        st.dataframe(top_n1, use_container_width=True, hide_index=True)

    st.markdown("#### Tabela consolidada de indicadores")
    st.dataframe(df_metric, use_container_width=True, hide_index=True)


# =====================
# APP STREAMLIT
# =====================

def carregar_tudo():
    dados_segmentos: Dict[str, Dict[str, Any]] = {}
    for nome, arquivo in ARQUIVOS.items():
        caminho = os.path.join(BASE_DIR, arquivo)
        data = carregar_json(caminho)
        df, meta = preparar_dataframe(data)
        agregados = agregar_dados(df)
        dados_segmentos[nome] = {
            "df": df,
            "meta": meta,
            "agregados": agregados,
        }
    return dados_segmentos


def main_app():
    st.set_page_config(
        page_title="Painel Marketplace - Agro & Veículos",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.5rem; padding-bottom: 1rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("📊 Painel de Categorias - Marketplace")
    st.caption("Baseado em raspagem de categorias e contagem de produtos (Agro & Veículos).")

    dados_segmentos = carregar_tudo()

    with st.sidebar:
        st.markdown("### Filtro de visão")
        opcao = st.radio(
            "Selecione o segmento:",
            ("Veículos", "Agro", "Comparativo (Agro x Veículos)"),
        )
        st.markdown("---")
        st.markdown("**Parâmetros atuais:**")
        st.write(f"Top subcategorias: {TOP_N_SUBCATEGORIAS}")
        st.write(f"Limiar de gap: {LIMIAR_GAP_QTD} produtos")

    if opcao in ("Veículos", "Agro"):
        dados = dados_segmentos[opcao]
        montar_dashboard_segmento(
            opcao,
            dados["df"],
            dados["meta"],
            dados["agregados"],
        )
    else:
        montar_dashboard_comparativo(dados_segmentos)


if __name__ == "__main__":
    main_app()
