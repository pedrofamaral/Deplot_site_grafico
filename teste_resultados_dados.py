import os
import json
from typing import Dict, Any, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import plotly.express as px  # Necessário para a aba Inteligência
import streamlit as st
from urllib.parse import urlparse

# =====================
# CONFIGURAÇÕES
# =====================

ARQUIVOS = {
    "Veículos": "mapeamento_veiculos_atualizados.json",
    "Agro": "Agro_infos.json",
}

TOP_N_SUBCATEGORIAS = 20
LIMIAR_GAP_QTD = 10
LIMIAR_OPORTUNIDADE_MAX = 100  # Para a inteligência estratégica

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()


# =====================
# PROCESSAMENTO DE DADOS
# =====================

def carregar_json(caminho: str) -> Dict[str, Any]:
    if not os.path.isfile(caminho):
        st.error(f"Arquivo não encontrado: {caminho}")
        return {"detalhes": []}
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
    if not detalhes:
        return pd.DataFrame(), {}
        
    df = pd.DataFrame(detalhes)

    # Garante colunas obrigatórias
    for col in ["categoria", "quantidade", "link"]:
        if col not in df.columns:
            df[col] = 0 if col == "quantidade" else ""

    # Split de categorias (Garante 4 níveis para evitar erros)
    categorias_split = df["categoria"].fillna("").str.split(" > ", expand=True)
    while categorias_split.shape[1] < 4:
        categorias_split[categorias_split.shape[1]] = None
        
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
        "total_acumulado_veiculos": data.get("total_acumulado_veiculos"),
        "ultima_atualizacao": data.get("ultima_atualizacao"),
        "categorias_concluidas_count": data.get("categorias_concluidas_count"),
        "total_quantidade_calculado": total_quantidade,
        "total_linhas": int(len(df)),
    }
    return df, meta

def agregar_dados(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if df.empty: return {}
    resultados: Dict[str, pd.DataFrame] = {}

    # Agregação Nível 1
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

    # Agregação Nível 1 + 2
    agg_n1_n2 = (
        df.groupby(["nivel_1", "nivel_2"], dropna=False)["quantidade"]
        .sum()
        .reset_index()
        .rename(columns={"quantidade": "quantidade_total"})
        .sort_values("quantidade_total", ascending=False)
    )
    resultados["agg_nivel_1_2"] = agg_n1_n2

    # Top N
    top_n = df.sort_values("quantidade", ascending=False).head(TOP_N_SUBCATEGORIAS).copy()
    resultados["top_n_subcategorias"] = top_n

    # Gaps e Zeros
    gaps = df[df["quantidade"] <= LIMIAR_GAP_QTD].copy()
    resultados["gaps_subcategorias"] = gaps
    resultados["subcategorias_maiores_zero"] = df[df["quantidade"] > 0].copy()
    resultados["subcategorias_zero"] = df[df["quantidade"] == 0].copy()
    
    # Oportunidades (Novo - para Inteligência)
    resultados["oportunidades"] = df[
        (df["quantidade"] > 0) & (df["quantidade"] <= LIMIAR_OPORTUNIDADE_MAX)
    ].sort_values("quantidade", ascending=True).copy()

    return resultados

# =====================
# FUNÇÕES VISUAIS: ORIGINAL (MATPLOTLIB)
# =====================

def gerar_resumo_executivo(meta: Dict[str, Any], agregados: Dict[str, pd.DataFrame]) -> str:
    total_produtos = meta["total_quantidade_calculado"]
    total_subcats = meta["total_linhas"]
    zeros = agregados["subcategorias_zero"]
    gaps = agregados["gaps_subcategorias"]

    linhas = []
    linhas.append(f"Total de produtos mapeados: {total_produtos:,}".replace(",", "."))
    linhas.append(f"Total de subcategorias analisadas: {total_subcats}")
    linhas.append(f"Subcategorias sem nenhum produto: {len(zeros)}")
    linhas.append(f"Subcategorias com até {LIMIAR_GAP_QTD} produtos (possíveis gaps): {len(gaps)}")

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

def grafico_top_subcategorias(df_top: pd.DataFrame):
    df_plot = df_top.sort_values("quantidade", ascending=True)
    fig, ax = plt.subplots(figsize=(11, max(6, len(df_plot) * 0.4)))
    ax.barh(df_plot["categoria_completa"], df_plot["quantidade"])
    ax.set_xlabel("Quantidade de produtos")
    ax.set_ylabel("Subcategoria")
    ax.set_title(f"Top {len(df_plot)} subcategorias por quantidade")
    ax.tick_params(labelsize=8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{int(x):,}".replace(",", ".")))
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
    if df_plot.empty: return None
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
# FUNÇÕES VISUAIS: NOVA (PLOTLY - ESTRATÉGICO)
# =====================

def analisar_desbalanceamento(df: pd.DataFrame) -> pd.DataFrame:
    if "nivel_2" not in df.columns or "nivel_3" not in df.columns:
        return pd.DataFrame()
    
    # Agrupa por Pai (Nivel 1 > Nivel 2)
    stats = df.groupby(["nivel_1", "nivel_2"])["quantidade"].agg(
        ['count', 'sum', 'max', 'mean', 'median']
    ).reset_index()
    
    stats = stats[stats['count'] > 1].copy()
    stats['ratio_desbalanceamento'] = (stats['max'] / stats['median']).fillna(0)

    cols_arredondar = ['mean', 'median', 'ratio_desbalanceamento']
    stats[cols_arredondar] = stats[cols_arredondar].round(2)
    
    return stats.sort_values("ratio_desbalanceamento", ascending=False)

def renderizar_analise_estrategica(df: pd.DataFrame, agregados: Dict[str, pd.DataFrame]):
    st.markdown("## 🧠 Inteligência de Mercado & Ação")
    st.markdown("Respondendo: Onde somos fortes? Onde falta produto? Onde o catálogo está sujo?")

    tab1, tab2, tab3, tab4 = st.tabs([
        "💎 Fortalezas (Mapas)", 
        "🚀 Oportunidades (Gaps)", 
        "💀 Limpeza de Catálogo",
        "⚖️ Desbalanceamento"
    ])

    # --- TAB 1: TREEMAP ---
    with tab1:
        st.markdown("### Mapa de Calor do Estoque")
        try:
            fig_tree = px.treemap(
                df, 
                path=[px.Constant("Todos"), 'nivel_1', 'nivel_2', 'nivel_3'], 
                values='quantidade',
                color='quantidade',
                color_continuous_scale='RdBu_r'
            )
            st.plotly_chart(fig_tree, use_container_width=True)
        except Exception as e:
            st.error(f"Erro ao gerar Treemap: {e}")

    # --- TAB 2: OPORTUNIDADES (ATUALIZADA) ---
    with tab2:
        st.markdown(f"### 🚀 Oportunidades (1 a {LIMIAR_OPORTUNIDADE_MAX} itens)")
        df_opt = agregados["oportunidades"]
        if not df_opt.empty:
            col1, col2 = st.columns([3, 1])
            with col1:
                # Agora mostrando Nivel 1 e Nivel 2 explicitamente
                cols_mostrar = ["nivel_1", "nivel_2", "categoria_real", "quantidade", "link"]
                # Garante que as colunas existem antes de mostrar
                cols_validas = [c for c in cols_mostrar if c in df_opt.columns]
                
                st.dataframe(
                    df_opt[cols_validas],
                    use_container_width=True,
                    hide_index=True
                )
            with col2:
                st.metric("Nichos identificados", len(df_opt))
                csv = df_opt.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Baixar Lista CSV",
                    data=csv,
                    file_name="oportunidades.csv",
                    mime="text/csv"
                )
        else:
            st.warning("Nenhuma oportunidade encontrada nesta faixa.")

    # --- TAB 3: LIMPEZA ---
    with tab3:
        st.markdown("### 💀 Categorias Zeradas")
        df_zero = agregados["subcategorias_zero"]
        if not df_zero.empty:
            st.metric("Total Zeradas", len(df_zero))
            csv = df_zero.to_csv(index=False).encode('utf-8')
            st.download_button("Baixar Relatório", csv, "zeradas.csv", "text/csv")
            st.dataframe(df_zero[["categoria", "link"]], use_container_width=True)

    # --- TAB 4: DESBALANCEAMENTO (ATUALIZADA COM GRÁFICO) ---
    with tab4:
        st.markdown("### ⚖️ Árvores Desbalanceadas")
        st.markdown("Identifique categorias onde um único produto domina o volume, distorcendo a análise.")
        
        df_balanco = analisar_desbalanceamento(df)
        
        if not df_balanco.empty:
            # Tabela Geral
            st.dataframe(
                df_balanco.head(20).style.background_gradient(subset=['ratio_desbalanceamento'], cmap='Reds'),
                use_container_width=True
            )
            
            st.markdown("---")
            st.markdown("#### 🔎 Zoom em um Pai Desbalanceado")
            
            # Selectbox para escolher qual categoria investigar
            pais_disponiveis = df_balanco["nivel_2"].unique()
            pai_selecionado = st.selectbox("Selecione a Categoria Pai para investigar:", pais_disponiveis)
            
            if pai_selecionado:
                # Filtra os dados apenas para esse pai
                filhos = df[df["nivel_2"] == pai_selecionado].sort_values("quantidade", ascending=True)
                
                # Gráfico de barras interativo
                fig_bar = px.bar(
                    filhos, 
                    x="quantidade", 
                    y="categoria_real", 
                    orientation='h',
                    title=f"Distribuição dentro de: {pai_selecionado}",
                    text_auto=True,  # Mostra o número na barra
                    height=max(400, len(filhos) * 30) # Ajusta altura dinamicamente
                )
                fig_bar.update_layout(xaxis_title="Quantidade de Produtos", yaxis_title="Subcategoria (Folha)")
                st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Dados insuficientes para calcular desbalanceamento (precisa de nível 2 e 3).")

# =====================
# DASHBOARD BUILDERS
# =====================

def montar_dashboard_segmento(nome: str, df: pd.DataFrame, meta: Dict[str, Any], agregados: Dict[str, pd.DataFrame]):
    # Seletor Principal
    modo_visao = st.radio(
        "Modo de Visualização:", 
        ["📊 Visão Geral (Clássica)", "🧠 Inteligência Estratégica"], 
        horizontal=True
    )
    
    st.divider()

    if modo_visao == "🧠 Inteligência Estratégica":
        renderizar_analise_estrategica(df, agregados)
    else:
        # --- SCRIPT ORIGINAL MANTIDO ---
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
            pct_top = (agregados["top_n_subcategorias"]["quantidade"].sum() / meta["total_quantidade_calculado"] * 100)

        st.markdown("#### Insights rápidos")
        insights = []
        if top_n1 is not None:
            insights.append(f"- A maior macro categoria é **{top_n1['nivel_1']}** com **{int(top_n1['quantidade_total']):,}** produtos.")
        if top_sub is not None:
            insights.append(f"- A subcategoria líder é **{top_sub['categoria_completa']}** com **{int(top_sub['quantidade']):,}** itens.")
        insights.append(f"- As top {len(agregados['top_n_subcategorias'])} subcategorias concentram **{pct_top:.1f}%** do total.")
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
        if fig_cauda: st.pyplot(fig_cauda)

        st.markdown("#### Gaps (subcategorias com baixa oferta)")
        gaps_view = agregados["gaps_subcategorias"][["categoria_completa", "quantidade", "link"]].sort_values("quantidade", ascending=True)
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
        metric_rows.append({
            "Segmento": nome,
            "Total_Produtos": meta["total_quantidade_calculado"],
            "Subcategorias": meta["total_linhas"],
            "Subcategorias_Zero": len(agregados["subcategorias_zero"]),
            "Gaps_Até_Limiar": len(agregados["gaps_subcategorias"]),
        })
    df_metric = pd.DataFrame(metric_rows)

    col1, col2, col3, col4 = st.columns(4)
    for _, row in df_metric.iterrows():
        if row["Segmento"] == "Veículos": col1.metric(row["Segmento"], f"{row['Total_Produtos']:,}".replace(",", "."))
        else: col2.metric(row["Segmento"], f"{row['Total_Produtos']:,}".replace(",", "."))
    col3.metric("Total subcategorias", int(df_metric["Subcategorias"].sum()))
    col4.metric("Total segmentos", len(df_metric))

    st.markdown("#### Volume total por segmento")
    fig_totais = grafico_comparativo_totais(df_metric)
    st.pyplot(fig_totais)

    st.markdown("#### Comparativo Top categorias de nível 1")
    for nome, dados in dados_segmentos.items():
        st.markdown(f"**{nome} – Top nível 1**")
        st.dataframe(dados["agregados"]["agg_nivel_1"].head(10), use_container_width=True, hide_index=True)

# =====================
# APP PRINCIPAL
# =====================

def carregar_tudo():
    dados_segmentos: Dict[str, Dict[str, Any]] = {}
    for nome, arquivo in ARQUIVOS.items():
        caminho = os.path.join(BASE_DIR, arquivo)
        if os.path.exists(caminho):
            data = carregar_json(caminho)
            df, meta = preparar_dataframe(data)
            agregados = agregar_dados(df)
            dados_segmentos[nome] = {"df": df, "meta": meta, "agregados": agregados}
    return dados_segmentos

def main_app():
    st.set_page_config(page_title="Painel Marketplace", layout="wide", initial_sidebar_state="expanded")
    
    st.title("📊 Painel de Categorias - Marketplace")
    st.caption("Baseado em raspagem de categorias e contagem de produtos.")

    dados_segmentos = carregar_tudo()
    if not dados_segmentos:
        st.warning("Nenhum arquivo JSON encontrado.")
        return

    with st.sidebar:
        st.markdown("### Menu")
        opcao = st.radio("Selecione:", list(dados_segmentos.keys()) + ["Comparativo"])
        st.markdown("---")
        st.write(f"Top N: {TOP_N_SUBCATEGORIAS}")
        st.write(f"Limiar Gap: {LIMIAR_GAP_QTD}")

    if opcao in dados_segmentos:
        dados = dados_segmentos[opcao]
        montar_dashboard_segmento(opcao, dados["df"], dados["meta"], dados["agregados"])
    else:
        montar_dashboard_comparativo(dados_segmentos)

if __name__ == "__main__":
    main_app()