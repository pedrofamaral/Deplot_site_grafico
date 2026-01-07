import json
import os
from typing import Tuple, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from urllib.parse import urlparse


ARQUIVO_JSON = "mapeamento_veiculos_final.json"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "graficos")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tabelas")
HTML_DIR = os.path.join(OUTPUT_DIR, "html")

for d in (OUTPUT_DIR, FIGURES_DIR, TABLES_DIR, HTML_DIR):
    os.makedirs(d, exist_ok=True)

TOP_N_SUBCATEGORIAS = 20          
N_MENORES_SUBCATEGORIAS = 30      
LIMIAR_GAP_QTD = 10               


def carregar_json(caminho: str) -> Dict[str, Any]:
    if not os.path.isfile(caminho):
        raise FileNotFoundError(f"Arquivo JSON não encontrado: {caminho}")

    with open(caminho, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Erro ao decodificar JSON: {e}") from e

    if "detalhes" not in data or not isinstance(data["detalhes"], list):
        raise ValueError("JSON não possui a chave 'detalhes' como lista.")

    return data


def extrair_slug_corrigido(link: str) -> str:
    if not isinstance(link, str):
        return "Geral"

    try:
        path = urlparse(link).path
        if "_NoIndex" in path:
            path = path.split("_NoIndex")[0]
        
        parts = [p for p in path.split('/') if p]
        
        if not parts:
            return "Geral"

        slug = parts[-1]
        
        slug_limpo = slug.replace('-', ' ').title()
        
        if not slug_limpo.strip():
            return "Geral"
            
        return slug_limpo

    except Exception:
        return "Geral"


def preparar_dataframe(data: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    detalhes = data["detalhes"]
    df = pd.DataFrame(detalhes)

    colunas_obrigatorias = ["categoria", "quantidade", "link"]
    for col in colunas_obrigatorias:
        if col not in df.columns:
            raise ValueError(f"Coluna obrigatória '{col}' ausente em 'detalhes'.")

    categorias_split = df["categoria"].fillna("").str.split(" > ", expand=True)
    
    num_niveis = len(categorias_split.columns)
    col_names = [f"nivel_{i+1}" for i in range(num_niveis)]
    categorias_split.columns = col_names
    
    df = pd.concat([df, categorias_split], axis=1)

    df[col_names] = df[col_names].replace("", pd.NA)

    df["categoria_real"] = df["link"].apply(extrair_slug_corrigido)
    
    df["quantidade_original"] = df["quantidade"] 
    df["quantidade"] = pd.to_numeric(df["quantidade"], errors="coerce")

    qtd_na = df["quantidade"].isna().sum()
    if qtd_na > 0:
        print(
            f"Atenção: {qtd_na} linha(s) não puderam ser convertidas para numérico "
            f"em 'quantidade'. Serão tratadas como 0."
        )
        df["quantidade"] = df["quantidade"].fillna(0).astype(int)
    else:
        df["quantidade"] = df["quantidade"].astype(int)

    def montar_categoria_completa(row) -> str:
        partes_validas = []
        
        for col in col_names:
            val = row.get(col)
            if isinstance(val, str):
                p_limpo = val.strip()
                if p_limpo:
                    partes_validas.append(p_limpo)
        
        slug = row.get("categoria_real")
        if isinstance(slug, str) and slug.strip():
            partes_validas.append(slug.strip())
        else:
            partes_validas.append("Geral")

        return " / ".join(partes_validas)

    df["categoria_completa"] = df.apply(montar_categoria_completa, axis=1)

    total_quantidade = df["quantidade"].sum()
    if total_quantidade > 0:
        df["participacao_%"] = (df["quantidade"] / total_quantidade * 100).round(4)
    else:
        df["participacao_%"] = 0.0

    meta = {
        "total_acumulado_veiculos": data.get("total_acumulado_veiculos"),
        "ultima_atualizacao": data.get("ultima_atualizacao"),
        "categorias_concluidas_count": data.get("categorias_concluidas_count"),
        "total_quantidade_calculado": int(total_quantidade),
        "total_linhas": int(len(df)),
    }

    return df, meta

def imprimir_metricas_gerais(meta: Dict[str, Any]) -> None:
    print("\n=== Métricas Gerais ===")
    print(f"Total de produtos (soma do DataFrame): {meta['total_quantidade_calculado']:,}".replace(",", "."))
    print(f"Total de linhas (subcategorias mapeadas): {meta['total_linhas']:,}".replace(",", "."))

    print(f"total_acumulado_veiculos (JSON): {meta.get('total_acumulado_veiculos')}")
    print(f"ultima_atualizacao (JSON): {meta.get('ultima_atualizacao')}")
    print(f"categorias_concluidas_count (JSON): {meta.get('categorias_concluidas_count')}")

    total_json = meta.get("total_acumulado_veiculos")
    if isinstance(total_json, (int, float)):
        diff = meta["total_quantidade_calculado"] - total_json
        perc_diff = (diff / total_json * 100) if total_json != 0 else None
        print("\nComparação total calculado x total_acumulado_veiculos:")
        print(f"  Diferença absoluta: {diff:,}".replace(",", "."))
        if perc_diff is not None:
            print(f"  Diferença relativa: {perc_diff:.4f}%")
    else:
        print("\nAviso: total_acumulado_veiculos não está numérico no JSON; não foi possível comparar.")


def agregar_dados(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    resultados = {}

    agg_n1 = (
        df.groupby("nivel_1", dropna=False)["quantidade"]
        .sum()
        .reset_index()
        .rename(columns={"quantidade": "quantidade_total"})
        .sort_values("quantidade_total", ascending=False)
    )
    total = agg_n1["quantidade_total"].sum()
    agg_n1["participacao_%"] = (agg_n1["quantidade_total"] / total * 100).round(4)
    resultados["agg_nivel_1"] = agg_n1

    agg_n1_n2 = (
        df.groupby(["nivel_1", "nivel_2"], dropna=False)["quantidade"]
        .sum()
        .reset_index()
        .rename(columns={"quantidade": "quantidade_total"})
        .sort_values("quantidade_total", ascending=False)
    )
    resultados["agg_nivel_1_2"] = agg_n1_n2

    df_n3 = df[~df["nivel_3"].isna()].copy()
    if not df_n3.empty:
        agg_n3 = (
            df_n3.groupby("nivel_3")["quantidade"]
            .sum()
            .reset_index()
            .rename(columns={"quantidade": "quantidade_total"})
            .sort_values("quantidade_total", ascending=False)
        )
        resultados["agg_nivel_3"] = agg_n3

    top_n = df.sort_values("quantidade", ascending=False).head(TOP_N_SUBCATEGORIAS).copy()
    resultados["top_n_subcategorias"] = top_n

    menores_ate_50 = df[df["quantidade"] <= 50].sort_values("quantidade", ascending=True).copy()
    resultados["subcategorias_ate_50"] = menores_ate_50

    iguais_zero = df[df["quantidade"] == 0].copy()
    maiores_zero = df[df["quantidade"] > 0].copy()
    gaps = df[df["quantidade"] <= LIMIAR_GAP_QTD].copy()

    resultados["subcategorias_zero"] = iguais_zero
    resultados["subcategorias_maiores_zero"] = maiores_zero
    resultados["gaps_subcategorias"] = gaps

    return resultados


def salvar_resumo_global(df: pd.DataFrame) -> pd.DataFrame:
    resumo_geral = (
        df.groupby("categoria_real")["quantidade"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    resumo_geral.columns = ["Categoria", "Qtd_Produtos"]

    total_produtos = resumo_geral["Qtd_Produtos"].sum()
    resumo_geral["% do Total"] = (resumo_geral["Qtd_Produtos"] / total_produtos) * 100
    resumo_geral["% Acumulado"] = resumo_geral["% do Total"].cumsum()
    resumo_geral["Rank"] = resumo_geral.index + 1

    caminho_csv = os.path.join(TABLES_DIR, "relatorio_completo_categorias.csv")
    resumo_geral.to_csv(
        caminho_csv,
        index=False,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )

    print(f"\nArquivo 'relatorio_completo_categorias.csv' salvo com {len(resumo_geral)} linhas.")
    return resumo_geral


def grafico_cauda_longa(resumo_geral: pd.DataFrame, arquivo: str | None = None) -> None:
    if arquivo is None:
        arquivo = os.path.join(FIGURES_DIR, "grafico_cauda_longa_todos_dados.png")

    print("Gerando gráfico de Cauda Longa (log-log)...")
    plt.figure(figsize=(12, 6))
    plt.plot(
        resumo_geral["Rank"],
        resumo_geral["Qtd_Produtos"],
        marker=".",
        linestyle="none",
        alpha=0.3,
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Visão de Todo o Universo (Rank x Quantidade) - Escala Log", fontsize=14)
    plt.xlabel("Ranking da Categoria (1º ao último)", fontsize=12)
    plt.ylabel("Quantidade de Produtos", fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.axhline(y=1000, color="r", linestyle="--", label="Linha de 1.000 produtos")
    plt.legend()
    plt.tight_layout()
    plt.savefig(arquivo, dpi=300)
    plt.close()


def grafico_histograma_global(resumo_geral: pd.DataFrame, arquivo: str | None = None) -> None:
    if arquivo is None:
        arquivo = os.path.join(FIGURES_DIR, "grafico_histograma_distribuicao.png")

    print("Gerando histograma global de distribuição (< 50k)...")
    plt.figure(figsize=(10, 6))
    dados_hist = resumo_geral[resumo_geral["Qtd_Produtos"] < 50000]["Qtd_Produtos"]
    plt.hist(dados_hist, bins=50)
    plt.title("Distribuição: Quantas categorias têm determinado volume? (Zoom < 50k)", fontsize=14)
    plt.xlabel("Volume de Produtos")
    plt.ylabel("Número de Categorias")
    plt.tight_layout()
    plt.savefig(arquivo, dpi=300)
    plt.close()

def grafico_treemap_global(df: pd.DataFrame, arquivo: str | None = None) -> None:
    if arquivo is None:
        arquivo = os.path.join(HTML_DIR, "grafico_treemap_completo.html")
    print("Gerando Treemap Global (HTML) com base em categoria_completa...")

    if "categoria_completa" not in df.columns:
        raise ValueError("DataFrame não possui a coluna 'categoria_completa'.")
    if "quantidade" not in df.columns:
        raise ValueError("DataFrame não possui a coluna 'quantidade'.")

    df_split = df["categoria_completa"].str.split(" / ", expand=True)
    
    col_names = [f"L{i+1}" for i in range(df_split.shape[1])]
    df_split.columns = col_names
    
    df_tree = pd.concat([df_split, df["quantidade"]], axis=1)

    df_tree = (
        df_tree
        .groupby(col_names, dropna=False)["quantidade"]
        .sum()
        .reset_index()
    )

    total_produtos = df_tree["quantidade"].sum()

    df_tree["caminho_legivel"] = df_tree[col_names].apply(
        lambda row: " > ".join([str(val) for val in row if pd.notna(val) and val != ""]),
        axis=1
    )

    def get_last_label(row):
        for col in reversed(col_names):
            val = row[col]
            if pd.notna(val) and val != "":
                return val
        return "Geral"

    df_tree["nome_categoria"] = df_tree.apply(get_last_label, axis=1)

    df_tree["texto_quadrado"] = (
        df_tree["nome_categoria"]
        + "<br>Qtd: "
        + df_tree["quantidade"].apply(lambda x: f"{x:,.0f}")
    )

    fig = px.treemap(
        df_tree,
        path=col_names,  
        values="quantidade",
        title=f"Mapa de Categorias - {total_produtos:,.0f} itens",
        color="quantidade",
        color_continuous_scale="Viridis",
    )

    fig.update_traces(
        text=df_tree["texto_quadrado"],
        customdata=df_tree["caminho_legivel"],
        textinfo="text",
        texttemplate="%{text}",
        hovertemplate="<b>%{customdata}</b><br>Quantidade: %{value:,.0f}<extra></extra>",
    )

    html_grafico = fig.to_html(full_html=False, include_plotlyjs="cdn")

    html_pagina = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8" />
    <title>Mapa de Categorias</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: #f3f4f6;
            color: #111827;
        }}
        header {{
            background: #111827;
            color: #f9fafb;
            padding: 16px 24px;
        }}
        header h1 {{
            margin: 0;
            font-size: 1.4rem;
        }}
        main {{
            max-width: 1100px;
            margin: 24px auto 40px;
            padding: 0 16px;
        }}
        .card {{
            background: #ffffff;
            border-radius: 12px;
            box-shadow: 0 10px 25px rgba(15, 23, 42, 0.1);
            padding: 24px;
        }}
        .card h2 {{
            margin-top: 0;
            margin-bottom: 8px;
            font-size: 1.3rem;
        }}
        .card p {{
            margin-top: 4px;
            margin-bottom: 12px;
            line-height: 1.5;
        }}
        .help-box {{
            background: #f9fafb;
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 16px;
            border-left: 4px solid #3b82f6;
            font-size: 0.9rem;
        }}
        .plot-container {{
            margin-top: 16px;
        }}
        footer {{
            text-align: center;
            font-size: 0.8rem;
            color: #6b7280;
            margin-bottom: 24px;
        }}
    </style>
</head>
<body>
    <header>
        <h1>Visão geral de categorias</h1>
    </header>

    <main>
        <section class="card">
            <h2>Como ler este gráfico</h2>
            <p>
                Cada bloco representa uma categoria de produto. Quanto maior o bloco,
                maior a quantidade de itens nessa categoria.
            </p>
            <div class="help-box">
                <strong>Dica rápida:</strong><br />
                • Passe o mouse sobre um bloco para ver o caminho completo da categoria.<br />
                • A hierarquia foi ajustada com base no link do produto para maior precisão.<br />
                • A cor indica o volume: quanto mais claro, maior a quantidade.
            </div>

            <div class="plot-container">
                {html_grafico}
            </div>
        </section>
    </main>

    <footer>
        Mapa gerado automaticamente · {total_produtos:,.0f} itens totais
    </footer>
</body>
</html>
"""

    with open(arquivo, "w", encoding="utf-8") as f:
        f.write(html_pagina)

    print(f"Página salva em: {arquivo}")

def grafico_treemap_niveis(df: pd.DataFrame, arquivo: str | None = None) -> None:
    if arquivo is None:
        arquivo = os.path.join(HTML_DIR, "treemap_categorias_niveis.html")
    print("Gerando Treemap por níveis (HTML)...")
    df_tree = (
        df.groupby(["nivel_1", "nivel_2", "nivel_3"], dropna=False)["quantidade"]
        .sum()
        .reset_index()
    )

    for col in ["nivel_1", "nivel_2", "nivel_3"]:
        df_tree[col] = df_tree[col].fillna(f"Sem {col}")

    fig = px.treemap(
        df_tree,
        path=["nivel_1", "nivel_2", "nivel_3"],
        values="quantidade",
        color="quantidade",
        color_continuous_scale="Blues",
        title="Treemap de Categorias - Veículos (Níveis 1-3)",
    )
    fig.write_html(arquivo)


def grafico_barra_top_n(df_top: pd.DataFrame, arquivo: str | None = None) -> None:
    if arquivo is None:
        arquivo = os.path.join(FIGURES_DIR, "grafico_barras_top_subcategorias.png")

    print(f"Gerando gráfico de barras Top {len(df_top)} subcategorias...")
    df_plot = df_top.sort_values("quantidade", ascending=True)

    plt.figure(figsize=(10, max(6, len(df_plot) * 0.4)))
    sns.barplot(
        data=df_plot,
        x="quantidade",
        y="categoria_completa",
        orient="h",
    )
    plt.xlabel("Quantidade de produtos")
    plt.ylabel("Subcategoria (categoria completa)")
    plt.title(f"Top {len(df_plot)} Subcategorias por Quantidade de Produtos")
    plt.tight_layout()
    plt.savefig(arquivo, dpi=300)
    plt.close()


def grafico_pareto(df: pd.DataFrame, arquivo: str | None = None) -> Tuple[float, float]:
    if arquivo is None:
        arquivo = os.path.join(FIGURES_DIR, "grafico_pareto_subcategorias.png")
    
    print("Lendo dados para gráfico de Pareto e mostrando diretorio atual:", os.getcwd())
    print("Gerando gráfico de Pareto...")
    
    df_pareto = df.sort_values("quantidade", ascending=False).reset_index(drop=True)
    df_pareto["quantidade_acumulada"] = df_pareto["quantidade"].cumsum()
    total = df_pareto["quantidade"].sum()
    df_pareto["perc_acumulado"] = df_pareto["quantidade_acumulada"] / total * 100

    idx_80 = (df_pareto["perc_acumulado"] >= 80).idxmax()
    qtd_subcats_80 = idx_80 + 1
    total_subcats = len(df_pareto)
    perc_subcats_80 = qtd_subcats_80 / total_subcats * 100
    perc_total_80 = df_pareto.loc[idx_80, "perc_acumulado"]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.bar(df_pareto.index, df_pareto["quantidade"])
    ax1.set_xlabel("Subcategorias (ordenadas por quantidade)")
    ax1.set_ylabel("Quantidade de produtos", color="black")

    ax2 = ax1.twinx()
    ax2.plot(df_pareto.index, df_pareto["perc_acumulado"], marker="o")
    ax2.set_ylabel("% acumulado", color="black")
    ax2.axhline(80, color="red", linestyle="--", linewidth=1)
    ax2.set_ylim(0, 100)

    plt.title("Gráfico de Pareto - Subcategorias (Veículos)")
    plt.tight_layout()
    plt.savefig(arquivo, dpi=300)
    plt.close()

    print(
        f"As {qtd_subcats_80} maiores subcategorias ({perc_subcats_80:.2f}% do total) "
        f"respondem por {perc_total_80:.2f}% dos produtos."
    )

    return perc_subcats_80, perc_total_80


def graficos_histogramas(df: pd.DataFrame, arquivo_linear: str | None = None, arquivo_log: str | None = None) -> None:
    if arquivo_linear is None:
        arquivo_linear = os.path.join(FIGURES_DIR, "histograma_quantidade.png")

    if arquivo_log is None:
        arquivo_log = os.path.join(FIGURES_DIR, "histograma_quantidade_log.png")

    print("Gerando histogramas (linear e log)...")

    plt.figure(figsize=(10, 6))
    plt.hist(df["quantidade"], bins=50)
    plt.xlabel("Quantidade de produtos por subcategoria")
    plt.ylabel("Frequência")
    plt.title("Distribuição da quantidade por subcategoria (escala linear)")
    plt.tight_layout()
    plt.savefig(arquivo_linear, dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(df["quantidade"], bins=50)
    plt.yscale("log")
    plt.xlabel("Quantidade de produtos por subcategoria")
    plt.ylabel("Frequência (escala log)")
    plt.title("Distribuição da quantidade por subcategoria (escala logarítmica no eixo Y)")
    plt.tight_layout()
    plt.savefig(arquivo_log, dpi=300)
    plt.close()


def grafico_cauda_longa_menores(df_maiores_zero: pd.DataFrame, arquivo: str | None = None) -> None:
    if arquivo is None:
        arquivo = os.path.join(FIGURES_DIR, "subcategorias_menores.png")

    n = min(N_MENORES_SUBCATEGORIAS, len(df_maiores_zero))
    print(f"Gerando gráfico de cauda longa (menores > 0, n={n})...")

    df_menores = df_maiores_zero.sort_values("quantidade", ascending=True).head(n).copy()

    plt.figure(figsize=(10, max(6, n * 0.4)))
    sns.barplot(
        data=df_menores,
        x="quantidade",
        y="categoria_completa",
        orient="h",
    )
    plt.xlabel("Quantidade de produtos (> 0)")
    plt.ylabel("Subcategorias com menor quantidade")
    plt.title(f"{n} menores subcategorias com produtos (cauda longa)")
    plt.tight_layout()
    plt.savefig(arquivo, dpi=300)
    plt.close()


def salvar_tabela_gaps(df_gaps: pd.DataFrame, arquivo_csv: str | None = None, arquivo_grafico: str | None = None) -> None:
    if arquivo_csv is None:
        arquivo_csv = os.path.join(TABLES_DIR, "tabela_gaps_subcategorias.csv")
    
    if arquivo_grafico is None:
        arquivo_grafico = os.path.join(FIGURES_DIR, "grafico_gaps_subcategorias.png")

    if df_gaps is None:
        print("DataFrame de gaps é None; nada a salvar.")
        return

    df_gaps = df_gaps.copy()
    df_gaps["quantidade"] = (
        pd.to_numeric(df_gaps["quantidade"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    print(f"Salvando tabela de gaps em CSV: {arquivo_csv}")
    cols = [
        "categoria_completa",
        "categoria",
        "nivel_1",
        "nivel_2",
        "nivel_3",
        "nivel_4",
        "categoria_real",
        "quantidade",
        "link",
    ]
    cols = [c for c in cols if c in df_gaps.columns]
    df_gaps[cols].to_csv(arquivo_csv, index=False, encoding="utf-8-sig")

    if df_gaps.empty:
        print("Não há gaps (quantidade <= LIMIAR_GAP_QTD) para plotar.")
        return

    n = min(30, len(df_gaps))
    df_plot = (
        df_gaps
        .sort_values(["quantidade", "categoria_completa"], ascending=[True, True])
        .head(n)
        .copy()
    )

    print(f"Gerando gráfico de gaps (n={n})...")
    plt.figure(figsize=(10, max(6, n * 0.4)))
    sns.barplot(
        data=df_plot,
        x="quantidade",
        y="categoria_completa",
        orient="h",
    )
    plt.xlabel("Quantidade de produtos (gaps)")
    plt.ylabel("Subcategorias")
    plt.title(f"Principais gaps de subcategorias (quantidade <= {LIMIAR_GAP_QTD})")

    max_qtd = max(df_plot["quantidade"].max(), 0)
    plt.xlim(0, max_qtd + 1)

    plt.tight_layout()
    plt.savefig(arquivo_grafico, dpi=300)
    plt.close()


# =====================
# INSIGHTS TEXTUAIS
# =====================

def gerar_insights(df: pd.DataFrame,
                   meta: Dict[str, Any],
                   agregados: Dict[str, pd.DataFrame],
                   perc_subcats_80: float,
                   perc_total_80: float) -> None:
    """
    Imprime insights automáticos no console (Pareto, distribuição, gaps etc.).
    """
    print("\n=== Insights Automáticos ===")

    total_produtos = meta["total_quantidade_calculado"]
    total_subcats = meta["total_linhas"]

    agg_n1 = agregados.get("agg_nivel_1")
    if agg_n1 is not None:
        print("\nTop 5 categorias (Nível 1) por quantidade de produtos:")
        top5_n1 = agg_n1.head(5)
        for _, row in top5_n1.iterrows():
            print(
                f"  - {row['nivel_1']}: {row['quantidade_total']:,} produtos "
                f"({row['participacao_%']:.2f}% do total)".replace(",", ".")
            )

    print(
        f"\nRegra de Pareto aproximada: "
        f"{perc_subcats_80:.2f}% das subcategorias respondem por {perc_total_80:.2f}% do total de produtos."
    )

    qtd_zero = (df["quantidade"] == 0).sum()
    qtd_ate_10 = (df["quantidade"].between(1, 10)).sum()
    qtd_11_100 = (df["quantidade"].between(11, 100)).sum()
    qtd_101_1000 = (df["quantidade"].between(101, 1000)).sum()
    qtd_acima_1000 = (df["quantidade"] > 1000).sum()

    print("\nDistribuição de subcategorias por faixas de quantidade:")
    print(f"  - 0 produtos: {qtd_zero} subcategorias")
    print(f"  - 1 a 10 produtos: {qtd_ate_10} subcategorias")
    print(f"  - 11 a 100 produtos: {qtd_11_100} subcategorias")
    print(f"  - 101 a 1000 produtos: {qtd_101_1000} subcategorias")
    print(f"  - Acima de 1000 produtos: {qtd_acima_1000} subcategorias")

    top_10 = df.sort_values("quantidade", ascending=False).head(10)
    total_top_10 = top_10["quantidade"].sum()
    perc_top_10 = total_top_10 / total_produtos * 100

    print(
        f"\nAs 10 maiores subcategorias representam {perc_top_10:.2f}% de todos os produtos "
        f"({total_top_10:,} de {total_produtos:,}).".replace(",", ".")
    )

    qtd_gaps_zero = qtd_zero
    qtd_gaps_ate_50 = (df["quantidade"].between(1, 50)).sum()

    print(f"Existem {qtd_gaps_zero} subcategorias com nenhum produto ofertado (possíveis gaps).")
    print(f"Existem {qtd_gaps_ate_50} subcategorias com 1 a 50 produtos (nichos potenciais ou baixa oferta).")

    print("\nResumo geral:")
    print(f"  - Total de subcategorias analisadas: {total_subcats}")
    print(f"  - Total de produtos mapeados: {total_produtos:,}".replace(",", "."))
    print(
        f"  - {qtd_zero + qtd_ate_10} subcategorias ("
        f"{(qtd_zero + qtd_ate_10) / total_subcats * 100:.2f}%) têm até 10 produtos ofertados."
    )


def gerar_analise_completa():
    print("--- INICIANDO ANÁLISE COMPLETA ---")

    try:
        data = carregar_json(ARQUIVO_JSON)
    except (FileNotFoundError, ValueError) as e:
        print(f"Erro ao carregar o arquivo JSON: {e}")
        return

    df, meta = preparar_dataframe(data)

    imprimir_metricas_gerais(meta)

    agregados = agregar_dados(df)

    resumo_geral = salvar_resumo_global(df)

    try:
        grafico_cauda_longa(resumo_geral)
    except Exception as e:
        print(f"Falha ao gerar gráfico de cauda longa global: {e}")

    try:
        grafico_histograma_global(resumo_geral)
    except Exception as e:
        print(f"Falha ao gerar histograma global: {e}")

    try:
        grafico_treemap_global(df)
    except Exception as e:
        print(f"Falha ao gerar treemap global: {e}")

    try:
        grafico_treemap_niveis(df)
    except Exception as e:
        print(f"Falha ao gerar treemap por níveis: {e}")

    try:
        grafico_barra_top_n(agregados["top_n_subcategorias"])
    except Exception as e:
        print(f"Falha ao gerar gráfico Top N: {e}")

    try:
        perc_subcats_80, perc_total_80 = grafico_pareto(df)
    except Exception as e:
        print(f"Falha ao gerar gráfico de Pareto: {e}")
        perc_subcats_80, perc_total_80 = 0.0, 0.0

    try:
        graficos_histogramas(df)
    except Exception as e:
        print(f"Falha ao gerar histogramas detalhados: {e}")

    try:
        grafico_cauda_longa_menores(agregados["subcategorias_maiores_zero"])
    except Exception as e:
        print(f"Falha ao gerar gráfico de cauda longa (menores > 0): {e}")

    try:
        salvar_tabela_gaps(agregados["gaps_subcategorias"])
    except Exception as e:
        print(f"Falha ao salvar/plotar gaps: {e}")

    try:
        gerar_insights(df, meta, agregados, perc_subcats_80, perc_total_80)
    except Exception as e:
        print(f"Falha ao gerar insights: {e}")

    print("\n--- RESUMO EXECUTIVO ---")
    total_produtos = resumo_geral["Qtd_Produtos"].sum()
    print(f"Total de Produtos Mapeados: {total_produtos:,.0f}")
    print(f"Total de Categorias (categoria_real) Únicas: {len(resumo_geral)}")
    if not resumo_geral.empty:
        print(
            f"Categoria #1: {resumo_geral.iloc[0]['Categoria']} "
            f"({resumo_geral.iloc[0]['Qtd_Produtos']:,.0f} itens)"
        )
    print(f"Mediana de Produtos por Categoria: {resumo_geral['Qtd_Produtos'].median():.0f}")
    print("Concluído!")


if __name__ == "__main__":
    gerar_analise_completa()
