# pip install pandas matplotlib seaborn plotly

import json
import os
from typing import Tuple, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from urllib.parse import urlparse

ARQUIVO_JSON = "mapeamento_veiculos_final.json"

# Parâmetros de análise
TOP_N_SUBCATEGORIAS = 20          
N_MENORES_SUBCATEGORIAS = 30      
LIMIAR_GAP_QTD = 10              


def extrair_slug_final(url: str):
    """
    Extrai o último pedaço relevante da URL, que representa a subcategoria final.
    Ex.: .../acesso-lateral/barras-passo-nerf/_NoIndex_True -> 'Barras Passo Nerf'
    """
    if not isinstance(url, str) or not url:
        return pd.NA

    # Remove query string e hash
    url_limpa = url.split("?")[0].split("#")[0]

    partes = [p for p in url_limpa.split("/") if p and "_NoIndex" not in p]

    if len(partes) < 2:
        return pd.NA

    slug = partes[-1]

    # Se o último for algo genérico, não considerar como nível final
    if slug in ("exterior", "acesso-lateral", "veiculos", "acessorios-veiculos"):
        return pd.NA

    slug = slug.replace("-", " ").strip()
    if not slug:
        return pd.NA

    # Capitaliza cada palavra
    return slug.title()

def carregar_json(caminho: str) -> Dict[str, Any]:
    """
    Carrega o arquivo JSON e retorna um dicionário Python.
    """
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


def preparar_dataframe(data: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    detalhes = data["detalhes"]

    # Cria DataFrame a partir da lista de detalhes
    df = pd.DataFrame(detalhes)

    # Garante que as colunas esperadas existam
    for col in ["categoria", "quantidade", "link"]:
        if col not in df.columns:
            raise ValueError(f"Coluna obrigatória '{col}' ausente em 'detalhes'.")

    # Quebra da coluna 'categoria' em níveis hierárquicos
    # Limita a 4 níveis (nivel_1 a nivel_4)
    categorias_split = df["categoria"].fillna("").str.split(" > ", n=3, expand=True)
    categorias_split = categorias_split.rename(
        columns={0: "nivel_1", 1: "nivel_2", 2: "nivel_3", 3: "nivel_4"}
    )

    # Anexa colunas de níveis ao DataFrame original
    df = pd.concat([df, categorias_split], axis=1)

    # Normaliza strings vazias para NaN para facilitar filtros posteriores
    df[["nivel_1", "nivel_2", "nivel_3", "nivel_4"]] = (
        df[["nivel_1", "nivel_2", "nivel_3", "nivel_4"]]
        .replace("", pd.NA)
    )

    # =========================
    # Nível final via URL (nivel_5)
    # =========================
    df["nivel_5"] = df["link"].apply(extrair_slug_final)

    # Conversão da coluna 'quantidade' para numérico
    df["quantidade_original"] = df["quantidade"]  # copia para debug se necessário
    df["quantidade"] = pd.to_numeric(df["quantidade"], errors="coerce")

    # Trata possíveis valores não numéricos
    qtd_na = df["quantidade"].isna().sum()
    if qtd_na > 0:
        print(f"Atenção: {qtd_na} linha(s) não puderam ser convertidas para numérico em 'quantidade'. Serão tratadas como 0.")
        df["quantidade"] = df["quantidade"].fillna(0).astype(int)
    else:
        df["quantidade"] = df["quantidade"].astype(int)

    # Cria coluna com categoria completa (níveis concatenados)
    def montar_categoria_completa(row):
        niveis = [
            row.get("nivel_1"),
            row.get("nivel_2"),
            row.get("nivel_3"),
            row.get("nivel_4"),
            row.get("nivel_5"),  # <- novo nível vindo da URL
        ]
        niveis = [n for n in niveis if pd.notna(n)]
        if niveis:
            return " / ".join(niveis)
        # fallback para a string original
        return row.get("categoria", "")

    df["categoria_completa"] = df.apply(montar_categoria_completa, axis=1)

    # Calcula participação percentual em relação ao total
    total_quantidade = df["quantidade"].sum()
    df["participacao_%"] = (df["quantidade"] / total_quantidade * 100).round(4)

    # Metadados do JSON
    meta = {
        "total_acumulado_veiculos": data.get("total_acumulado_veiculos"),
        "ultima_atualizacao": data.get("ultima_atualizacao"),
        "categorias_concluidas_count": data.get("categorias_concluidas_count"),
        "total_quantidade_calculado": int(total_quantidade),
        "total_linhas": int(len(df)),
    }

    return df, meta

def imprimir_metricas_gerais(meta: Dict[str, Any]) -> None:
    """
    Imprime métricas gerais e compara total calculado com o total do JSON.
    """
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
    """
    Cria agregações por diferentes níveis e DataFrames auxiliares.
    Retorna um dicionário com vários DataFrames.
    """
    resultados = {}

    # Agregação por nível 1 (macro categoria)
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

    # Agregação por nível 1 + nível 2
    agg_n1_n2 = (
        df.groupby(["nivel_1", "nivel_2"], dropna=False)["quantidade"]
        .sum()
        .reset_index()
        .rename(columns={"quantidade": "quantidade_total"})
        .sort_values("quantidade_total", ascending=False)
    )
    resultados["agg_nivel_1_2"] = agg_n1_n2

    # Agregação por nível 3 (quando existir)
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

    # DataFrame com top N subcategorias (linhas originais com maior quantidade)
    top_n = df.sort_values("quantidade", ascending=False).head(TOP_N_SUBCATEGORIAS).copy()
    resultados["top_n_subcategorias"] = top_n

    # Subcategorias com menor quantidade (até 50 produtos, por exemplo)
    menores_ate_50 = df[df["quantidade"] <= 50].sort_values("quantidade", ascending=True).copy()
    resultados["subcategorias_ate_50"] = menores_ate_50

    # Subcategorias com quantidade == 0
    iguais_zero = df[df["quantidade"] == 0].copy()
    resultados["subcategorias_zero"] = iguais_zero

    # Subcategorias com quantidade > 0 (útil para cauda longa / histogramas sem zeros)
    maiores_zero = df[df["quantidade"] > 0].copy()
    resultados["subcategorias_maiores_zero"] = maiores_zero

    # Gaps (quantidade <= LIMIAR_GAP_QTD)
    gaps = df[df["quantidade"] <= LIMIAR_GAP_QTD].copy()
    resultados["gaps_subcategorias"] = gaps

    return resultados


# ====================
# Gráficos
# ====================

def gerar_treemap(df: pd.DataFrame, arquivo_saida: str = "treemap_categorias.html") -> None:
    """
    Gera um treemap da hierarquia nivel_1 > nivel_2 > nivel_3 usando Plotly
    e salva em arquivo HTML.
    """
    print(f"Gerando treemap: {arquivo_saida}")

    # Agrupa por combinação de níveis para evitar duplicidade
    df_tree = (
        df.groupby(["nivel_1", "nivel_2", "nivel_3"], dropna=False)["quantidade"]
        .sum()
        .reset_index()
    )

    # Preenche NaN com rótulos genéricos para o gráfico
    for col in ["nivel_1", "nivel_2", "nivel_3"]:
        df_tree[col] = df_tree[col].fillna(f"Sem {col}")

    fig = px.treemap(
        df_tree,
        path=["nivel_1", "nivel_2", "nivel_3"],
        values="quantidade",
        color="quantidade",
        color_continuous_scale="Blues",
        title="Treemap de Categorias - Veículos (Mercado Livre)",
    )

    fig.write_html(arquivo_saida)


def gerar_barra_top_n(df_top: pd.DataFrame, arquivo_saida: str = "top20_subcategorias.png") -> None:
    """
    Gera gráfico de barras horizontais com as Top N subcategorias.
    """
    print(f"Gerando gráfico de barras Top {len(df_top)}: {arquivo_saida}")

    # Ordena para exibição (maior para menor)
    df_plot = df_top.sort_values("quantidade", ascending=True)  # ascending para barh (última é maior)

    plt.figure(figsize=(10, max(6, len(df_plot) * 0.4)))
    sns.barplot(
        data=df_plot,
        x="quantidade",
        y="categoria_completa",
        orient="h",
    )
    plt.xlabel("Quantidade de produtos")
    plt.ylabel("Subcategoria (nível mais detalhado)")
    plt.title(f"Top {len(df_plot)} Subcategorias por Quantidade de Produtos")
    plt.tight_layout()
    plt.savefig(arquivo_saida, dpi=300)
    plt.close()


def gerar_pareto(df: pd.DataFrame, arquivo_saida: str = "pareto_subcategorias.png") -> Tuple[float, float]:
    """
    Gera gráfico de Pareto considerando as subcategorias (linhas originais).
    Retorna (perc_subcategorias_80, perc_total_80) para uso nos insights.
    """
    print(f"Gerando gráfico de Pareto: {arquivo_saida}")

    df_pareto = df.sort_values("quantidade", ascending=False).reset_index(drop=True)
    df_pareto["quantidade_acumulada"] = df_pareto["quantidade"].cumsum()
    total = df_pareto["quantidade"].sum()
    df_pareto["perc_acumulado"] = df_pareto["quantidade_acumulada"] / total * 100

    # Encontra o ponto em que atingimos (ou passamos) 80% do total
    idx_80 = (df_pareto["perc_acumulado"] >= 80).idxmax()
    qtd_subcats_80 = idx_80 + 1  # idx é 0-based
    total_subcats = len(df_pareto)
    perc_subcats_80 = qtd_subcats_80 / total_subcats * 100
    perc_total_80 = df_pareto.loc[idx_80, "perc_acumulado"]

    # Gráfico
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Barras (quantidade)
    ax1.bar(df_pareto.index, df_pareto["quantidade"])
    ax1.set_xlabel("Subcategorias (ordenadas por quantidade)")
    ax1.set_ylabel("Quantidade de produtos", color="black")

    # Eixo secundário para a linha de % acumulada
    ax2 = ax1.twinx()
    ax2.plot(df_pareto.index, df_pareto["perc_acumulado"], marker="o")
    ax2.set_ylabel("% acumulado", color="black")
    ax2.axhline(80, color="red", linestyle="--", linewidth=1)
    ax2.set_ylim(0, 100)

    plt.title("Gráfico de Pareto - Subcategorias (Veículos)")
    plt.tight_layout()
    plt.savefig(arquivo_saida, dpi=300)
    plt.close()

    # Mensagem de Pareto
    print(
        f"As {qtd_subcats_80} maiores subcategorias ({perc_subcats_80:.2f}% do total) "
        f"respondem por {perc_total_80:.2f}% dos produtos."
    )

    return perc_subcats_80, perc_total_80


def gerar_histogramas(df: pd.DataFrame,
                      arquivo_linear: str = "histograma_quantidade.png",
                      arquivo_log: str = "histograma_quantidade_log.png") -> None:
    """
    Gera histogramas da distribuição da quantidade por subcategoria:
    - Escala normal
    - Escala logarítmica no eixo Y
    """
    print(f"Gerando histograma (escala linear): {arquivo_linear}")
    plt.figure(figsize=(10, 6))
    plt.hist(df["quantidade"], bins=50)
    plt.xlabel("Quantidade de produtos por subcategoria")
    plt.ylabel("Frequência")
    plt.title("Distribuição da quantidade por subcategoria (escala linear)")
    plt.tight_layout()
    plt.savefig(arquivo_linear, dpi=300)
    plt.close()

    print(f"Gerando histograma (escala logarítmica no eixo Y): {arquivo_log}")
    plt.figure(figsize=(10, 6))
    plt.hist(df["quantidade"], bins=50)
    plt.yscale("log")
    plt.xlabel("Quantidade de produtos por subcategoria")
    plt.ylabel("Frequência (escala log)")
    plt.title("Distribuição da quantidade por subcategoria (escala logarítmica no eixo Y)")
    plt.tight_layout()
    plt.savefig(arquivo_log, dpi=300)
    plt.close()


def gerar_cauda_longa(df_maiores_zero: pd.DataFrame,
                      arquivo_saida: str = "subcategorias_menores.png") -> None:
    """
    Gera gráfico de barras horizontais com as N subcategorias com menor quantidade (> 0).
    """
    n = min(N_MENORES_SUBCATEGORIAS, len(df_maiores_zero))
    print(f"Gerando gráfico de cauda longa (menores > 0, n={n}): {arquivo_saida}")

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
    plt.savefig(arquivo_saida, dpi=300)
    plt.close()


def salvar_tabela_gaps(df_gaps: pd.DataFrame,
                       arquivo_csv: str = "gaps_subcategorias.csv",
                       arquivo_grafico: str = "gaps_subcategorias.png") -> None:
    """
    Salva tabela de gaps (quantidade = 0 ou muito baixa) em CSV.
    Opcionalmente gera um gráfico de barras horizontais.
    """
    if df_gaps is None:
        print("DataFrame de gaps é None; nada a salvar.")
        return

    # Trabalha em uma cópia para não mexer no DF original
    df_gaps = df_gaps.copy()

    # Garante que 'quantidade' é numérica inteira (0 a LIMIAR_GAP_QTD)
    df_gaps["quantidade"] = (
        pd.to_numeric(df_gaps["quantidade"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    print(f"Salvando tabela de gaps em CSV: {arquivo_csv}")
    # Mantém apenas colunas relevantes (incluindo nivel_5, se existir)
    cols = [
        "categoria_completa",
        "categoria",
        "nivel_1",
        "nivel_2",
        "nivel_3",
        "nivel_4",
        "nivel_5",      # novo nível, se existir
        "quantidade",
        "link",
    ]
    cols = [c for c in cols if c in df_gaps.columns]
    df_gaps[cols].to_csv(arquivo_csv, index=False, encoding="utf-8-sig")

    if df_gaps.empty:
        print("Não há gaps (quantidade <= LIMIAR_GAP_QTD) para plotar.")
        return

    # Gera gráfico (por exemplo, top 30 menores dentro dos gaps)
    n = min(30, len(df_gaps))
    df_plot = (
        df_gaps
        .sort_values(["quantidade", "categoria_completa"], ascending=[True, True])
        .head(n)
        .copy()
    )

    print(f"Gerando gráfico de gaps (n={n}): {arquivo_grafico}")
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

    # Garante eixo X começando em 0 e indo até o máximo + 1
    max_qtd = max(df_plot["quantidade"].max(), 0)
    plt.xlim(0, max_qtd + 1)

    plt.tight_layout()
    plt.savefig(arquivo_grafico, dpi=300)
    plt.close()


# ====================
# Insights em texto
# ====================

def gerar_insights(df: pd.DataFrame,
                   meta: Dict[str, Any],
                   agregados: Dict[str, pd.DataFrame],
                   perc_subcats_80: float,
                   perc_total_80: float) -> None:
    """
    Gera e imprime alguns insights automáticos no console.
    """
    print("\n=== Insights Automáticos ===")

    total_produtos = meta["total_quantidade_calculado"]
    total_subcats = meta["total_linhas"]

    # Top 5 categorias (nível 1)
    agg_n1 = agregados.get("agg_nivel_1")
    if agg_n1 is not None:
        print("\nTop 5 categorias (Nível 1) por quantidade de produtos:")
        top5_n1 = agg_n1.head(5)
        for _, row in top5_n1.iterrows():
            print(
                f"  - {row['nivel_1']}: {row['quantidade_total']:,} produtos "
                f"({row['participacao_%']:.2f}% do total)".replace(",", ".")
            )

    # Regra de Pareto aproximada (80/20)
    print(
        f"\nRegra de Pareto aproximada: "
        f"{perc_subcats_80:.2f}% das subcategorias respondem por {perc_total_80:.2f}% do total de produtos."
    )

    # Quantidade de subcategorias por faixas de quantidade
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

    # Frases em texto
    # X maiores subcategorias (por exemplo, Top 10) representam Y% do total
    top_10 = df.sort_values("quantidade", ascending=False).head(10)
    total_top_10 = top_10["quantidade"].sum()
    perc_top_10 = total_top_10 / total_produtos * 100

    print(
        f"\nAs 10 maiores subcategorias representam {perc_top_10:.2f}% de todos os produtos "
        f"({total_top_10:,} de {total_produtos:,}).".replace(",", ".")
    )

    # Gaps e nichos potenciais
    qtd_gaps_zero = qtd_zero
    qtd_gaps_ate_50 = (df["quantidade"].between(1, 50)).sum()

    print(f"Existem {qtd_gaps_zero} subcategorias com nenhum produto ofertado (possíveis gaps).")
    print(f"Existem {qtd_gaps_ate_50} subcategorias com 1 a 50 produtos (nichos potenciais ou baixa oferta).")

    print("\nResumo geral:")
    print(
        f"  - Total de subcategorias analisadas: {total_subcats}"
    )
    print(
        f"  - Total de produtos mapeados: {total_produtos:,}".replace(",", ".")
    )
    print(
        f"  - {qtd_zero + qtd_ate_10} subcategorias ("
        f"{(qtd_zero + qtd_ate_10) / total_subcats * 100:.2f}%) têm até 10 produtos ofertados."
    )


def main():
    """
    Script principal para:
    - Ler o JSON de categorias de veículos do Mercado Livre
    - Organizar a hierarquia em DataFrame
    - Gerar agregações
    - Criar gráficos (treemap, top N, Pareto, histogramas, cauda longa, gaps)
    - Imprimir insights em texto
    - Salvar gráficos e tabelas na pasta local
    """
    try:
        data = carregar_json(ARQUIVO_JSON)
    except (FileNotFoundError, ValueError) as e:
        print(f"Erro ao carregar o arquivo JSON: {e}")
        return

    # Prepara DataFrame e metadados
    df, meta = preparar_dataframe(data)

    # Imprime métricas gerais
    imprimir_metricas_gerais(meta)

    # Agregações e DataFrames auxiliares
    agregados = agregar_dados(df)

    # Geração dos gráficos
    try:
        gerar_treemap(df, "treemap_categorias.html")
    except Exception as e:
        print(f"Falha ao gerar treemap: {e}")

    try:
        gerar_barra_top_n(agregados["top_n_subcategorias"], "top20_subcategorias.png")
    except Exception as e:
        print(f"Falha ao gerar gráfico Top 20: {e}")

    try:
        perc_subcats_80, perc_total_80 = gerar_pareto(df, "pareto_subcategorias.png")
    except Exception as e:
        print(f"Falha ao gerar gráfico de Pareto: {e}")
        # Fallback para não quebrar os insights
        perc_subcats_80, perc_total_80 = 0.0, 0.0

    try:
        gerar_histogramas(df, "histograma_quantidade.png", "histograma_quantidade_log.png")
    except Exception as e:
        print(f"Falha ao gerar histogramas: {e}")

    try:
        gerar_cauda_longa(agregados["subcategorias_maiores_zero"], "subcategorias_menores.png")
    except Exception as e:
        print(f"Falha ao gerar gráfico de cauda longa: {e}")

    try:
        salvar_tabela_gaps(agregados["gaps_subcategorias"], "gaps_subcategorias.csv", "gaps_subcategorias.png")
    except Exception as e:
        print(f"Falha ao salvar/plotar gaps: {e}")

    # Geração de insights em texto
    try:
        gerar_insights(df, meta, agregados, perc_subcats_80, perc_total_80)
    except Exception as e:
        print(f"Falha ao gerar insights: {e}")


if __name__ == "__main__":
    main()
