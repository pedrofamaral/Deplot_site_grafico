import json
import os
from typing import Dict, Any, Tuple

import pandas as pd
import matplotlib.pyplot as plt
from urllib.parse import urlparse

# =====================
# CONFIGURAÇÕES GERAIS
# =====================

ARQUIVOS_DISPONIVEIS = {
    "veiculos": "mapeamento_veiculos_atualizados.json",
    "agro": "Agro_infos.json",
}
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

INPUT_JSON = os.path.join(BASE_DIR, ARQUIVOS_DISPONIVEIS["veiculos", "agro"])
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "graficos")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tabelas")
HTML_DIR = os.path.join(OUTPUT_DIR, "html")

for d in (OUTPUT_DIR, FIGURES_DIR, TABLES_DIR, HTML_DIR):
    os.makedirs(d, exist_ok=True)

TOP_N_SUBCATEGORIAS = 20
LIMIAR_GAP_QTD = 10


# =====================
# CARREGAMENTO E PREPARO
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

    # garantias básicas
    for col in ["categoria", "quantidade", "link"]:
        if col not in df.columns:
            raise ValueError(f"Coluna obrigatória ausente: {col}")

    # quebra da árvore de categorias
    categorias_split = df["categoria"].fillna("").str.split(" > ", expand=True)
    col_names = [f"nivel_{i+1}" for i in range(categorias_split.shape[1])]
    categorias_split.columns = col_names
    df = pd.concat([df, categorias_split], axis=1)
    df[col_names] = df[col_names].replace("", pd.NA)

    # categoria derivada do link (mais confiável em alguns casos)
    df["categoria_real"] = df["link"].apply(extrair_slug_corrigido)

    # quantidade numérica
    df["quantidade"] = pd.to_numeric(df["quantidade"], errors="coerce").fillna(0).astype(int)

    # categoria completa legível (para diretoria)
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
        "total_acumulado_veiculos_json": data.get("total_acumulado_veiculos"),
        "ultima_atualizacao": data.get("ultima_atualizacao"),
        "categorias_concluidas_count": data.get("categorias_concluidas_count"),
        "total_quantidade_calculado": total_quantidade,
        "total_linhas": int(len(df)),
    }
    return df, meta


# =====================
# AGREGAÇÕES
# =====================

def agregar_dados(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    resultados: Dict[str, pd.DataFrame] = {}

    # Nível 1 (macro categorias)
    agg_n1 = (
        df.groupby("nivel_1", dropna=False)["quantidade"]
        .sum()
        .reset_index()
        .rename(columns={"quantidade": "quantidade_total"})
        .sort_values("quantidade_total", ascending=False)
    )
    total = agg_n1["quantidade_total"].sum()
    agg_n1["participacao_%"] = (agg_n1["quantidade_total"] / total * 100).round(2)
    resultados["agg_nivel_1"] = agg_n1

    # Nível 1 + 2
    agg_n1_n2 = (
        df.groupby(["nivel_1", "nivel_2"], dropna=False)["quantidade"]
        .sum()
        .reset_index()
        .rename(columns={"quantidade": "quantidade_total"})
        .sort_values("quantidade_total", ascending=False)
    )
    resultados["agg_nivel_1_2"] = agg_n1_n2

    # top N subcategorias por quantidade
    top_n = df.sort_values("quantidade", ascending=False).head(TOP_N_SUBCATEGORIAS).copy()
    resultados["top_n_subcategorias"] = top_n

    # gaps: até LIMIAR_GAP_QTD
    gaps = df[df["quantidade"] <= LIMIAR_GAP_QTD].copy()
    resultados["gaps_subcategorias"] = gaps

    # subcategorias com mais de 0 produtos (para cauda longa de menores)
    maiores_zero = df[df["quantidade"] > 0].copy()
    resultados["subcategorias_maiores_zero"] = maiores_zero

    # subcategorias com 0
    iguais_zero = df[df["quantidade"] == 0].copy()
    resultados["subcategorias_zero"] = iguais_zero

    return resultados


# =====================
# TABELAS E CSV
# =====================

def salvar_resumo_global(df: pd.DataFrame) -> pd.DataFrame:
    resumo = (
        df.groupby("categoria_completa")["quantidade"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    resumo.columns = ["Categoria", "Qtd_Produtos"]

    total_produtos = resumo["Qtd_Produtos"].sum()
    resumo["% do Total"] = (resumo["Qtd_Produtos"] / total_produtos * 100)
    resumo["% Acumulado"] = resumo["% do Total"].cumsum()
    resumo["Rank"] = resumo.index + 1

    caminho = os.path.join(TABLES_DIR, "relatorio_completo_categorias.csv")
    resumo.to_csv(caminho, index=False, sep=";", decimal=",", encoding="utf-8-sig")
    print(f"[OK] Resumo completo salvo em: {caminho}")
    return resumo


def salvar_tabela_gaps(df_gaps: pd.DataFrame) -> None:
    caminho = os.path.join(TABLES_DIR, "tabela_gaps_subcategorias.csv")
    if df_gaps.empty:
        print("[INFO] Não há gaps (qtd <= LIMIAR_GAP_QTD) para salvar.")
        return

    cols = [
        "categoria_completa",
        "categoria",
        "nivel_1",
        "nivel_2",
        "nivel_3",
        "categoria_real",
        "quantidade",
        "link",
    ]
    cols = [c for c in cols if c in df_gaps.columns]
    df_gaps[cols].to_csv(caminho, index=False, encoding="utf-8-sig")
    print(f"[OK] Tabela de gaps salva em: {caminho}")


# =====================
# GRÁFICOS (PNG) PARA DIRETORIA
# =====================

def grafico_top_n(df_top: pd.DataFrame) -> None:
    caminho = os.path.join(FIGURES_DIR, "top_subcategorias.png")
    df_plot = df_top.sort_values("quantidade", ascending=True)

    plt.figure(figsize=(11, max(6, len(df_plot) * 0.4)))
    plt.barh(df_plot["categoria_completa"], df_plot["quantidade"])
    plt.xlabel("Quantidade de produtos")
    plt.ylabel("Subcategoria")
    plt.title(f"Top {len(df_plot)} subcategorias por quantidade de produtos")
    plt.tight_layout()
    plt.savefig(caminho, dpi=300)
    plt.close()
    print(f"[OK] Gráfico Top N salvo em: {caminho}")


def grafico_cauda_longa_menores(df_maiores_zero: pd.DataFrame) -> None:
    caminho = os.path.join(FIGURES_DIR, "subcategorias_menores.png")
    df_plot = df_maiores_zero.sort_values("quantidade", ascending=True).head(30)

    if df_plot.empty:
        print("[INFO] Não há subcategorias > 0 para cauda longa.")
        return

    plt.figure(figsize=(11, max(6, len(df_plot) * 0.4)))
    plt.barh(df_plot["categoria_completa"], df_plot["quantidade"])
    plt.xlabel("Quantidade de produtos (> 0)")
    plt.ylabel("Subcategorias com menor quantidade")
    plt.title("Menores subcategorias com produtos (cauda longa)")
    plt.tight_layout()
    plt.savefig(caminho, dpi=300)
    plt.close()
    print(f"[OK] Gráfico cauda longa (menores) salvo em: {caminho}")


def grafico_histograma(df: pd.DataFrame) -> None:
    caminho = os.path.join(FIGURES_DIR, "histograma_subcategorias.png")

    plt.figure(figsize=(10, 6))
    plt.hist(df["quantidade"], bins=50)
    plt.xlabel("Quantidade de produtos por subcategoria")
    plt.ylabel("Número de subcategorias")
    plt.title("Distribuição de quantidade por subcategoria")
    plt.tight_layout()
    plt.savefig(caminho, dpi=300)
    plt.close()
    print(f"[OK] Histograma salvo em: {caminho}")


# =====================
# INSIGHTS E DASHBOARD HTML SIMPLES
# =====================

def gerar_resumo_executivo(meta: Dict[str, Any], agregados: Dict[str, pd.DataFrame]) -> str:
    total_produtos = meta["total_quantidade_calculado"]
    total_subcats = meta["total_linhas"]
    zeros = agregados["subcategorias_zero"]
    gaps = agregados["gaps_subcategorias"]

    texto = []
    texto.append(f"Total de produtos mapeados: {total_produtos:,}".replace(",", "."))
    texto.append(f"Total de subcategorias analisadas: {total_subcats}")
    texto.append(f"Subcategorias sem nenhum produto: {len(zeros)}")
    texto.append(
        f"Subcategorias com até {LIMIAR_GAP_QTD} produtos (possíveis gaps): {len(gaps)}"
    )

    top_n1 = agregados["agg_nivel_1"].head(5)
    texto.append("")
    texto.append("Top 5 categorias de nível 1 por volume:")
    for _, row in top_n1.iterrows():
        linha = (
            f"- {row['nivel_1']}: {row['quantidade_total']:,} produtos "
            f"({row['participacao_%']:.2f}% do total)"
        ).replace(",", ".")
        texto.append(linha)

    resumo_txt = "\n".join(texto)
    caminho = os.path.join(OUTPUT_DIR, "resumo_executivo.txt")
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(resumo_txt)

    print(f"[OK] Resumo executivo salvo em: {caminho}")
    return resumo_txt


def gerar_dashboard_html(resumo_txt: str) -> None:
    caminho = os.path.join(HTML_DIR, "dashboard_veiculos.html")

    html = f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <title>Painel - Veículos (Marketplace)</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body {{
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0f172a;
      color: #e5e7eb;
    }}
    header {{
      padding: 16px 32px;
      background: #020617;
      border-bottom: 1px solid #1f2937;
      position: sticky;
      top: 0;
      z-index: 10;
    }}
    header h1 {{
      margin: 0;
      font-size: 1.4rem;
    }}
    main {{
      max-width: 1200px;
      margin: 24px auto 40px;
      padding: 0 16px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 2fr 1.5fr;
      gap: 24px;
    }}
    .card {{
      background: #020617;
      border-radius: 12px;
      padding: 20px 20px 16px;
      border: 1px solid #1f2937;
      box-shadow: 0 18px 45px rgba(15,23,42,0.6);
    }}
    .card h2 {{
      margin: 0 0 8px;
      font-size: 1.1rem;
      color: #e5e7eb;
    }}
    .card p, .card pre {{
      margin: 0;
      font-size: 0.9rem;
      color: #9ca3af;
      white-space: pre-wrap;
    }}
    .thumbs {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin-top: 12px;
    }}
    .thumb {{
      background: #020617;
      border-radius: 10px;
      border: 1px dashed #374151;
      padding: 8px;
      text-align: center;
      font-size: 0.8rem;
      color: #9ca3af;
    }}
    img {{
      max-width: 100%;
      border-radius: 8px;
      display: block;
      margin-bottom: 6px;
    }}
    footer {{
      text-align: center;
      font-size: 0.75rem;
      color: #6b7280;
      margin-bottom: 24px;
    }}
    code {{
      background: #111827;
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 0.8rem;
    }}
  </style>
</head>
<body>
<header>
  <h1>Visão de categorias - Veículos (Marketplace)</h1>
</header>
<main>
  <section class="grid">
    <article class="card">
      <h2>Resumo executivo</h2>
      <pre>{resumo_txt}</pre>
      <p style="margin-top: 10px;">
        Arquivos detalhados (CSVs e gráficos em PNG) foram gerados na pasta <code>output/</code>
        para uso da TI em análises mais profundas ou integração com Power BI.
      </p>
    </article>
    <article class="card">
      <h2>Como usar este painel</h2>
      <p>
        Este painel é estático, pensado para diretoria: basta abrir este arquivo HTML em qualquer navegador.
        Os gráficos foram gerados em alta resolução para uso em apresentações, relatórios e reuniões.
      </p>
      <p style="margin-top: 8px;">
        Recomendações:
      </p>
      <ul style="padding-left: 18px; margin: 4px 0; font-size: 0.85rem; color: #9ca3af;">
        <li>Use o gráfico de Top Subcategorias para discutir foco comercial.</li>
        <li>Use o histograma para mostrar distribuição geral de oferta.</li>
        <li>Use a tabela de gaps para priorizar categorias com baixa ou nenhuma oferta.</li>
      </ul>
    </article>
  </section>

  <section style="margin-top: 24px;" class="card">
    <h2>Gráficos gerados</h2>
    <div class="thumbs">
      <div class="thumb">
        <img src="../graficos/top_subcategorias.png" alt="Top subcategorias" />
        Top subcategorias por quantidade de produtos
      </div>
      <div class="thumb">
        <img src="../graficos/histograma_subcategorias.png" alt="Histograma" />
        Distribuição de produtos por subcategoria
      </div>
      <div class="thumb">
        <img src="../graficos/subcategorias_menores.png" alt="Menores subcategorias" />
        Menores subcategorias com produtos (cauda longa)
      </div>
    </div>
  </section>
</main>
<footer>
  Painel gerado automaticamente a partir do mapeamento de categorias de veículos.
</footer>
</body>
</html>
"""

    with open(caminho, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[OK] Dashboard HTML salvo em: {caminho}")


# =====================
# ORQUESTRAÇÃO
# =====================

def main() -> None:
    print("=== Análise de categorias - Veículos ===")
    print(f"Lendo JSON em: {INPUT_JSON}")

    data = carregar_json(INPUT_JSON)
    df, meta = preparar_dataframe(data)
    print(f"[OK] DataFrame carregado com {meta['total_linhas']} linhas.")

    agregados = agregar_dados(df)

    # CSVs
    resumo_geral = salvar_resumo_global(df)
    salvar_tabela_gaps(agregados["gaps_subcategorias"])

    # Gráficos
    grafico_top_n(agregados["top_n_subcategorias"])
    grafico_cauda_longa_menores(agregados["subcategorias_maiores_zero"])
    grafico_histograma(df)

    # Resumo + HTML
    resumo_txt = gerar_resumo_executivo(meta, agregados)
    gerar_dashboard_html(resumo_txt)

    print("\nFinalizado. Verifique a pasta 'output/' para arquivos gerados.")


if __name__ == "__main__":
    main()
