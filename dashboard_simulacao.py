import streamlit as st
import pandas as pd
import plotly.express as px
import os
from simulador import executar_simulacao_dashboard
import plotly.graph_objects as go
import joblib
from datetime import date


# === FUNÇÃO DE RESET GLOBAL ===
def resetar_simulacao():
    caminho = "estado_estoque.csv"
    if os.path.exists(caminho):
        df = pd.read_csv(caminho, parse_dates=["data_atual"])
        if len(df) > 1:
            df = df.iloc[:-1]
            df.to_csv(caminho, index=False)
        else:
            os.remove(caminho)
        return True
    return False

# === AUTENTICAÇÃO ===
if "autenticado" not in st.session_state:
    st.session_state.autenticado = False

def autenticar(usuario, senha):
    return usuario == "gerente" and senha == "1234"

if not st.session_state.autenticado:
    st.markdown("## 🔒 Acesso Restrito")
    usuario = st.text_input("Usuário")
    senha = st.text_input("Senha", type="password")
    if st.button("Entrar"):
        if autenticar(usuario, senha):
            st.session_state.autenticado = True
            st.experimental_rerun()
        else:
            st.error("Usuário ou senha incorretos.")
    st.stop()

# === CONFIGURAÇÃO DA PÁGINA ===
st.set_page_config(page_title="Dashboard - Código Neural", layout="wide")

# === ESTILO VISUAL ===
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    font-size: 18px;
    color: #1f2937;
    background-color: #f1f5f9;
    margin: 0;
    padding: 0;
}
.block-container {
    background-color: #f8fafc;
    padding: 2rem 3rem;
    border-radius: 18px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.06);
}
h1, h2, h3 {
    font-weight: 700;
    color: #1d4ed8;
    margin-bottom: 0.3rem;
}
h1 { font-size: 2.4rem; }
h2 { font-size: 1.6rem; }
h3 { font-size: 1.2rem; }
.stMetric {
    background-color: #ffffff;
    padding: 1.5rem;
    border-radius: 16px;
    box-shadow: 0 3px 12px rgba(0, 0, 0, 0.08);
    border-left: 6px solid #60a5fa;
    transition: transform 0.2s ease;
}
.stMetric:hover { transform: scale(1.03); }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #f3f4f6 100%) !important;
    border-right: 2px solid #cbd5e1;
}
section[data-testid="stSidebar"] label { color: #0f172a !important; }
section[data-testid="stSidebar"] button[kind="primary"] {
    background: linear-gradient(to right, #3b82f6, #2563eb) !important;
    color: #ffffff !important;
    font-weight: bold;
    padding: 0.75rem 1.5rem !important;
    border-radius: 12px !important;
}
section[data-testid="stSidebar"] button[kind="primary"]:hover {
    background: linear-gradient(to right, #1d4ed8, #1e40af) !important;
    box-shadow: 0 4px 16px rgba(37, 99, 235, 0.3);
}
</style>
""", unsafe_allow_html=True)

# === TÍTULO ===
st.title("Dashboard de Operações")

# === FUNÇÕES DE CARREGAMENTO ===
@st.cache_data
def load_previsoes():
    df = pd.read_csv("relatorio_previsoes.csv", sep=",")
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        "Data": "Data",
        "SKU": "SKU",
        "Kg a Retirar Hoje": "Kg a Retirar Hoje",
        "Kg em Descongelamento D1": "Kg em Descongelamento",  # <- este é o nome correto
        "Kg Disponível para Venda": "Kg Pronto para Venda",
        "Perda Estimada": "Perda Estimada"
    })
    df["Data"] = pd.to_datetime(df["Data"], dayfirst=True)
    df["Kg a Retirar Hoje"] = pd.to_numeric(df["Kg a Retirar Hoje"], errors="coerce")
    df["Kg em Descongelamento"] = pd.to_numeric(
    df["Kg em Descongelamento"].astype(str).str.replace(" kg", "").str.replace(",", "."), errors="coerce"
    )
    df["Kg Pronto para Venda"] = pd.to_numeric(
        df["Kg Pronto para Venda"].astype(str).str.replace(" kg", "").str.replace(",", "."), errors="coerce"
    )
    df["Perda Estimada"] = pd.to_numeric(
        df["Perda Estimada"].astype(str).str.extract(r"([\d.,]+)")[0].str.replace(",", "."), errors="coerce"
    )
    return df

@st.cache_data
def load_estoque():
    estoque = pd.read_csv("estado_estoque.csv", parse_dates=["data_atual"])
    estoque["data_atual"] = estoque["data_atual"].dt.date  
    estoque = estoque.sort_values("data_atual")
    estoque["Perda Real"] = pd.to_numeric(estoque["perda_real"], errors="coerce").fillna(0)
    return estoque

@st.cache_resource
def carregar_modelo_ia():
    return joblib.load("modelo_vendas.joblib")

modelo = carregar_modelo_ia()

    
# === CARREGAMENTO DOS DADOS ===
df = load_previsoes()
estoque = load_estoque()


# === FILTROS ===
st.sidebar.header("Filtro de Período")
datas = df["Data"].dt.date.unique()
data_inicio = st.sidebar.date_input("Data Inicial", min_value=min(datas), max_value=max(datas), value=min(datas))
data_fim = st.sidebar.date_input("Data Final", min_value=min(datas), max_value=max(datas), value=max(datas))

dados_filtrados = df[(df["Data"].dt.date >= data_inicio) & (df["Data"].dt.date <= data_fim)]
estoque_filtrado = estoque[(estoque["data_atual"] >= data_inicio) & (estoque["data_atual"] <= data_fim)]

if st.sidebar.button("Recarregar Dados"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("## Simulação")
venda_real = st.sidebar.number_input("Venda Real do Dia (kg)", min_value=0.0, step=1.0, value=0.0)
if st.sidebar.button("Executar Próximo Dia"):
    sucesso = executar_simulacao_dashboard(venda_real)
    if sucesso:
        st.success("✅ Simulação executada com sucesso! Atualize o dashboard para ver os novos dados.")
    else:
        st.error("❌ Ocorreu um erro ao executar a simulação.")

if st.sidebar.button("Resetar Última Simulação"):
    sucesso = resetar_simulacao()
    if sucesso:
        st.success("🔁 Última simulação removida com sucesso!")
        st.rerun()
    else:
        st.warning("⚠️ Nenhuma simulação encontrada para resetar.")

st.sidebar.markdown("---")
st.sidebar.markdown("### Sessão")
if st.sidebar.button("Sair"):
    st.session_state.autenticado = False
    st.rerun()



# === CONTEÚDO PRINCIPAL ===
if dados_filtrados.empty:
    st.warning("⚠️ Nenhum dado encontrado para o intervalo selecionado.")
else:
    st.markdown("## Indicadores")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("📦 Kg a Retirar", f"{dados_filtrados['Kg a Retirar Hoje'].sum():.2f} kg")
    col2.metric("🧊 Kg em Descongelamento", f"{dados_filtrados['Kg em Descongelamento'].sum():.2f} kg")
    col3.metric("🛒 Pronto para Venda", f"{dados_filtrados['Kg Pronto para Venda'].sum():.2f} kg")
    col4.metric("🗑️ Perda Real", f"{estoque_filtrado['Perda Real'].sum():.2f} kg")

    # Calcular MAPE e RMSE se possível
    from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
    import numpy as np

    try:
        dados_venda = estoque_filtrado.copy()
        dados_venda["venda_real"] = pd.to_numeric(dados_venda["venda_real"], errors="coerce")
        dados_venda = dados_venda.dropna(subset=["venda_real", "kg_pronto_venda_dia1"])
        y_true = dados_venda["venda_real"]
        y_pred = dados_venda["kg_pronto_venda_dia1"]
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        col5.metric("📉 MAPE", f"{mape:.2f} %")
        col6.metric("📈 RMSE", f"{rmse:.2f} kg")
    except Exception:
        col5.metric("📉 MAPE", "N/A")
        col6.metric("📈 RMSE", "N/A")



    st.markdown("## Evolução Diária das Operações")
    fig1 = px.line(
        dados_filtrados,
        x="Data",
        y=["Kg a Retirar Hoje", "Kg em Descongelamento", "Kg Pronto para Venda"],
        labels={"value": "Kg", "variable": "Indicador"},
        title="Indicadores Operacionais por Dia",
        markers=True
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("## Análise de Perdas")
    estoque_filtrado["Ultrapassou Limite Diário"] = estoque_filtrado["Perda Real"] > 10
    total_perda = estoque_filtrado["Perda Real"].sum()
    dias = len(estoque_filtrado)
    limite_total = dias * 10
    ultrapassou_total = total_perda > limite_total

    VALOR_POR_KG = 12.50
    custo_total_perda = total_perda * VALOR_POR_KG
    estoque_filtrado["Custo da Perda"] = estoque_filtrado["Perda Real"] * VALOR_POR_KG


    if ultrapassou_total:
        st.error(f"🚨 Atenção: O total de perda ({total_perda:.2f} kg) ultrapassou o limite permitido para o intervalo selecionado ({limite_total:.2f} kg).")
    else:
        st.success(f"✅ Perda controlada! Total de {total_perda:.2f} kg dentro do limite de {limite_total:.2f} kg.")

    pie_data = estoque_filtrado["Ultrapassou Limite Diário"].value_counts().rename(index={
        True: "Acima do Limite", False: "Dentro do Limite"
    }).reset_index()
    pie_data.columns = ["Status", "Dias"]

    fig_pie = px.pie(
        pie_data,
        names="Status",
        values="Dias",
        title="Distribuição de Dias por Status de Perda",
        color="Status",
        color_discrete_map={
            "Dentro do Limite": "#10b981",
            "Acima do Limite": "#ef4444"
        }
    )
    st.plotly_chart(fig_pie, use_container_width=True)

        # === GRÁFICO DE BARRAS: Custo da Perda por Dia ===
    st.markdown("## 💰 Custo Diário das Perdas")

    fig_custo = px.bar(
        estoque_filtrado,
        x="data_atual",
        y="Custo da Perda",
        title="Custo das Perdas por Dia",
        labels={"data_atual": "Data", "Custo da Perda": "R$"},
        color="Custo da Perda",
        color_continuous_scale="reds"
    )
    fig_custo.update_layout(yaxis_tickprefix="R$ ")
    st.plotly_chart(fig_custo, use_container_width=True)


    st.markdown("## Indicador de Risco Financeiro")

    gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=custo_total_perda,
        delta={"reference": 500, "increasing": {"color": "red"}},
        gauge={
            "axis": {"range": [0, max(1000, custo_total_perda * 1.2)]},
            "bar": {"color": "darkred"},
            "steps": [
                {"range": [0, 300], "color": "#d1fae5"},
                {"range": [300, 500], "color": "#fef9c3"},
                {"range": [500, 1000], "color": "#fee2e2"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 500
            }
        },
        title={"text": "Custo Acumulado das Perdas (R$)"}
    ))
    st.plotly_chart(gauge, use_container_width=True)


    # === ANÁLISE DE SAZONALIDADE ===
    st.markdown("## Análise de Sazonalidade por Dia da Semana")

    sazonalidade = estoque_filtrado.copy()
    sazonalidade["data_atual"] = pd.to_datetime(sazonalidade["data_atual"])
    sazonalidade["Dia da Semana"] = sazonalidade["data_atual"].dt.day_name(locale="pt_BR")
    sazonalidade["venda_real"] = pd.to_numeric(sazonalidade["venda_real"], errors="coerce")

    # Agrupar e calcular média
    venda_media = sazonalidade.dropna(subset=["venda_real"]).groupby("Dia da Semana")["venda_real"].mean().reset_index()
    ordem_dias = ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira", "sábado", "domingo"]
    venda_media["Dia da Semana"] = venda_media["Dia da Semana"].str.lower()
    venda_media = venda_media.set_index("Dia da Semana").reindex(ordem_dias).reset_index()

    fig_sazonal = px.bar(
        venda_media,
        x="Dia da Semana",
        y="venda_real",
        labels={"venda_real": "Média de Vendas (kg)"},
        title="Tendência de Vendas por Dia da Semana",
        color="venda_real",
        color_continuous_scale="blues"
    )
    fig_sazonal.update_layout(xaxis_title="Dia da Semana", yaxis_title="Média (kg)")
    st.plotly_chart(fig_sazonal, use_container_width=True)

    # Insights visuais
    maior = venda_media.loc[venda_media["venda_real"].idxmax()]
    menor = venda_media.loc[venda_media["venda_real"].idxmin()]
    st.info(f" **Maior média:** {maior['Dia da Semana'].capitalize()} com **{maior['venda_real']:.2f} kg**")
    st.warning(f" **Menor média:** {menor['Dia da Semana'].capitalize()} com **{menor['venda_real']:.2f} kg**")


    # === SAZONALIDADE POR MÊS ===
    st.markdown("## Análise de Sazonalidade por Mês")

    sazonalidade["Mês"] = sazonalidade["data_atual"].dt.month_name(locale="pt_BR")
    venda_mensal = sazonalidade.dropna(subset=["venda_real"]).groupby("Mês")["venda_real"].mean().reset_index()

    # Ordem correta dos meses
    ordem_meses = [
        "janeiro", "fevereiro", "março", "abril", "maio", "junho",
        "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"
    ]
    venda_mensal["Mês"] = venda_mensal["Mês"].str.lower()
    venda_mensal = venda_mensal.set_index("Mês").reindex(ordem_meses).dropna().reset_index()

    fig_mes = px.line(
        venda_mensal,
        x="Mês",
        y="venda_real",
        title="Tendência de Vendas Médias por Mês",
        labels={"venda_real": "Média de Vendas (kg)"},
        markers=True
    )
    fig_mes.update_layout(xaxis_title="Mês", yaxis_title="Média (kg)")
    st.plotly_chart(fig_mes, use_container_width=True)

    # Insights visuais
    maior_mes = venda_mensal.loc[venda_mensal["venda_real"].idxmax()]
    menor_mes = venda_mensal.loc[venda_mensal["venda_real"].idxmin()]
    st.info(f" **Maior média mensal:** {maior_mes['Mês'].capitalize()} com **{maior_mes['venda_real']:.2f} kg**")
    st.warning(f"d **Menor média mensal:** {menor_mes['Mês'].capitalize()} com **{menor_mes['venda_real']:.2f} kg**")

    # === ALERTA DE VALIDADE / DESCARTE IMEDIATO ===
    st.markdown("## Alerta de Validade / Descarte Imediato")

    validade_alerta = estoque_filtrado.copy()
    validade_alerta["Dias Armazenado"] = (data_fim - validade_alerta["data_atual"]).apply(lambda x: x.days)
    validade_alerta = validade_alerta[
        (validade_alerta["kg_pronto_venda_dia2"] > 0) & 
        (validade_alerta["Dias Armazenado"] > 2)
    ]

    if validade_alerta.empty:
        st.success("✅ Nenhum lote vencido! Todo o estoque está dentro do prazo de validade.")
    else:
        st.error(f"🚨 {len(validade_alerta)} lote(s) com possível vencimento detectado(s).")
        st.dataframe(
            validade_alerta[["data_atual", "kg_pronto_venda_dia2", "Dias Armazenado"]],
            use_container_width=True
        )
        fig_alerta = px.bar(
            validade_alerta,
            x="data_atual",
            y="kg_pronto_venda_dia2",
            color="Dias Armazenado",
            labels={"data_atual": "Data", "kg_pronto_venda_dia2": "Kg Antigo"},
            title="Estoque Antigo Acima do Prazo de Validade",
        )
        st.plotly_chart(fig_alerta, use_container_width=True)

    st.markdown("## Evolução da Perda Real")
    fig2 = px.line(
        estoque_filtrado,
        x="data_atual",
        y="Perda Real",
        title="Perdas Reais (Frango fora do prazo)",
        labels={"data_atual": "Data", "Perda Real": "Perda (kg)"},
        markers=True
    )
    st.plotly_chart(fig2, use_container_width=True)


if "venda_real" in estoque_filtrado.columns:
    dados_venda = estoque_filtrado.copy()
    dados_venda["venda_real"] = pd.to_numeric(dados_venda["venda_real"], errors="coerce")
    dados_venda = dados_venda.dropna(subset=["venda_real"])

    if not dados_venda.empty:
        st.markdown("## Comparativo de Previsão vs Venda Real")

        grafico1 = px.line(
            dados_venda,
            x="data_atual",
            y=["kg_pronto_venda_dia1", "venda_real"],
            labels={"value": "Kg", "variable": "Indicador"},
            title="Venda Real x Estoque Disponível (Lote Novo)",
            markers=True
        )
        st.plotly_chart(grafico1, use_container_width=True)

        dados_venda["previsao_venda"] = dados_venda["kg_pronto_venda_dia1"]

        fig_venda = px.line(
            dados_venda,
            x="data_atual",
            y=["previsao_venda", "venda_real"],
            labels={"value": "Kg", "variable": "Tipo"},
            title="Comparação: Previsão vs Venda Real (Kg)",
            markers=True
        )
        fig_venda.update_traces(mode="lines+markers")
        st.plotly_chart(fig_venda, use_container_width=True)



    # Apenas datas futuras
    df_futuro = df[df["Data"].dt.date > estoque["data_atual"].max()]

    fig_forecast = px.line(
        df_futuro,
        x="Data",
        y=["Kg a Retirar Hoje", "Kg em Descongelamento", "Kg Pronto para Venda"],
        title="Tendência de Previsão para os Próximos Dias",
        markers=True
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

    st.markdown("## Comparativo Diário")
    fig3 = px.bar(
        dados_filtrados,
        x="Data",
        y=["Kg a Retirar Hoje", "Kg em Descongelamento", "Kg Pronto para Venda"],
        barmode="group",
        labels={"value": "Kg", "variable": "Categoria"},
        title="Comparação Diária dos Indicadores"
    )
    st.plotly_chart(fig3, use_container_width=True)

     # === TABELA FINAL COM DOWNLOAD ===
    st.markdown("## Tabela de Operações")

    # Juntar dados do dashboard com estado_estoque (para pegar Perda Real correta)
    tabela_final = dados_filtrados.copy()
    tabela_final["Perda Estimada"] = 10.0  # Força 10kg para todas as linhas
    perdas = estoque[["data_atual", "perda_real"]].rename(
    columns={"data_atual": "Data", "perda_real": "Perda Real"}
    )

    tabela_final = tabela_final.dropna(subset=["Data"])
    perdas = perdas.dropna(subset=["Data"])

    tabela_final["Data"] = tabela_final["Data"].dt.date
    perdas["Data"] = pd.to_datetime(perdas["Data"]).dt.date

    tabela_final = tabela_final.merge(perdas, on="Data", how="left")

    # Reorganizar colunas finais
    tabela_final = tabela_final[
        ["Data", "SKU", "Kg a Retirar Hoje", "Kg em Descongelamento", "Kg Pronto para Venda", "Perda Estimada", "Perda Real"]
    ]

    st.dataframe(tabela_final.set_index("Data"), use_container_width=True)

    # Arquivo Excel
    from io import BytesIO
    output = BytesIO()
    tabela_final.to_excel(output, index=False, sheet_name="Resumo Operacional")
    st.download_button(
        label="Baixar",
        data=output.getvalue(),
        file_name="tabela_operacoes_dashboard.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.markdown("""
    <hr style="margin-top: 3rem; margin-bottom: 1rem; border: none; border-top: 1px solid #cbd5e1;" />
    <div style="text-align: center; font-size: 15px; color: #64748b; padding-bottom: 1rem;">
        <p>© 2025 <strong>Código Neural</strong> - Todos os direitos reservados.</p>
        <p>Projeto Jovem Tech 7 — Grupo Mateus</p>
    </div>
""", unsafe_allow_html=True)