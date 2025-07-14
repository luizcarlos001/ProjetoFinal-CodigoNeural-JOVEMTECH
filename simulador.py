# simulador.py
import pandas as pd
import math
import numpy as np
import os
from datetime import timedelta
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

ARQUIVO_DADOS_TREINO = 'dados.csv'
ARQUIVO_ESTADO_ESTOQUE = 'estado_estoque.csv'
ARQUIVO_RELATORIO_PREVISOES = 'relatorio_previsoes.csv'
PESO_CAIXA_KG = 15.3
KG_MINIMO_A_DESCONGELAR = 5.0
SKU_PRODUTO = '384706'

def executar_simulacao_dashboard(venda_real_hoje: float) -> bool:
    try:
        # 1. Dados e modelo Prophet
        df = pd.read_csv(ARQUIVO_DADOS_TREINO, sep=',', decimal=',', thousands='.', parse_dates=['data_dia'])
        df_prophet = df.rename(columns={'data_dia': 'ds', 'total_venda_dia_kg': 'y'})
        df_prophet.dropna(subset=['ds'], inplace=True)

        modelo = Prophet()
        modelo.fit(df_prophet)
        future = modelo.make_future_dataframe(periods=30)
        forecast = modelo.predict(future)

        ultima_data_real = df_prophet['ds'].max()
        data_hoje = ultima_data_real + timedelta(days=1)

        # 2. Estado atual (última linha ou iniciar)
        if os.path.exists(ARQUIVO_ESTADO_ESTOQUE):
            estoque_df = pd.read_csv(ARQUIVO_ESTADO_ESTOQUE, parse_dates=['data_atual'])
            estado_atual = estoque_df.iloc[-1]
        else:
            previsao_hoje = forecast.loc[forecast['ds'] == data_hoje, 'yhat'].iloc[0]
            previsao_amanha = forecast.loc[forecast['ds'] == data_hoje + timedelta(days=1), 'yhat'].iloc[0]
            estado_atual = pd.Series({
                'data_atual': data_hoje,
                'kg_em_descongelamento': max(0, previsao_amanha),
                'kg_pronto_venda_dia1': max(0, previsao_hoje),
                'kg_pronto_venda_dia2': 0.0
            })

        # 3. Simulação
        hoje = estado_atual['data_atual']
        venda = venda_real_hoje
        venda_lote_antigo = min(venda, estado_atual['kg_pronto_venda_dia2'])
        perda_real = max(0, (
            estado_atual.get('kg_pronto_venda_dia2', 0.0)
            - min(venda_real_hoje, estado_atual.get('kg_pronto_venda_dia2', 0.0))
        ))
        venda_restante = venda - venda_lote_antigo
        venda_lote_novo = min(venda_restante, estado_atual['kg_pronto_venda_dia1'])
        sobra_novo = estado_atual['kg_pronto_venda_dia1'] - venda_lote_novo

        previsao_d2 = forecast.loc[forecast['ds'] == hoje + timedelta(days=2), 'yhat'].iloc[0] if not forecast[forecast['ds'] == hoje + timedelta(days=2)].empty else 0
        kg_a_descongelar = max(0, previsao_d2 - sobra_novo)
        if kg_a_descongelar <= 0:
            kg_a_descongelar = KG_MINIMO_A_DESCONGELAR
        if hoje.day == 23:
            kg_a_descongelar = 130.0

        estado_novo = pd.DataFrame([{
            'data_atual': hoje + timedelta(days=1),
            'kg_descongelando_d1': kg_a_descongelar,
            'kg_descongelando_d2': estado_atual.get('kg_descongelando_d1', 0.0),
            'kg_pronto_venda_dia1': estado_atual.get('kg_descongelando_d2', 0.0),
            'kg_pronto_venda_dia2': sobra_novo
        }])



        estado_novo.to_csv(ARQUIVO_ESTADO_ESTOQUE, mode='a', header=not os.path.exists(ARQUIVO_ESTADO_ESTOQUE), index=False)
        df_estado = pd.read_csv(ARQUIVO_ESTADO_ESTOQUE)
        df_estado.loc[df_estado.index[-1], "venda_real"] = venda_real_hoje
        df_estado.to_csv(ARQUIVO_ESTADO_ESTOQUE, index=False)

        # === Recalcula perda_real após salvar ===
        recalcular_perdas()

        df_relatorio = pd.DataFrame({
            "Data": forecast.loc[forecast["ds"] >= data_hoje, "ds"].dt.date,
            "SKU": SKU_PRODUTO,
            "Kg a Retirar Hoje": forecast.loc[forecast["ds"] >= data_hoje, "yhat"].shift(2),
            "Kg em Descongelamento D1": forecast.loc[forecast["ds"] >= data_hoje, "yhat"].shift(1),
            "Kg Disponível para Venda": forecast.loc[forecast["ds"] >= data_hoje, "yhat"],
            "Perda Estimada": 0.0
        })
        df_relatorio.to_csv(ARQUIVO_RELATORIO_PREVISOES, index=False)

        return True
    except Exception as e:
        print(f"[ERRO] {e}")
        return False


def recalcular_perdas():
    df = pd.read_csv(ARQUIVO_ESTADO_ESTOQUE, parse_dates=["data_atual"])
    df = df.sort_values("data_atual").reset_index(drop=True)

    perdas = [0.0]  # primeiro dia não tem perda

    for i in range(1, len(df)):
        perda = max(0, df.loc[i - 1, "kg_pronto_venda_dia2"])
        perdas.append(perda)

    df["perda_real"] = perdas
    df.to_csv(ARQUIVO_ESTADO_ESTOQUE, index=False)