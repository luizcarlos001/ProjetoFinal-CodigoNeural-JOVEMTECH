import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import math
import numpy as np
from datetime import datetime, timedelta
import os
import logging
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# --- CONFIGURAÇÕES E CONSTANTES ---
ARQUIVO_DADOS_TREINO = 'dados.csv'
ARQUIVO_ESTADO_ESTOQUE = 'estado_estoque.csv'
ARQUIVO_RELATORIO_DIARIO = 'relatorio_diario.xlsx'
ARQUIVO_RELATORIO_PREVISOES = 'relatorio_previsoes.xlsx'
DIAS_VALIDADE_PRATELEIRA = 2
PESO_CAIXA_KG = 15.3
KG_MINIMO_A_DESCONGELAR = 5.0
SKU_PRODUTO = '384706'

logging.getLogger('prophet').setLevel(logging.ERROR)

# --- FUNÇÕES DE GERENCIAMENTO ---

def resetar_estado():
    if os.path.exists(ARQUIVO_ESTADO_ESTOQUE):
        os.remove(ARQUIVO_ESTADO_ESTOQUE); print(f"✅ Estado do estoque resetado.")
    else:
        print(" Nenhum estado de estoque encontrado para resetar.")

def carregar_ou_iniciar_estoque(data_inicial, previsoes_df):
    if os.path.exists(ARQUIVO_ESTADO_ESTOQUE):
        print(f"\n Carregando estado do estoque de '{ARQUIVO_ESTADO_ESTOQUE}'...")
        df_estoque = pd.read_csv(ARQUIVO_ESTADO_ESTOQUE, parse_dates=['data_atual'])
        return df_estoque.iloc[-1]
    else:
        print("\n Iniciando nova simulação e populando estoque inicial com previsões...")
        previsao_hoje = previsoes_df.loc[previsoes_df['ds'] == data_inicial, 'yhat'].iloc[0] if not previsoes_df[previsoes_df['ds'] == data_inicial].empty else 0
        previsao_amanha = previsoes_df.loc[previsoes_df['ds'] == data_inicial + timedelta(days=1), 'yhat'].iloc[0] if not previsoes_df[previsoes_df['ds'] == data_inicial + timedelta(days=1)].empty else 0
        return pd.Series({
            'data_atual': data_inicial,
            'kg_em_descongelamento': max(0, previsao_amanha), # O que estará pronto amanhã
            'kg_pronto_venda_dia1': max(0, previsao_hoje),      # O que está pronto hoje (lote novo)
            'kg_pronto_venda_dia2': 0.0,                      # O que está pronto hoje (lote antigo)
        })

def gerar_relatorio_previsoes(forecast_df, data_inicio):
    previsoes_futuras = forecast_df[forecast_df['ds'] >= data_inicio].copy()
    lista_relatorio_previsao = []
    for data_base in previsoes_futuras['ds']:
        data_d1 = data_base + timedelta(days=1); data_d2 = data_base + timedelta(days=2)
        if data_d2 in previsoes_futuras['ds'].values:
            kg_disponivel_hoje = previsoes_futuras.loc[previsoes_futuras['ds'] == data_base, 'yhat'].iloc[0]
            kg_em_descongelamento = previsoes_futuras.loc[previsoes_futuras['ds'] == data_d1, 'yhat'].iloc[0]
            kg_a_retirar = previsoes_futuras.loc[previsoes_futuras['ds'] == data_d2, 'yhat'].iloc[0]
            caixas_a_retirar = math.ceil(max(0, kg_a_retirar) / PESO_CAIXA_KG)
            lista_relatorio_previsao.append({
                'Data da Retirada': data_base.strftime('%d/%m/%Y'), 'SKU (frango)': SKU_PRODUTO,
                'Kg a Retirar (Caixas)': f"{max(0, kg_a_retirar):.2f} kg ({caixas_a_retirar} cx)",
                'Kg em Descongelamento (pronto amanhã)': f"{max(0, kg_em_descongelamento):.2f} kg",
                'Kg Disponível para Venda Hoje (virtual)': f"{max(0, kg_disponivel_hoje):.2f} kg",
                'Projeção de Perdas (Kg/Caixas)': "0.00 kg (0 cx)"
            })
    if lista_relatorio_previsao:
        df_relatorio = pd.DataFrame(lista_relatorio_previsao)
        df_relatorio.to_excel(ARQUIVO_RELATORIO_PREVISOES, index=False)
        df_relatorio.to_csv('relatorio_previsoes.csv', index=False, sep=',', decimal='.')
        print(f"✅ Relatórios de previsões puras '{ARQUIVO_RELATORIO_PREVISOES}' e 'relatorio_previsoes.csv' foram gerados.")

# --- FUNÇÃO PRINCIPAL DA SIMULAÇÃO ---
def executar_rodada_diaria(estado_atual, previsoes_df):
    hoje = estado_atual['data_atual']
    print("\n" + "="*50); print(f"🗓️  SIMULAÇÃO PARA O DIA: {hoje.strftime('%d/%m/%Y')} (Hoje)"); print("="*50)

    kg_pronto_venda_hoje_total = estado_atual['kg_pronto_venda_dia1'] + estado_atual['kg_pronto_venda_dia2']
    print(f"📦 Estoque disponível para venda hoje: {kg_pronto_venda_hoje_total:.2f} kg")
    print(f"   - Lote novo (1º dia): {estado_atual['kg_pronto_venda_dia1']:.2f} kg"); print(f"   - Lote antigo (2º dia): {estado_atual['kg_pronto_venda_dia2']:.2f} kg"); print("-" * 50)
    
    while True:
        try:
            venda_real_hoje_str = input(f"✅ Insira a VENDA REAL de frango (kg) de HOJE ({hoje.strftime('%d/%m/%Y')}): ")
            venda_real_hoje = float(venda_real_hoje_str.replace(',', '.')); break
        except ValueError: print("❌ Entrada inválida.")

    venda_do_lote_antigo = min(venda_real_hoje, estado_atual['kg_pronto_venda_dia2'])
    perda_real_hoje_kg = estado_atual['kg_pronto_venda_dia2'] - venda_do_lote_antigo
    venda_restante = venda_real_hoje - venda_do_lote_antigo
    venda_do_lote_novo = min(venda_restante, estado_atual['kg_pronto_venda_dia1'])
    sobra_lote_novo = estado_atual['kg_pronto_venda_dia1'] - venda_do_lote_novo
    print(f"✔️  Ok. Sobra para amanhã (do lote novo de hoje): {sobra_lote_novo:.2f} kg")

    previsao_amanha = previsoes_df.loc[previsoes_df['ds'] == hoje + timedelta(days=1), 'yhat'].iloc[0] if not previsoes_df.loc[previsoes_df['ds'] == hoje + timedelta(days=1)].empty else 0
    perda_projetada_amanha_kg = max(0, sobra_lote_novo - previsao_amanha)
    perda_projetada_amanha_caixas = math.ceil(perda_projetada_amanha_kg / PESO_CAIXA_KG) if perda_projetada_amanha_kg > 0 else 0

    print(f"\n--- Resumo do Dia ({hoje.strftime('%d/%m/%Y')}) ---")
    if perda_real_hoje_kg > 0: print(f"🗑️ Perda Realizada Hoje (lote de ontem expirou): {perda_real_hoje_kg:.2f} kg")
    else: print(" Nenhuma perda real de produto hoje!")

    data_alvo_previsao = hoje + timedelta(days=2)
    previsao_kg = previsoes_df.loc[previsoes_df['ds'] == data_alvo_previsao, 'yhat'].iloc[0] if not previsoes_df.loc[previsoes_df['ds'] == data_alvo_previsao].empty else 0.0
    kg_a_descongelar_calculado = max(0, previsao_kg - sobra_lote_novo)
    kg_a_descongelar_hoje = kg_a_descongelar_calculado if kg_a_descongelar_calculado > 0 else KG_MINIMO_A_DESCONGELAR
    
    if hoje.day == 23:
        kg_a_descongelar_hoje = 130.0
        print(f"⚠️ ATENÇÃO: Regra especial do dia 23 ativada para garantir o estoque do dia 25.")
        print(f"   - Kg a descongelar hoje foi forçado para: {kg_a_descongelar_hoje:.2f} kg.")

    caixas_a_retirar = math.ceil(kg_a_descongelar_hoje / PESO_CAIXA_KG)
    if estado_atual['kg_pronto_venda_dia2'] > 0: idade_lote_virtual = '2 dias'
    elif estado_atual['kg_pronto_venda_dia1'] > 0: idade_lote_virtual = '1 dia'
    else: idade_lote_virtual = 'N/A'
        
    relatorio_df = pd.DataFrame([{'Data da Retirada': hoje.strftime('%d/%m/%Y'), 'SKU (frango)': SKU_PRODUTO,
        'Kg a Retirar (Caixas)': f"{kg_a_descongelar_hoje:.2f} kg ({caixas_a_retirar} cx)",
        'Kg em Descongelamento (pronto amanhã)': f"{estado_atual['kg_em_descongelamento']:.2f} kg",
        'Kg Disponível para Venda Hoje ': f"{kg_pronto_venda_hoje_total:.2f} kg",
        'Idade do Lote Descongelado ': idade_lote_virtual,
        'Projeção de Perdas (Kg/Caixas)': f"{perda_projetada_amanha_kg:.2f} kg ({perda_projetada_amanha_caixas} cx)"
    }])
    relatorio_df.to_excel(ARQUIVO_RELATORIO_DIARIO, index=False)
    print(f"\n✅ Relatório de ação diária '{ARQUIVO_RELATORIO_DIARIO}' foi gerado.")

    estado_amanha = {'data_atual': hoje + timedelta(days=1),
        'kg_em_descongelamento': kg_a_descongelar_hoje,
        'kg_pronto_venda_dia1': estado_atual['kg_em_descongelamento'],
        'kg_pronto_venda_dia2': sobra_lote_novo,
    }
    return pd.Series(estado_amanha)

# --- BLOCO DE EXECUÇÃO PRINCIPAL ---
if __name__ == "__main__":
    print("--- Ferramenta de Gestão de Estoque de Frango ---")
    acao = input("O que você deseja fazer?\n 1 - Executar para o próximo dia\n 2 - Resetar toda a simulação\n 3 - Sair\nEscolha uma opção: ")

    if acao == '2': resetar_estado()
    elif acao == '1':
        print("\n Carregando dados de treino..."); df = pd.read_csv(ARQUIVO_DADOS_TREINO, sep=',', decimal=',', thousands='.', parse_dates=['data_dia'])
        df_prophet = df.rename(columns={'data_dia': 'ds', 'total_venda_dia_kg': 'y'}); df_prophet.dropna(subset=['ds'], inplace=True)
        
        print(" Treinando o modelo Prophet..."); modelo = Prophet(); modelo.fit(df_prophet)
        future = modelo.make_future_dataframe(periods=30); forecast = modelo.predict(future); print(" Modelo treinado e previsão gerada.")
        
        ultima_data_real = df_prophet['ds'].max(); data_de_partida = ultima_data_real + timedelta(days=1)
        
        gerar_relatorio_previsoes(forecast, data_de_partida)

        estado_atual = carregar_ou_iniciar_estoque(data_de_partida, forecast)
        proximo_estado = executar_rodada_diaria(estado_atual, forecast)
        
        df_proximo_estado = pd.DataFrame([proximo_estado])
        df_proximo_estado.to_csv(ARQUIVO_ESTADO_ESTOQUE, mode='a', header=not os.path.exists(ARQUIVO_ESTADO_ESTOQUE), index=False)
        print(f"\nEstado para amanhã salvo com sucesso no histórico '{ARQUIVO_ESTADO_ESTOQUE}'.")
        
        print("\n" + "="*50); print("📊 MÉTRICAS DE AVALIAÇÃO DO MODELO (vs. Dados Históricos)"); print("="*50)
        df_metrics_comparison = forecast.set_index('ds')[['yhat']].join(df_prophet.set_index('ds')[['y']].rename(columns={'y': 'y_real'})); df_metrics_comparison.dropna(inplace=True)

        if not df_metrics_comparison.empty:
            mape = mean_absolute_percentage_error(df_metrics_comparison['y_real'], df_metrics_comparison['yhat']) * 100
            rmse = np.sqrt(mean_squared_error(df_metrics_comparison['y_real'], df_metrics_comparison['yhat']))
            print(f"MAPE (Erro Percentual Médio): {mape:.2f}%"); print(f"RMSE (Erro Absoluto Médio): {rmse:.2f} kg")
        else:
            print("\nNão há dados históricos suficientes para calcular MAPE e RMSE.")

        print("\nProcesso finalizado.")
    else:
        print("Saindo do programa.")