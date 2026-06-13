import os
# ⚙️ CONFIGURAÇÃO DE REDE E AUTENTICAÇÃO OFICIAL (V8.0.3)
os.environ['TABPFN_DISABLE_TELEMETRY'] = '1'

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Insira aqui a sua API Key longa que você pegou no painel da PriorLabs
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
os.environ['TABPFN_TOKEN'] = ""
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

import time
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier  # Import unificado oficial

# 2. IMPORTANDO TODAS AS MÉTRICAS
from sklearn.metrics import (accuracy_score, hamming_loss, f1_score, 
                             precision_score, recall_score, jaccard_score)

warnings.filterwarnings("ignore")

# ==========================================
# ⚙️ CONFIGURAÇÕES GERAIS E CONTROLE
# ==========================================
SEED = 46785                  # Semente aleatória fixa
N_FOLDS = 5                   # Número de dobras para a validação cruzada
USAR_AMOSTRA = False           # Troque para False no experimento oficial
TAMANHO_AMOSTRA = 150         # Amostra para testes rápidos
pasta_arquivos = 'C://' # Ajustado para barras normais do Windows


print("\n" + "="*85)
print(f"🧪 EXPERIMENTO SIDER: SNIPER LOOP + {N_FOLDS}-FOLD CV")
print("="*85)

# Carregando a base SIDER
df_completo = pd.read_parquet(pasta_arquivos+'dataset_sider_fingerprints.parquet')

if USAR_AMOSTRA:
    n_linhas = min(TAMANHO_AMOSTRA, len(df_completo))
    df_completo = df_completo.sample(n=n_linhas, random_state=SEED).reset_index(drop=True)
    print(f"⚠️ MODO DE TESTE: Base reduzida para {n_linhas} moléculas.")

X_cols = [str(i) for i in range(512)]
y_cols = [col for col in df_completo.columns if col not in X_cols]

resultados_detalhados = []
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# ==========================================
# 🤖 INICIALIZAÇÃO FIXA DOS ESTIMADORES (Otimizado fora do loop)
# ==========================================
rf_base = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
xgb_base = XGBClassifier(n_estimators=100, random_state=SEED, n_jobs=-1, eval_metric='logloss')

# Forçando o mesmo checkpoint default do SIDER multirrótulo para garantir paridade metodológica
tabpfn_base = TabPFNClassifier(
    model_path="tabpfn-v3-classifier-v3_default.ckpt",
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

print(f"\n⚙️  [VALIDAÇÃO] Motor local: {tabpfn_base.model_path.upper()} | Dispositivo: {str(tabpfn_base.device).upper()} | Pronto para as Vias do SIDER!\n")

modelos = {
    'Random Forest': rf_base,
    'XGBoost': xgb_base,
    'TabPFN': tabpfn_base
}

# 3. O SNIPER LOOP COM K-FOLD
for idx, via in enumerate(y_cols, 1):
    print(f"\n🔬 [{idx}/{len(y_cols)}] Analisando Sistema: {via}")
    
    # Isolar a classe e dropar NaNs específicos
    df_alvo = df_completo[X_cols + [via]].dropna(subset=[via])
    X = df_alvo[X_cols].values
    y = df_alvo[via].astype(int).values
    
    # Validação mínima para permitir a estratificação do K-Fold
    if y.sum() < N_FOLDS or (len(y) - y.sum()) < N_FOLDS:
        print(f"   ⚠️ Ignorada: Poucos casos positivos ou negativos para {N_FOLDS} folds.")
        continue
        
    print(f"   📊 Treinando com {len(y)} amostras | Casos Positivos: {y.sum()}")

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Contagem específica do conjunto de teste
        casos_positivos_teste = y_test.sum()
        
        for nome_modelo, modelo in modelos.items():
            start = time.time()
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)
            tempo = time.time() - start
            
            resultados_detalhados.append({
                'Sistema_Orgao': via,
                'Fold': fold,
                'Modelo': nome_modelo,
                'Amostras_Validas': len(y),
                'Casos_Positivos_Teste': casos_positivos_teste,
                'Acuracia': accuracy_score(y_test, y_pred) * 100,
                'Hamming_Loss': hamming_loss(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
                'Recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
                'F1_Macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
                'F1_Micro': f1_score(y_test, y_pred, average='micro', zero_division=0),
                'Jaccard_Score': jaccard_score(y_test, y_pred, average='macro', zero_division=0),
                'Tempo_Execucao': tempo
            })

    # Salvamento Incremental (Checkpoint seguro por iteração de via)
    pd.DataFrame(resultados_detalhados).to_csv(pasta_arquivos+'TABPFNV3_resultados_sider_sniper_kfold_bruto.csv', index=False)

# =========================================================
# 📊 4. CONSOLIDAÇÃO FINAL (Médias dos Folds)
# =========================================================
if resultados_detalhados:
    df_res = pd.DataFrame(resultados_detalhados)
    
    # 1. Arquivo por Sistema (Média das 5 dobras por modelo/via)
    df_final_vias = df_res.groupby(['Sistema_Orgao', 'Modelo']).mean(numeric_only=True).reset_index().drop(columns='Fold')
    df_final_vias.to_csv(pasta_arquivos+'TABPFNV3_sider_metricas_vias_individuais_kfold.csv', index=False)
    
    # 2. Relatório Global (Consolidado de todos os sistemas)
    relatorio_global = []

    for modelo in df_final_vias['Modelo'].unique():
        df_mod = df_final_vias[df_final_vias['Modelo'] == modelo]
        
        # F1-Micro Global: Média ponderada pelo suporte (Amostras_Validas) de cada sistema
        f1_micro_global = (df_mod['F1_Micro'] * df_mod['Amostras_Validas']).sum() / df_mod['Amostras_Validas'].sum()
        
        # Outras métricas globais: Média aritmética simples
        precision_global = df_mod['Precision'].mean()
        f1_macro_global = df_mod['F1_Macro'].mean()
        acuracia_global = df_mod['Acuracia'].mean()
        
        relatorio_global.append({
            'Modelo': modelo,
            'Acuracia_Global': acuracia_global,
            'Precision_Global': precision_global,
            'F1_Micro_Global': f1_micro_global,
            'F1_Macro_Global': f1_macro_global,
            'Sistemas_Processados': len(df_mod)
        })

    df_global = pd.DataFrame(relatorio_global)
    df_global.to_csv(pasta_arquivos+'TABPFNV3_sider_relatorio_final_kfold.csv', index=False)

    print("\n" + "="*85)
    print("✅ SUCESSO! Arquivos 'sider_metricas_vias_individuais_kfold.csv' e 'sider_relatorio_final_kfold.csv' gerados.")
    print("="*85)