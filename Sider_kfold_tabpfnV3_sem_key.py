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
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier  # Import unificado
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import (accuracy_score, hamming_loss, f1_score, 
                             average_precision_score)

warnings.filterwarnings("ignore")

# ==========================================
# ⚙️ CONFIGURAÇÕES GERAIS E CONTROLE
# ==========================================
SEED = 46785                  
N_FOLDS = 5                   
USAR_AMOSTRA = False         # Altere para False quando for rodar o SIDER cheio
TAMANHO_AMOSTRA = 100        
pasta_arquivos = 'C://' # Use barras normais no Windows para evitar quebras de string

print("\n" + "="*85)
print("💊 EXPERIMENTO SIDER: CROSS-VALIDATION (5 FOLDS) | BR vs CC | PR-AUC")
print("="*85)

# Carregamento do dataset...
df = pd.read_parquet(pasta_arquivos + 'dataset_sider_fingerprints.parquet')

if USAR_AMOSTRA:
    n_linhas = min(TAMANHO_AMOSTRA, len(df))
    df = df.sample(n=n_linhas, random_state=SEED).reset_index(drop=True)
    print(f"⚠️ MODO DE TESTE (AMOSTRA): Rodando com {n_linhas} moléculas aleatórias.")

X_cols = [str(i) for i in range(512)]
y_cols = [col for col in df.columns if col not in X_cols]

X = df[X_cols].values
y = df[y_cols].values

# ==========================================
# 🤖 INICIALIZAÇÃO DOS ESTIMADORES
# ==========================================
rf_base = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
xgb_base = XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=100, tree_method='hist', random_state=SEED, n_jobs=-1)

# De acordo com a documentação: Sem caminhos complexos, puríssimo. A v3 já é o default!
tabpfn_base = TabPFNClassifier(model_path="tabpfn-v3-classifier-v3_default.ckpt",device="cuda")

# Garante que o modelo seja carregado e inspecionado dinamicamente
print(f"\n⚙️  [VALIDAÇÃO] Motor local: {tabpfn_base.model_path.upper()} | Dispositivo: {str(tabpfn_base.device).upper()} | Pronto para o SIDER!\n")

modelos = {
    'Random Forest': rf_base, 
    'XGBoost': xgb_base, 
    'TabPFN': tabpfn_base
}


# 2. Configurando a Estratificação Iterativa (Padrão Ouro para Multirrótulo)
k_fold = MultilabelStratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
resultados_globais = []

# 3. O Super Loop de Validação Cruzada
for nome_modelo, modelo_base in modelos.items():
    print(f"\n⏳ Avaliando Modelo: {nome_modelo}...")
    
    # Dicionários para acumular as notas dos 5 Folds
    metricas_br = {'acc': [], 'h_loss': [], 'f1_samples': [], 'f1_macro': [], 'pr_auc': [], 'tempo': []}
    metricas_cc = {'acc': [], 'h_loss': [], 'f1_samples': [], 'f1_macro': [], 'pr_auc': [], 'tempo': []}
    
    n_jobs_multi = 1 if nome_modelo == 'TabPFN' else -1
   
    
    for fold, (train_idx, test_idx) in enumerate(k_fold.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # ==========================================
        # ESTRATÉGIA 1: BINARY RELEVANCE (BR)
        # ==========================================
        start_br = time.time()
        modelo_br = MultiOutputClassifier(modelo_base, n_jobs=n_jobs_multi).fit(X_train, y_train)
        y_pred_br = modelo_br.predict(X_test)
        
        # Engenharia para extrair probabilidades do BR (Para o PR-AUC)
        y_prob_list = modelo_br.predict_proba(X_test)
        try:
            y_prob_br = np.array([prob[:, 1] for prob in y_prob_list]).T
        except IndexError:
            y_prob_br = y_pred_br # Trava de segurança
            
        tempo_br = time.time() - start_br
        
        # Calculando as métricas e adicionando na lista do BR
        metricas_br['acc'].append(accuracy_score(y_test, y_pred_br) * 100)
        metricas_br['h_loss'].append(hamming_loss(y_test, y_pred_br))
        metricas_br['f1_samples'].append(f1_score(y_test, y_pred_br, average='samples', zero_division=0))
        metricas_br['f1_macro'].append(f1_score(y_test, y_pred_br, average='macro', zero_division=0))
        metricas_br['pr_auc'].append(average_precision_score(y_test, y_prob_br, average='macro'))
        metricas_br['tempo'].append(tempo_br)

        # ==========================================
        # ESTRATÉGIA 2: CLASSIFIER CHAINS (CC SIMPLES)
        # ==========================================
        start_cc = time.time()
        modelo_cc = ClassifierChain(modelo_base, order='random', random_state=SEED).fit(X_train, y_train)
        y_pred_cc = modelo_cc.predict(X_test)
        
        # O CC do Scikit-Learn já retorna a matriz de probabilidade direitinho!
        y_prob_cc = modelo_cc.predict_proba(X_test) 
        tempo_cc = time.time() - start_cc
        
        # Calculando as métricas e adicionando na lista do CC
        metricas_cc['acc'].append(accuracy_score(y_test, y_pred_cc) * 100)
        metricas_cc['h_loss'].append(hamming_loss(y_test, y_pred_cc))
        metricas_cc['f1_samples'].append(f1_score(y_test, y_pred_cc, average='samples', zero_division=0))
        metricas_cc['f1_macro'].append(f1_score(y_test, y_pred_cc, average='macro', zero_division=0))
        metricas_cc['pr_auc'].append(average_precision_score(y_test, y_prob_cc, average='macro'))
        metricas_cc['tempo'].append(tempo_cc)
        
        print(f"   -> Fold {fold}/5 finalizado | BR (F1: {metricas_br['f1_macro'][-1]:.4f}) vs CC (F1: {metricas_cc['f1_macro'][-1]:.4f})")

    # 4. Agregando a Média dos 5 Folds para exportar
    resultados_globais.append({
        'Modelo': f"{nome_modelo} (BR)",
        'Acuracia_Exata (%)': np.mean(metricas_br['acc']),
        'Hamming_Loss': np.mean(metricas_br['h_loss']),
        'F1_Samples (Example-based)': np.mean(metricas_br['f1_samples']),
        'F1_Macro (Label-based)': np.mean(metricas_br['f1_macro']),
        'PR_AUC': np.mean(metricas_br['pr_auc']),
        'Tempo_Medio_Treino (s)': np.mean(metricas_br['tempo'])
    })
    
    resultados_globais.append({
        'Modelo': f"{nome_modelo} (CC)",
        'Acuracia_Exata (%)': np.mean(metricas_cc['acc']),
        'Hamming_Loss': np.mean(metricas_cc['h_loss']),
        'F1_Samples (Example-based)': np.mean(metricas_cc['f1_samples']),
        'F1_Macro (Label-based)': np.mean(metricas_cc['f1_macro']),
        'PR_AUC': np.mean(metricas_cc['pr_auc']),
        'Tempo_Medio_Treino (s)': np.mean(metricas_cc['tempo'])
    })

# 5. Exportando a Tabela Final
print("\n" + "="*85)
df_res = pd.DataFrame(resultados_globais)

# Ordenando pelo PR-AUC, que é a métrica mais rigorosa do nosso teste
df_res = df_res.sort_values(by='PR_AUC', ascending=False)

df_res.to_csv('TABPFNV3_resultados_sider_cv_metricas.csv', index=False)
print("✅ Concluído! Arquivo 'resultados_sider_cv_metricas.csv' salvo com a consolidação dos 5 Folds.")

# Mostrando a tabela bonita no terminal
colunas_destaque = ['Modelo', 'F1_Samples (Example-based)', 'F1_Macro (Label-based)', 'PR_AUC']
try:
    from IPython.display import display
    display(df_res[colunas_destaque])
except ImportError:
    print(df_res[colunas_destaque].to_string(index=False))