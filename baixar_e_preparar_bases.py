import warnings
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

# 1. IMPORTANDO O CONTROLADOR DE LOGS DO RDKIT
from rdkit import RDLogger 

warnings.filterwarnings("ignore")

# 2. SILENCIANDO O CONSOLE
RDLogger.DisableLog('rdApp.*') 

print("\n" + "="*75)
print("🧬 DATA PREP BRACIS: EXTRAINDO TOX21, SIDER E TOXCAST (MoleculeNet)")
print("="*75 + "\n")

morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)

# 3. LISTA GLOBAL DE AUDITORIA DE ERROS
auditoria_erros = []

def processar_tox21():
    print("📥 Baixando Tox21 diretamente do servidor do MoleculeNet...")
    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz'
    df_raw = pd.read_csv(url)
    
    print("💾 Salvando cópia local da base Tox21 crua (tox21_raw.parquet)...")
    df_raw.to_parquet('tox21_raw.parquet', index=False)
    
    smiles_list = df_raw['smiles'].tolist()
    print(f"🔬 Convertendo {len(smiles_list)} moléculas (SMILES) do Tox21 em Fingerprints...")
    
    X_features = []
    indices_validos = [] # <-- GUARDA AS LINHAS QUE DERAM CERTO
    
    for idx, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = morgan_gen.GetFingerprint(mol)
            X_features.append(np.array(fp))
            indices_validos.append(idx)
        else:
            # SE FALHOU: Não adiciona zeros. Apenas anota o erro.
            auditoria_erros.append({
                'Base': 'Tox21',
                'Linha_Index': idx,
                'SMILES_Original': smiles,
                'Motivo': 'Excluída: Falha na conversão (RDKit retornou None)'
            })
            
    X = pd.DataFrame(X_features)
    X.columns = X.columns.astype(str)
    
    y_cols = [col for col in df_raw.columns if col.lower() not in ['smiles', 'mol_id']]
    # RECORTA O Y USANDO OS ÍNDICES VÁLIDOS
    y = df_raw[y_cols].iloc[indices_validos].reset_index(drop=True)
    
    df_final = pd.concat([X, y], axis=1)
    df_final.to_parquet('dataset_tox21_fingerprints.parquet', index=False)
    
    print("✅ Concluído! Salvo como 'dataset_tox21_fingerprints.parquet'")
    print(f"   -> Amostras Originais: {len(smiles_list)}")
    print(f"   -> Amostras Válidas (Mantidas): {df_final.shape[0]} | Rótulos: {y.shape[1]}\n")


def processar_sider():
    print("📥 Baixando SIDER diretamente do servidor do MoleculeNet...")
    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz'
    df_raw = pd.read_csv(url)
    
    print("💾 Salvando cópia local da base SIDER crua (sider_raw.parquet)...")
    df_raw.to_parquet('sider_raw.parquet', index=False)
    
    smiles_list = df_raw['smiles'].tolist()
    print(f"🔬 Convertendo {len(smiles_list)} moléculas (SMILES) do SIDER em Fingerprints...")
    
    X_features = []
    indices_validos = []
    
    for idx, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = morgan_gen.GetFingerprint(mol)
            X_features.append(np.array(fp))
            indices_validos.append(idx)
        else:
            auditoria_erros.append({
                'Base': 'SIDER',
                'Linha_Index': idx,
                'SMILES_Original': smiles,
                'Motivo': 'Excluída: Falha na conversão (RDKit retornou None)'
            })
            
    X = pd.DataFrame(X_features)
    X.columns = X.columns.astype(str)
    
    y_cols = [col for col in df_raw.columns if col.lower() != 'smiles']
    y = df_raw[y_cols].iloc[indices_validos].reset_index(drop=True)
    
    df_final = pd.concat([X, y], axis=1)
    df_final.to_parquet('dataset_sider_fingerprints.parquet', index=False)
    
    print("✅ Concluído! Salvo como 'dataset_sider_fingerprints.parquet'")
    print(f"   -> Amostras Originais: {len(smiles_list)}")
    print(f"   -> Amostras Válidas (Mantidas): {df_final.shape[0]} | Rótulos: {y.shape[1]}\n")


def processar_toxcast():
    print("📥 Baixando ToxCast diretamente do servidor do MoleculeNet...")
    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/toxcast_data.csv.gz'
    df_raw = pd.read_csv(url)
    
    print("💾 Salvando cópia local da base ToxCast crua (toxcast_raw.parquet)...")
    df_raw.to_parquet('toxcast_raw.parquet', index=False)
    
    smiles_list = df_raw['smiles'].tolist()
    
    print(f"🔬 Convertendo {len(smiles_list)} moléculas (SMILES) do ToxCast em Fingerprints...")
    X_features = []
    indices_validos = []
    
    for idx, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = morgan_gen.GetFingerprint(mol)
            X_features.append(np.array(fp))
            indices_validos.append(idx)
        else:
            auditoria_erros.append({
                'Base': 'ToxCast',
                'Linha_Index': idx,
                'SMILES_Original': smiles,
                'Motivo': 'Excluída: Falha na conversão (RDKit retornou None)'
            })
            
    X = pd.DataFrame(X_features)
    X.columns = X.columns.astype(str) 
    
    y_cols = [col for col in df_raw.columns if col.lower() != 'smiles']
    y = df_raw[y_cols].iloc[indices_validos].reset_index(drop=True)
    
    df_final = pd.concat([X, y], axis=1)
    df_final.to_parquet('dataset_toxcast_fingerprints.parquet', index=False)
    
    print("✅ Concluído! Salvo como 'dataset_toxcast_fingerprints.parquet'")
    print(f"   -> Amostras Originais: {len(smiles_list)}")
    print(f"   -> Amostras Válidas (Mantidas): {df_final.shape[0]} | Rótulos: {y.shape[1]}\n")


# Rodando os três processamentos em sequência
processar_tox21()
processar_sider()
processar_toxcast()

# ==========================================
# 4. EXPORTANDO O RELATÓRIO DE AUDITORIA
# ==========================================
print("="*75)
if auditoria_erros:
    df_erros = pd.DataFrame(auditoria_erros)
    df_erros.to_csv('relatorio_moleculas_rejeitadas.csv', index=False)
    print(f"⚠️ ATENÇÃO: Foram banidas {len(auditoria_erros)} moléculas defeituosas do treinamento.")
    print("📋 O relatório detalhado foi salvo como 'relatorio_moleculas_rejeitadas.csv'")
else:
    print("✅ Excelente! Nenhuma molécula foi rejeitada pelo RDKit nas 3 bases.")

print("\n🎉 SUCESSO! As bases tratadas estão prontas (sem ruído de zeros nulos e mantendo os NaNs).")