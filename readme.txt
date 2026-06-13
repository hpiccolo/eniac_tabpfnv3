# Benchmarking Automated Machine Learning and TabPFN v3 for Molecular Toxicity Prediction

This repository contains the experimental framework used to predict molecular toxicity and adverse effects across the **Tox21** and **SIDER** datasets. The project compares traditional tree-based ensemble models (Random Forest and XGBoost) against TabPFN v3.

---

## English

### 📁 Repository Structure and File Functions

* **`baixar_e_preparar_bases.py`**: Data pipeline initialization script. It downloads the raw chemical compound data, handles missing values, computes 512-bit *Morgan Fingerprints* using RDKit, and saves the final structured matrices into compressed Parquet files.
* **`Sider_kfold_tabpfnV3_sem_key.py`**: Executes the multi-label benchmark for the SIDER dataset using a 5-Fold Cross-Validation scheme. It compares **Binary Relevance (BR)** and **Classifier Chains (CC)** strategies using TabPFN's general-purpose third-generation checkpoint (`tabpfn-v3-classifier-v3_default.ckpt`).
* **`tox21_kfold_tabpfnv3_sem_key.py`**: Applies the exact same multi-label cross-validation pipeline (BR vs CC) to the Tox21 dataset, assessing model performance under a clean matrix approach (rows with NaNs removed).
* **`Sider_vias_individuais_kfold_tabpfn3_sem_key.py`**: Implements the organ-system-based approach for the SIDER dataset. It isolates each side-effect pathway individually.
* **`tox_21_vias_individuais_kfold_tabpfn3_sem_key.py`**: Runs the individual biological pathway approach for Tox21. The script scans the 12 cellular toxicity pathways independently, ensuring models learn from the maximum valid data support per bioassay.
* **`Gerador imagens_git.ipynb`**: Auxiliary Jupyter Notebook responsible for loading the benchmark-generated CSV files, aggregating statistical results, and plotting performance graphs (such as PR-AUC curves and F1-Scores).
* **`environment_eniac_tabpfnv3.yml`**: Conda environment configuration file. It contains the exact list of dependencies, deep learning frameworks (CUDA/PyTorch), data handling libraries, and cheminformatics tools required to replicate the project seamlessly.

---

## 🇧🇷 Português

### 📁 Estrutura do Repositório e Função dos Arquivos

* **`baixar_e_preparar_bases.py`**: Script de inicialização do pipeline de dados. Ele realiza o download dos dados brutos dos compostos químicos, trata valores ausentes e computa os *Morgan Fingerprints* de 512 bits via RDKit, salvando as matrizes estruturadas finais em arquivos compactados do tipo Parquet.
* **`Sider_kfold_tabpfnV3_sem_key.py`**: Executa o benchmark multirrótulo para a base SIDER utilizando validação cruzada de 5 dobras (*5-Fold Cross-Validation*). Compara as estratégias de **Binary Relevance (BR)** e **Classifier Chains (CC)** usando o checkpoint generalista da terceira geração do TabPFN (`tabpfn-v3-classifier-v3_default.ckpt`).
* **`tox21_kfold_tabpfnv3_sem_key.py`**: Aplica exatamente o mesmo fluxo de validação cruzada multirrótulo (BR vs CC) para a base do Tox21, avaliando o comportamento dos modelos sob uma abordagem de matriz limpa (linhas com NaNs removidas).
* **`Sider_vias_individuais_kfold_tabpfn3_sem_key.py`**: Implementa a abordagem baseada em sistemas de órgãos para a base SIDER. Isola cada via de efeito colateral individualmente.
* **`tox_21_vias_individuais_kfold_tabpfn3_sem_key.py`**: Executa a abordagem baseada em vias biológicas individuais para o Tox21. O script varre as 12 vias de toxicidade celular isoladamente, garantindo que o modelo aprenda com o suporte máximo de dados válidos por ensaio biológico.
* **`Gerador imagens_git.ipynb`**: Notebook Jupyter auxiliar responsável por carregar os arquivos CSV gerados pelos benchmarks, consolidar os resultados estatísticos e plotar os gráficos de desempenho das métricas (como curvas PR-AUC e F1-Score).
* **`environment_eniac_tabpfnv3.yml`**: Arquivo de configuração de ambiente do Conda. Contém a lista exata de dependências, bibliotecas de processamento gráfico (CUDA/PyTorch), manipulação de dados e ferramentas de quiminformática necessárias para reproduzir o projeto sem conflitos.

