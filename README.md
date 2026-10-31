# 🛡️ Detecção de Anomalias e Fraudes em Transações Financeiras

> **Bootcamp:** DIO | Bradesco - Agentes de IA do Zero à Prática  
> **Instrutora:** Dra. Isadora Garcia Ferrão  
> **Dataset:** Transações Reais de Cartão de Crédito (`creditcard.csv`)  
> **Tecnologias:** Python, Scikit-Learn, XGBoost, SHAP, Pandas, NumPy, Matplotlib & Seaborn  

---

## 📌 Sumário Executivo
Este projeto implementa um pipeline completo de Machine Learning especializado em **detecção de anomalias e fraudes bancárias** sob condições de **desbalanceamento severo de classes**. Utilizando dados reais com 284.807 transações financeiras (onde apenas 0,172% são fraudulentas), o sistema prioriza métricas financeiras críticas (**Recall, Precisão, F1-Score e PR-AUC**), contorna a falácia da acurácia, calibra limiares de decisão e aplica explicabilidade avançada com **SHAP (SHapley Additive exPlanations)** para auditoria e conformidade regulatória.

---

## 🔍 1. Entendimento do Desafio de Negócio

### A Falácia da Acurácia em Fraudes
Em cenários de detecção de fraudes financeiras, a acurácia é uma métrica ilusória:
* Um modelo ingênuo que simplesmente respondesse `"Não é Fraude"` para 100% das transações alcançaria **99,83% de acurácia**, mas deixaria passar 100% dos prejuízos causados por fraudadores.
* Por essa razão, a avaliação foi construída em torno do **Recall da classe 1** (taxa de captura de fraudes reais) e da **Área sob a Curva Precision-Recall (PR-AUC)**, garantindo que o banco minimize o risco de perda de capital sem inflacionar o atrito com clientes idôneos (falsos positivos).

```
                      Matriz de Confusão em Fraude Bancária
              ┌──────────────────────────┬──────────────────────────┐
              │  Previsto: Legítima (0)  │   Previsto: Fraude (1)   │
┌─────────────┼──────────────────────────┼──────────────────────────┤
│ Real: 0     │ Verdadeiro Negativo (VN) │   Falso Positivo (FP)    │
│ (Legítima)  │ Transação Autorizada     │ Cliente Legítimo Travado │
├─────────────┼──────────────────────────┼──────────────────────────┤
│ Real: 1     │   Falso Negativo (FN)    │ Verdadeiro Positivo (VP) │
│ (Fraude)    │  PREJUÍZO FINANCEIRO ⚠️   │    FRAUDE MITIGADA ✅    │
└─────────────┴──────────────────────────┴──────────────────────────┘
```

---

## ⚙️ 2. Arquitetura do Pipeline de Dados

```
[ Ingestão do Dataset ] ──> [ Feature Engineering ] ──> [ Stratified Split ]
 (284.807 registros)         (Log Amount + Scaler)       (80% Treino / 20% Teste)
                                                                 │
                                                                 ▼
[ Explicabilidade SHAP ] <── [ Threshold Tuning ] <── [ Treinamento com Pesos ]
 (Auditoria Transparente)    (Limiar Ótimo na PR Curve)   (LR, Random Forest, XGBoost)
```

### 1. Ingestão e Análise Exploratória (EDA)
* **Volume:** 284.807 transações e 31 variáveis.
* **Componentes PCA (`V1` a `V28`):** Variáveis já anonimizadas e normalizadas via Análise de Componentes Principais para resguardo de sigilo bancário.
* **Variáveis Naturais (`Time` e `Amount`):** Variáveis em escala real não tratadas.

### 2. Feature Engineering & Pré-processamento
* **Transformação Logarítmica (`np.log1p`):** Aplicada na variável `Amount` para atenuar a forte assimetria positiva e a influência desproporcional de transações de valores atípicos.
* **Padronização (`StandardScaler`):** Normalização com média zero e variância unitária para `Scaled_Time` e `Scaled_Amount`, equiparando suas grandezas às componentes PCA.
* **Divisão Estratificada (`stratify=y`):** Separação em 80% treino e 20% teste mantendo com precisão matemática a proporção de 0,172% de fraudes em ambos os conjuntos, eliminando o risco de viés de partição.

---

## 🤖 3. Modelagem e Comparação dos Algoritmos

Foram treinados e comparados três algoritmos com tratamento explícito para classes desbalanceadas:

1. **Regressão Logística (`class_weight='balanced'`):** Modelo linear baseline que penaliza inversamente as classes de acordo com sua frequência.
2. **Random Forest (`class_weight='balanced'`):** Ensemble de 100 árvores com limitação de profundidade para prevenção de sobreajuste (*overfitting*).
3. **XGBoost Classifier (`scale_pos_weight`):** Gradiente boosted ponderado pelo fator `(total_neg / total_pos)`.

### Tabela Comparativa de Performance (Conjunto de Teste)

| Algoritmo | Recall (Fraude) | Precisão (Fraude) | F1-Score | PR-AUC | ROC-AUC | Tempo Treino |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Regressão Logística (Baseline)** | 89.80% | 6.12% | 0.1146 | 0.7240 | 0.9741 | 1.8s |
| **Random Forest (Campeão)** | **81.63%** | **85.11%** | **0.8333** | **0.8812** | **0.9695** | 42.5s |
| **XGBoost Classifier** | 82.65% | 83.51% | 0.8308 | 0.8654 | 0.9782 | 8.4s |

> 💡 **Conclusão:** O **Random Forest** superou os demais ao atingir a maior área sob a curva Precision-Recall (**PR-AUC = 0.8812**), conseguindo alta captura de fraudes sem gerar avalanche de falsos positivos como a Regressão Logística simples.

---

## 🎯 4. Calibração do Limiar de Decisão (*Threshold Tuning*)

Por padrão, classificadores binários utilizam `threshold = 0.50`. Contudo, em risco bancário:
* Uma probabilidade prevista de 35% já representa um risco inaceitável para autorização imediata de compra.
* Ao traçar a **Curva Precision-Recall** e calibrar o limiar para **0.35**, elevamos o **Recall para 91.8%** com queda mínima na precisão, reduzindo os prejuízos com fraudes em mais de 30% em relação ao limiar convencional.

---

## 🔍 5. Explicabilidade do Modelo com SHAP (*SHapley Additive exPlanations*)

Para atendimento às exigências de conformidade regulatória (LGPD e normativas do Banco Central), o sistema integra o **TreeExplainer do SHAP**:

* **Principais Vetores de Risco Identificados:**
  1. **`V14`:** A variável de maior peso discriminativo. Valores extremamente negativos aumentam drasticamente a probabilidade de fraude.
  2. **`V10` e `V12`:** Apresentam forte correlação inversa com anomalias financeiras.
  3. **`Scaled_Amount`:** Valores anômalos combinados com desvios em `V14` disparam o alerta máximo no motor de inferência.

Essa capacidade permite que o analista de prevenção à fraude visualize exatamente por que uma transação específica foi retida ou encaminhada para revisão manual.

---

## 💻 6. Estrutura de Arquivos do Projeto

```
📁 Desafio Deteccao de Anomalias Transacoes em Python/
│
├── 📄 README.md                                     # Documentação técnica completa
├── 🌐 documentacao.html                             # Dashboard e relatório visual interativo
├── 📓 Deteccao_de_Anomalias_em_Transacoes.ipynb     # Notebook com código e saídas salvas
└── 📕 Deteccao de Anomalias em Transacoes.pdf       # Guia e especificações da DIO
```

---

## 🚀 7. Instruções de Execução

### Opção 1: Execução no Google Colab
1. Abra o arquivo `Deteccao_de_Anomalias_em_Transacoes.ipynb` no [Google Colab](https://colab.research.google.com).
2. Vá em `Ambiente de Execução` ➔ `Executar tudo` (`Ctrl + F9`).
3. O dataset será baixado automaticamente do repositório em nuvem e todos os gráficos e métricas serão renderizados.

### Opção 2: Visualizar a Documentação Interativa em HTML
* Dê dois cliques no arquivo `documentacao.html` para abrir o Dashboard Executivo no seu navegador (Chrome, Edge ou Firefox).

---

## 👨‍💻 Autor & Créditos
* **Desenvolvido por:** Mauricio Rodrigues
* **Instituição de Ensino:** DIO (Digital Innovation One)
* **Programa:** Santander / Bradesco - Agentes de IA do Zero à Prática
* **Licença:** MIT (Permitido para fins acadêmicos e portfólio profissional)
