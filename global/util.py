import numpy as np
import pandas as pd
import gc
from tqdm import tqdm
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score, brier_score_loss, precision_recall_curve
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
import seaborn as sns
import shap
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, FloatType, DoubleType, LongType, ShortType
import itertools
from sklearn.metrics import confusion_matrix, accuracy_score,f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve


def reduce_mem_usage(df, verbose=False):
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype
        if (col_type != object) and (str(col_type) != 'category'):
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                int_types = [
                    (np.int8, np.iinfo(np.int8).min, np.iinfo(np.int8).max),
                    (np.uint8, np.iinfo(np.uint8).min, np.iinfo(np.uint8).max),
                    (np.int16, np.iinfo(np.int16).min, np.iinfo(np.int16).max),
                    (np.uint16, np.iinfo(np.uint16).min, np.iinfo(np.uint16).max),
                    (np.int32, np.iinfo(np.int32).min, np.iinfo(np.int32).max),
                    (np.uint32, np.iinfo(np.uint32).min, np.iinfo(np.uint32).max),
                    (np.int64, np.iinfo(np.int64).min, np.iinfo(np.int64).max),
                    (np.uint64, np.iinfo(np.uint64).min, np.iinfo(np.uint64).max)
                ]
                for dtype, min_val, max_val in int_types:
                    if c_min > min_val and c_max < max_val:
                        df[col] = df[col].astype(dtype)
                        if verbose:
                            print(f"Casting column {col} to {str(dtype)}")
                        break
            elif str(col_type)[:5] == 'float':
                float_types = [
                    (np.float16, np.finfo(np.float16).min, np.finfo(np.float16).max),
                    (np.float32, np.finfo(np.float32).min, np.finfo(np.float32).max),
                    (np.float64, np.finfo(np.float64).min, np.finfo(np.float64).max)
                ]
                for dtype, min_val, max_val in float_types:
                    if c_min > min_val and c_max < max_val:
                        df[col] = df[col].astype(dtype)
                        if verbose:
                            print(f"Casting column {col} to {str(dtype)}")
                        break

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

def clf_metric_report(y_score, y_true):
    print('Evaluating the model...')
    roc_auc = roc_auc_score(y_true, y_score)
    brier = brier_score_loss(y_true, y_score)
    avg_precision = average_precision_score(y_true, y_score)
    logloss = log_loss(y_true, y_score)
    print(f'ROC AUC: {roc_auc}')
    print(f'Brier Score: {brier}')
    print(f'Average Precision: {avg_precision}')
    print(f'Log Loss: {logloss}')

def compute_and_plot_permutation_importance(model, X_test, y_test, metric='average_precision', n_repeats=5):
    # Calculate permutation importance
    result = permutation_importance(model, X_test, y_test, n_repeats=n_repeats, random_state=42, scoring=metric)
    features = X_test.columns.to_list()

    # Sort features by importance
    feature_importance = pd.DataFrame({'feature': features, 'importance': result.importances_mean})
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    # Plot top 20 most important features using seaborn
    plt.figure(figsize=(10, 12))
    sns.barplot(data=feature_importance, y='feature', x='importance')
    plt.xlabel('Permutation Importance')
    plt.ylabel('Features')
    plt.title('Top 20 Most Important Features')
    plt.tight_layout()
    plt.show()
    return feature_importance

def plot_calibration_curve(y_score, y_true, title='Calibration Curve'):
    prob_true, prob_pred = calibration_curve(y_score, y_true, n_bins=10)
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='.')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.title(title)
    plt.show()

def plot_pr_calib_curve(y_score, y_true):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')

    plt.subplot(1, 2, 2)
    plt.plot(prob_pred, prob_true, marker='.')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.title('Calibration Curve')

    plt.tight_layout()
    plt.show()

def plot_dis_probs(y_score, y_true):
    plt.figure(figsize=(10, 6))
    sns.histplot(y_score[y_true == 1], bins=50, color='red', label='Churn', kde=True, stat='density')
    sns.histplot(y_score[y_true == 0], bins=50, color='blue', label='Non-Churn', kde=True, stat='density')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predicted Probabilities for Churn vs Non-Churn')
    plt.legend()
    plt.show()

def plot_shap_values(shap_values, X_test, figsize=(10, 12), type='bar'):
    #plt.figure(figsize=figsize)
    shap.summary_plot(shap_values, X_test, plot_type='bar', max_display=25)

import numpy as np



def reduce_mem_usage_spark(df, verbose=False):
    # Iterando sobre as colunas do DataFrame
    for col in df.columns:
        # Pegando o tipo da coluna
        col_type = dict(df.dtypes)[col]
        
        # Calculando os valores mínimo e máximo da coluna
        c_min, c_max = df.select(F.min(col), F.max(col)).first()

        if isinstance(c_min, (int, float)) and isinstance(c_max, (int, float)):
            if 'int' in col_type:
                int_types = [
                    (IntegerType(), -2147483648, 2147483647),  # int32
                    (LongType(), -9223372036854775808, 9223372036854775807),  # int64
                ]
                for dtype, min_val, max_val in int_types:
                    if c_min > min_val and c_max < max_val:
                        df = df.withColumn(col, df[col].cast(dtype))
                        if verbose:
                            print(f"Casting column {col} to {str(dtype)}")
                        break

            elif 'float' in col_type:
                float_types = [
                    (FloatType(), -3.4028235e+38, 3.4028235e+38),  # float32
                    (DoubleType(), -1.7976931348623157e+308, 1.7976931348623157e+308),  # float64
                ]
                for dtype, min_val, max_val in float_types:
                    if c_min > min_val and c_max < max_val:
                        df = df.withColumn(col, df[col].cast(dtype))
                        if verbose:
                            print(f"Casting column {col} to {str(dtype)}")
                        break

    return df

# Função para verificar os metadados da tabela.
def generate_metadata(df, ids=None, targets=None, orderby='PC_NULOS'):
    """
    Esta função retorna uma tabela com informações descritivas sobre um DataFrame.

    Parâmetros:
    - df: DataFrame que você quer descrever.
    - ids: Lista de colunas que são identificadores.
    - targets: Lista de colunas que são variáveis alvo.
    - orderby: Coluna pela qual ordenar os resultados.

    Retorna:
    Um DataFrame com informações sobre o df original.
    """

    if ids is None:
        summary = pd.DataFrame({
            'FEATURE': df.columns,
            'USO_FEATURE': ['Target' if col in targets else 'Explicativa' for col in df.columns],
            'QT_NULOS': df.isnull().sum(),
            'PC_NULOS': round((df.isnull().sum() / len(df)) * 100, 2),
            'CARDINALIDADE': df.nunique(),
            'TIPO_FEATURE': df.dtypes
        })
    else:
        summary = pd.DataFrame({
            'FEATURE': df.columns,
            'USO_FEATURE': ['ID' if col in ids else 'Target' if col in targets else 'Explicativa' for col in df.columns],
            'QT_NULOS': df.isnull().sum(),
            'PC_NULOS': round((df.isnull().sum() / len(df)) * 100, 2),
            'CARDINALIDADE': df.nunique(),
            'TIPO_FEATURE': df.dtypes
        })

    summary_sorted = summary.sort_values(by=orderby, ascending=False)
    summary_sorted = summary_sorted.reset_index(drop=True)

    return summary_sorted



# =================================================================================================================================================== #



def custom_fillna(df):
    '''
    Esta função preenche os valores nulos do DataFrame com a média das colunas numéricas e com 'VERIFICAR' para as colunas categóricas.

    Parâmetros:
    - df: DataFrame a ser preenchido.

    Retorna:
    O DataFrame preenchido, um dicionário contendo as médias das colunas numéricas e outro dicionário com os valores usados para preencher as colunas categóricas.
    '''

    # Preenchimento para colunas numéricas
    numerical_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
    means = {}
    for col in numerical_cols:
        means[col] = df[col].mean()
        df[col].fillna(means[col], inplace=True)

    # Preenchimento para colunas categóricas
    categorical_cols = df.select_dtypes(include=['object']).columns
    modes = {}
    for col in categorical_cols:
        modes[col] = df[col].mode()[0] if not df[col].mode().empty else 'VERIFICAR'
        df[col].fillna(modes[col], inplace=True)

    return df, means, modes


# Função para preenchimento dos valores nulos em produção.
def custom_fillna_prod(df, means):
    '''
    Esta função preenche os valores nulos do DataFrame em produção.

    Parâmetros:
    - df: DataFrame a ser preenchido.
    - means: Dicionário contendo as médias das colunas numéricas.

    Retorna:
    O DataFrame preenchido.
    '''

    for col, mean_value in means.items():
      df[col].fillna(mean_value, inplace=True)

    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna('VERIFICAR')

    return df

# Função para o cálculo do WoE e IV.
def calculate_woe_iv(df, feature, target):
    '''
    Calcula o WoE (Weight of Evidence) e IV (Information Value) para uma variável.

    Parâmetros:
    - df: DataFrame contendo a variável.
    - feature: Nome da variável.
    - target: Nome da variável alvo.

    Retorna:
    - IV da variável.
    '''

    lst = []
    for i in range(df[feature].nunique()):
        val = list(df[feature].unique())[i]
        lst.append({
            'Value': val,
            'All': df[df[feature] == val].count()[feature],
            'Good': df[(df[feature] == val) & (df[target] == 1)].count()[feature],
            'Bad': df[(df[feature] == val) & (df[target] == 0)].count()[feature]
        })

    dset = pd.DataFrame(lst)
    dset['Distr_Good'] = dset['Good'] / dset['Good'].sum()
    dset['Distr_Bad'] = dset['Bad'] / dset['Bad'].sum()

    # Adicionando uma pequena constante para evitar divisão por zero.
    dset['Distr_Good'] = dset['Distr_Good'].replace({0: 1e-10})
    dset['Distr_Bad'] = dset['Distr_Bad'].replace({0: 1e-10})

    dset['WoE'] = np.log(dset['Distr_Good'] / dset['Distr_Bad'])
    dset = dset.replace({'WoE': {np.inf: 0, -np.inf: 0}})
    dset['IV'] = (dset['Distr_Good'] - dset['Distr_Bad']) * dset['WoE']
    iv = dset['IV'].sum()

    return iv


# Função para a criação de uma tabela com o IV das variáveis.
def iv_table(df, target):
    '''
    Retorna uma tabela com o IV para todas as variáveis em relação ao target.

    Parâmetros:
    - df: DataFrame contendo as variáveis.
    - target: Nome da variável alvo.

    Retorna:
    - DataFrame com o IV das variáveis.
    '''

    iv_list = []
    for col in df.columns:
        if col == target:
            continue
        iv = calculate_woe_iv(df, col, target)
        if iv < 0.02:
            predictiveness = 'Inútil para a predição'
        elif iv < 0.1:
            predictiveness = 'Preditor Fraco'
        elif iv < 0.3:
            predictiveness = 'Preditor Moderado'
        else:
            predictiveness = 'Preditor Forte'
        iv_list.append({
            'Variável': col,
            'IV': iv,
            'Preditividade': predictiveness
        })

    return pd.DataFrame(iv_list).sort_values(by='IV', ascending=False)




# =================================================================================================================================================== #



# Função para calcular o KS.
def calcular_ks_statistic(y_true, y_score):
    '''
    Calcula o KS (Kolmogorov-Smirnov) para avaliação de um modelo de classificação.

    Parâmetros:
    - y_true: Valores verdadeiros.
    - y_score: Escores previstos.

    Retorna:
    - Valor do KS.
    '''

    df = pd.DataFrame({'score': y_score, 'target': y_true})
    df = df.sort_values(by='score', ascending=False)
    total_events = df.target.sum()
    total_non_events = len(df) - total_events
    df['cum_events'] = df.target.cumsum()
    df['cum_non_events'] = (df.target == 0).cumsum()
    df['cum_events_percent'] = df.cum_events / total_events
    df['cum_non_events_percent'] = df.cum_non_events / total_non_events
    ks_statistic = np.abs(df.cum_events_percent - df.cum_non_events_percent).max()
    return ks_statistic


# Função para calcular o KS.
def gini_normalizado(y_true, y_pred):
    import pandas as pd
    import numpy as np
    # check and get number of samples
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]

    # sort rows on prediction column
    # (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:, 0].argsort()][::-1, 0]
    pred_order = arr[arr[:, 1].argsort()][::-1, 0]

    # get Lorenz curves
    L_true = np.cumsum(true_order) * 1. / np.sum(true_order)
    L_pred = np.cumsum(pred_order) * 1. / np.sum(pred_order)
    L_ones = np.linspace(1 / n_samples, 1, n_samples)

    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)

    # normalize to true Gini coefficient
    return G_pred * 1. / G_true


# Função para calcular as métricas e plotar.
def avaliar_modelo(X_train, y_train, X_test, y_test, modelo, nm_modelo):
    '''
    Avalia um modelo de classificação e plota várias métricas de desempenho.

    Parâmetros:
    - X_train: Features do conjunto de treino.
    - y_train: Variável alvo do conjunto de treino.
    - X_test: Features do conjunto de teste.
    - y_test: Variável alvo do conjunto de teste.
    - modelo: Modelo treinado.
    - nm_modelo: Nome do modelo.

    Retorna:
    Uma série de gráficos com as principais métricas de desempenho para treino e teste.
    '''

    feature_names = list(X_train.columns)
    # Criação da figura e dos eixos.
    fig, axs = plt.subplots(5, 2, figsize=(15, 30))     # Ajustado para incluir novos gráficos.
    plt.tight_layout(pad=6.0)

    # Cor azul claro.
    cor = 'skyblue'

    # Taxa de Evento e Não Evento.
    event_rate = np.mean(y_train)
    non_event_rate = 1 - event_rate
    axs[0, 0].bar(['Evento', 'Não Evento'], [event_rate, non_event_rate], color=[cor, 'lightcoral'])
    axs[0, 0].set_title('Taxa de Evento e Não Evento')
    axs[0, 0].set_ylabel('Proporção')

    # Importância dos Atributos.
    importancias = None
    if hasattr(modelo, 'coef_'):      # hasattr = Tem atributo? Se tem coeficiênte ou não, se não tiver ele calcula a feature importance, sem tem coeficiênte, tem beta, ele calcula o peso do beta.
        importancias = np.abs(modelo.coef_[0])
    elif hasattr(modelo, 'feature_importances_'):
        importancias = modelo.feature_importances_

    if importancias is not None:
        importancias_df = pd.DataFrame({'feature': feature_names, 'importance': importancias})
        importancias_df = importancias_df.sort_values(by='importance', ascending=True)

        axs[0, 1].barh(importancias_df['feature'], importancias_df['importance'], color=cor)
        axs[0, 1].set_title('Importância das Variáveis - ' + nm_modelo)
        axs[0, 1].set_xlabel('Importância')

    else:
        axs[0, 1].axis('off')     # Desativa o subplot se não houver importâncias para mostrar.

    # Confusion Matrix - Treino.
    y_pred_train = modelo.predict(X_train)
    cm_train = confusion_matrix(y_train, y_pred_train)
    axs[1, 0].imshow(cm_train, interpolation='nearest', cmap=plt.cm.Blues)
    axs[1, 0].set_title('Confusion Matrix - Treino - ' + nm_modelo)
    axs[1, 0].set_xticks([0, 1])
    axs[1, 0].set_yticks([0, 1])
    axs[1, 0].set_xticklabels(['0', '1'])
    axs[1, 0].set_yticklabels(['0', '1'])
    thresh = cm_train.max() / 2.
    for i, j in itertools.product(range(cm_train.shape[0]), range(cm_train.shape[1])):
        axs[1, 0].text(j, i, format(cm_train[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm_train[i, j] > thresh else "black")

    # Confusion Matrix - Teste.
    y_pred_test = modelo.predict(X_test)
    cm_test = confusion_matrix(y_test, y_pred_test)
    axs[1, 1].imshow(cm_test, interpolation='nearest', cmap=plt.cm.Blues)
    axs[1, 1].set_title('Confusion Matrix - Teste - ' + nm_modelo)
    axs[1, 1].set_xticks([0, 1])
    axs[1, 1].set_yticks([0, 1])
    axs[1, 1].set_xticklabels(['0', '1'])
    axs[1, 1].set_yticklabels(['0', '1'])
    thresh = cm_test.max() / 2.
    for i, j in itertools.product(range(cm_test.shape[0]), range(cm_test.shape[1])):
        axs[1, 1].text(j, i, format(cm_test[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm_test[i, j] > thresh else "black")

    # ROC Curve - Treino e Teste.
    y_score_train = modelo.predict_proba(X_train)[:, 1]
    fpr_train, tpr_train, _ = roc_curve(y_train, y_score_train)
    axs[2, 0].plot(fpr_train, tpr_train, color=cor, label='Treino')

    y_score_test = modelo.predict_proba(X_test)[:, 1]
    fpr_test, tpr_test, _ = roc_curve(y_test, y_score_test)
    axs[2, 0].plot(fpr_test, tpr_test, color='darkorange', label='Teste')

    axs[2, 0].plot([0, 1], [0, 1], color='navy', linestyle='--')
    axs[2, 0].set_title('ROC Curve - Treino e Teste - ' + nm_modelo)
    axs[2, 0].set_xlabel('False Positive Rate')
    axs[2, 0].set_ylabel('True Positive Rate')
    axs[2, 0].legend(loc="lower right")

    # Precision-Recall Curve - Treino e Teste.
    precision_train, recall_train, _ = precision_recall_curve(y_train, y_score_train)
    axs[2, 1].plot(recall_train, precision_train, color=cor, label='Treino')

    precision_test, recall_test, _ = precision_recall_curve(y_test, y_score_test)
    axs[2, 1].plot(recall_test, precision_test, color='darkorange', label='Teste')

    axs[2, 1].set_title('Precision-Recall Curve - Treino e Teste - ' + nm_modelo)
    axs[2, 1].set_xlabel('Recall')
    axs[2, 1].set_ylabel('Precision')
    axs[2, 1].legend(loc="upper right")

    # Gini - Treino e Teste.
    auc_train = roc_auc_score(y_train, y_score_train)
    gini_train = 2 * auc_train - 1
    auc_test = roc_auc_score(y_test, y_score_test)
    gini_test = 2 * auc_test - 1
    axs[3, 0].bar(['Treino', 'Teste'], [gini_train, gini_test], color=[cor, 'darkorange'])
    axs[3, 0].set_title('Gini - ' + nm_modelo)
    axs[3, 0].set_ylim(0, 1)
    axs[3, 0].text('Treino', gini_train + 0.01, f'{gini_train:.2f}', ha='center', va='bottom')
    axs[3, 0].text('Teste', gini_test + 0.01, f'{gini_test:.2f}', ha='center', va='bottom')

    # KS - Treino e Teste.
    ks_train = calcular_ks_statistic(y_train, y_score_train)
    ks_test = calcular_ks_statistic(y_test, y_score_test)
    axs[3, 1].bar(['Treino', 'Teste'], [ks_train, ks_test], color=[cor, 'darkorange'])
    axs[3, 1].set_title('KS - ' + nm_modelo)
    axs[3, 1].set_ylim(0, 1)
    axs[3, 1].text('Treino', ks_train + 0.01, f'{ks_train:.2f}', ha='center', va='bottom')
    axs[3, 1].text('Teste', ks_test + 0.01, f'{ks_test:.2f}', ha='center', va='bottom')

    # Decile Analysis - Teste.
    scores = modelo.predict_proba(X_test)[:, 1]
    noise = np.random.uniform(0, 0.0001, size=scores.shape)     # Adiciona um pequeno ruído.
    scores += noise
    deciles = pd.qcut(scores, q=10, duplicates='drop')
    decile_analysis = y_test.groupby(deciles).mean()
    axs[4, 1].bar(range(1, len(decile_analysis) + 1), decile_analysis, color='darkorange')
    axs[4, 1].set_title('Ordenação do Score - Teste - ' + nm_modelo)
    axs[4, 1].set_xlabel('Faixas de Score')
    axs[4, 1].set_ylabel('Taxa de Evento')

    # Decile Analysis - Treino.
    scores_train = modelo.predict_proba(X_train)[:, 1]
    noise = np.random.uniform(0, 0.0001, size=scores_train.shape)     # Adiciona um pequeno ruído.
    scores_train += noise
    deciles_train = pd.qcut(scores_train, q=10, duplicates='drop')
    decile_analysis_train = y_train.groupby(deciles_train).mean()
    axs[4, 0].bar(range(1, len(decile_analysis_train) + 1), decile_analysis_train, color=cor)
    axs[4, 0].set_title('Ordenação do Score - Treino - ' + nm_modelo)
    axs[4, 0].set_xlabel('Faixas de Score')
    axs[4, 0].set_ylabel('Taxa de Evento')

    # Mostrar os gráficos.
    plt.show()



# =================================================================================================================================================== #



# Função para criar um DataFrame com as métricas de todos os modelos treinados.
def evaluate_models(X_train, y_train, X_test, y_test, models):
    '''
    Avalia múltiplos modelos de classificação e retorna um DataFrame com as métricas de desempenho de cada modelo, destacando as métricas mais altas.

    Parâmetros:
    - X_train: Features do conjunto de treino.
    - y_train: Variável alvo do conjunto de treino.
    - X_test: Features do conjunto de teste.
    - y_test: Variável alvo do conjunto de teste.
    - models: Dicionário contendo os modelos treinados.

    Retorna:
    DataFrame contendo as métricas de desempenho de todos os modelos.
    '''

    metrics = []
    for name, model in models.items():
        # Iniciando o cronômetro.
        import time
        start_time = time.time()

        # Prever os rótulos para os conjuntos de treino e teste.
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        # Calcular as métricas.
        accuracy = accuracy_score(y_test, test_preds)
        precision = precision_score(y_test, test_preds)
        recall = recall_score(y_test, test_preds)
        f1 = f1_score(y_test, test_preds)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])     # Supondo que é um problema de classificação binária.
        gini = 2*auc - 1
        ks = calcular_ks_statistic(y_test, model.predict_proba(X_test)[:, 1])

        # Calculando o tempo de treinamento.
        end_time = time.time()
        training_time = end_time - start_time

        # Adicionar ao array de métricas.
        metrics.append({
            'Model': name,
            'AUC-ROC': auc,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Gini': gini,
            'KS': ks,
            'Training_Time(s)': training_time
        })

    # Convertendo o array de métricas em um DataFrame.
    metrics_df = pd.DataFrame(metrics)

    # Ordenando o DataFrame pela metrica AUC-ROC.
    metrics_df_sorted = metrics_df.sort_values(by='AUC-ROC', ascending=False)

    # Função para destacar o maior valor em azul claro.
    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: lightblue' if v else '' for v in is_max]

    # Destacando o maior valor de cada métrica
    metrics_df_max = metrics_df_sorted.style.apply(highlight_max, subset=metrics_df.columns[1:-1])

    return metrics_df_max

def plot_tx_event_volume_safra(dataframe, target, safra, ymax_volume=None, ymax_taxa_evento=None):
  import matplotlib.pyplot as plt
  import pandas as pd
  import numpy as np

  # Converte a coluna 'var' para string se seu tipo não for string
  if dataframe[safra].dtype != 'O':  
      dataframe[safra] = dataframe[safra].astype(str)

  # Calcule a média do TARGET_60_6 e o volume por AAAAMM
  resultado = dataframe.groupby(safra).agg({target: 'mean', safra: 'count'}).rename(columns={safra: 'Volume'}).reset_index()
  resultado.columns = ['Safra', 'Taxa_de_Evento', 'Volume']

  # Gráfico com barras para o volume e linha para a taxa de evento por safra (AAAAMM)
  fig, ax1 = plt.subplots(figsize=(12, 6))

  color = 'lightblue'
  ax1.bar(resultado['Safra'], resultado['Volume'], color=color, label='Volume')
  ax1.set_xlabel('Safra')
  ax1.set_ylabel('Volume', color='black')
  ax1.tick_params(axis='y', labelcolor='black')
  if ymax_volume:
      ax1.set_ylim(0, ymax_volume)

  ax2 = ax1.twinx()
  color = 'hotpink'
  ax2.plot(resultado['Safra'], resultado['Taxa_de_Evento'] * 100, marker='o', linestyle='-', color=color, label='Taxa de Evento (%)')
  ax2.set_ylabel('Taxa de Evento (%)', color='black')
  ax2.tick_params(axis='y', labelcolor='black')
  if ymax_taxa_evento:
      ax2.set_ylim(0, ymax_taxa_evento)

  for label in ax1.get_xticklabels() + ax1.get_yticklabels() + ax2.get_yticklabels():
      label.set_fontsize(7)
      label.set_color('black')

  plt.title('Volume e Taxa de Evento por Safra')
  plt.legend(loc='upper left')
  plt.show()

  return resultado

def analyze_variable(dataframe, variable, target):
  import pandas as pd
  import matplotlib.pyplot as plt

  # Se a variável for numérica, arredonda para 4 casas decimais
  if pd.api.types.is_numeric_dtype(dataframe[variable]):
      dataframe[variable] = dataframe[variable].round(4)
      dataframe[variable] = dataframe[variable].astype(str)

  # Calcula a taxa de evento e o volume para cada categoria da variável
  result = dataframe.groupby(variable).agg({target: 'mean', variable: 'count'}).rename(columns={variable: 'Volume'}).reset_index()
  result.columns = [variable, 'Taxa_de_Evento', 'Volume']

  # Ordena o resultado pela Taxa de Evento em ordem decrescente
  result = result.sort_values(by='Taxa_de_Evento', ascending=False)

  # Plota o gráfico
  fig, ax1 = plt.subplots(figsize=(12, 6))

  # Eixo Y esquerdo: Volume
  bars = ax1.bar(result[variable], result['Volume'], color='lightblue', label='Volume (Barras)')
  ax1.set_xlabel(variable)
  ax1.set_ylabel('Volume', color='black')
  ax1.tick_params(axis='y', labelcolor='black')

  # Eixo Y direito: Taxa de Evento
  ax2 = ax1.twinx()
  lines = ax2.plot(result[variable], result['Taxa_de_Evento'] * 100, marker='o', linestyle='-', color='hotpink', label='Taxa de Evento (Linha)')
  ax2.set_ylabel('Taxa de Evento (%)', color='black')
  ax2.tick_params(axis='y', labelcolor='black')

  # Combina as legendas de ambos os eixos, filtrando rótulos que começam com '_'
  plots = [item for item in bars + tuple(lines) if not item.get_label().startswith('_')]
  labels = [plot.get_label() for plot in plots]
  plt.legend(plots, labels, loc='upper left')

  plt.title(f'Volume e Taxa de Evento por {variable}')
  plt.xticks(rotation=45)  # Adicionado para melhor visualização dos labels no eixo X
  plt.tight_layout()
  plt.show()

  return result


def plot_by_safra(dataframe, target, explicativa, safra):

  import pandas as pd
  import matplotlib.pyplot as plt

  df_copy = dataframe.copy()
  # Se a variável explicativa for numérica, arredonda para 4 casas decimais e converte para string
  if pd.api.types.is_numeric_dtype(df_copy[explicativa]):
      df_copy[explicativa] = df_copy[explicativa].apply(lambda x: round(x, 4)).astype(str)

  # Calcula a taxa de evento e o volume por safra e categoria da variável explicativa
  result = df_copy.groupby([safra, explicativa]).agg({target: 'mean', explicativa: 'count'}).rename(columns={explicativa: 'Volume'}).reset_index()

  # Plota o gráfico
  fig, ax1 = plt.subplots(figsize=(15, 10))

  # Eixo Y esquerdo: Volume total por safra
  volume_by_safra = result.groupby(safra).agg({'Volume': 'sum'}).reset_index()
  bars = ax1.bar(volume_by_safra[safra], volume_by_safra['Volume'], color='lightblue', label='Volume Total (Barras)')
  ax1.set_xlabel('Safra')
  ax1.set_ylabel('Volume', color='black')
  ax1.tick_params(axis='y', labelcolor='black')

  # Eixo Y direito: Taxa de Evento por categoria
  ax2 = ax1.twinx()
  for category in result[explicativa].unique():
      subset = result[result[explicativa] == category]
      ax2.plot(subset[safra], subset[target] * 100, marker='o', linestyle='-', label=f'Taxa de Evento ({category})')
  ax2.set_ylabel('Taxa de Evento (%)', color='black')
  ax2.tick_params(axis='y', labelcolor='black')

  plt.title(f'Volume Total e Taxa de Evento por {explicativa} ao longo das Safras')
  plt.legend(loc='upper left')
  plt.xticks(rotation=45)
  plt.tight_layout()
  plt.show()



def group_and_plot_by_safra(dataframe, target, explicativa, safra, domain_mapping):

  import pandas as pd
  import matplotlib.pyplot as plt

  df_copy = dataframe.copy()
  # Se a variável explicativa for numérica, arredonda para 4 casas decimais e converte para string
  if pd.api.types.is_numeric_dtype(df_copy[explicativa]):
      df_copy[explicativa] = df_copy[explicativa].apply(lambda x: round(x, 4)).astype(str)

  # Cria uma coluna com os valores originais para mostrar a transformação posteriormente
  df_copy['original_' + explicativa] = df_copy[explicativa]

  # Aplica o mapeamento para os novos domínios
  df_copy[explicativa] = df_copy[explicativa].map(domain_mapping).fillna(df_copy[explicativa])

  # Calcula a taxa de evento e o volume por safra e categoria da variável explicativa
  result = df_copy.groupby([safra, explicativa]).agg({target: 'mean', explicativa: 'count'}).rename(columns={explicativa: 'Volume'}).reset_index()

  # Plota o gráfico
  fig, ax1 = plt.subplots(figsize=(12, 6))

  # Eixo Y esquerdo: Volume total por safra
  volume_by_safra = result.groupby(safra).agg({'Volume': 'sum'}).reset_index()
  bars = ax1.bar(volume_by_safra[safra], volume_by_safra['Volume'], color='lightblue', label='Volume Total (Barras)')
  ax1.set_xlabel('Safra')
  ax1.set_ylabel('Volume', color='black')
  ax1.tick_params(axis='y', labelcolor='black')

  # Eixo Y direito: Taxa de Evento por categoria
  ax2 = ax1.twinx()
  for category in result[explicativa].unique():
      subset = result[result[explicativa] == category]
      ax2.plot(subset[safra], subset[target] * 100, marker='o', linestyle='-', label=f'Taxa de Evento ({category})')
  ax2.set_ylabel('Taxa de Evento (%)', color='black')
  ax2.tick_params(axis='y', labelcolor='black')

  plt.title(f'Volume Total e Taxa de Evento por {explicativa} ao longo das Safras')
  plt.legend(loc='upper left')
  plt.xticks(rotation=45)
  plt.tight_layout()
  plt.show()

  # Cria um DataFrame de transformação
  transformation_df = df_copy[[explicativa, 'original_' + explicativa]].drop_duplicates().sort_values(by='original_' + explicativa)
  transformation_df.rename(columns={explicativa:'TFB_'+explicativa,'original_' + explicativa:explicativa},inplace=True)

  return transformation_df


def apply_grouping(data, transformation_df, explicativa):

  import pandas as pd
  df_copy = data.copy()

  if pd.api.types.is_numeric_dtype(df_copy[explicativa]):
    df_copy[explicativa] = df_copy[explicativa].apply(lambda x: round(x, 4)).astype(str)

  # Une o DataFrame de transformação com os novos dados para aplicar a transformação
  df_copy = df_copy.merge(transformation_df, left_on=explicativa, right_on=explicativa, how='left')

  # Aplica a transformação
  colname_transformed = 'TFB_' + explicativa
  df_copy[explicativa] = df_copy[colname_transformed].fillna(df_copy[explicativa])

  # Remove a coluna original
  df_copy.drop(columns=[explicativa], inplace=True)

  return df_copy

def categorize_with_decision_tree(dataframe, n_categories, target, numeric_var):
  import numpy as np
  import pandas as pd
  from sklearn.tree import DecisionTreeClassifier
  # Preparar os dados
  X = dataframe[[numeric_var]]
  y = dataframe[target]

  # Treinar uma árvore de decisão com profundidade máxima igual ao número de categorias desejadas
  tree = DecisionTreeClassifier(max_leaf_nodes=n_categories)
  tree.fit(X, y)

  # Predizer a categoria (folha) para cada entrada no DataFrame
  leaf_ids = tree.apply(X)

  # Criar um DataFrame temporário com as categorias (folhas), a variável numérica e o target
  temp_df = pd.DataFrame({numeric_var: dataframe[numeric_var], 'Leaf': leaf_ids, target: y})

  result = temp_df.groupby('Leaf').agg({target: 'mean', numeric_var: ['count', 'min', 'max']}).reset_index()
  result.columns = ['Leaf', 'Taxa_de_Evento', 'Volume', 'Lower_Bound', 'Upper_Bound']

  # Ajuste para garantir que os limites superior e inferior de bins adjacentes não se sobreponham
  result = result.sort_values(by='Lower_Bound')
  for i in range(1, len(result)):
      result.iloc[i, 3] = max(result.iloc[i, 3], result.iloc[i-1, 4])

  # Definir o limite inferior do primeiro bin como -inf e o limite superior do último bin como inf
  result.iloc[0, 3] = -np.inf
  result.iloc[-1, 4] = np.inf

  return result


def apply_tree_bins(data, transformation_df, numeric_var):

  import numpy as np
  df_copy = data.copy()

  # Obtenha os limites superiores e ordene-os
  upper_bounds = transformation_df['Upper_Bound'].sort_values().values

  # Use numpy.digitize para determinar a qual bin cada valor pertence
  df_copy[f"TFT_{numeric_var}"] = np.digitize(df_copy[numeric_var].values, upper_bounds)
  df_copy.drop(axis=1,columns=[numeric_var],inplace=True)

  return df_copy



def logistic_regression_with_scorecard(data, target_var, features):
  import pandas as pd
  import statsmodels.api as sm
  # Adicionando uma constante ao dataset (intercepto)
  data = sm.add_constant(data)

  # Ajustando o modelo de regressão logística
  model = sm.Logit(data[target_var], data[features + ['const']]).fit()
  # model = sm.Logit(data[target_var], data[features]).fit()

  # Coletando p-valores e estatísticas de Wald
  summary = model.summary2().tables[1]
  summary['Wald'] = summary['z']**2
  scorecard = summary[['Coef.', 'P>|z|', 'Wald']]
  scorecard.columns = ['Beta Coefficient', 'P-Value', 'Wald Statistic']
  scorecard = scorecard.sort_values(by='Wald Statistic', ascending=False)

  return model, scorecard

def calculate_metrics(train_df, test_df, score_column, target_column,bins=10):
  import matplotlib.pyplot as plt
  import pandas as pd
  from sklearn.metrics import roc_auc_score, roc_curve, auc
  import numpy as np
  def compute_metrics(df, score_column, target_column):
    import pandas as pd
    from sklearn.metrics import roc_auc_score, roc_curve, auc
    import numpy as np
    df_sorted = df.sort_values(by=score_column, ascending=False)

    # Calcular KS
    df_sorted['cum_good'] = (1 - df_sorted[target_column]).cumsum() / (1 - df_sorted[target_column]).sum()
    df_sorted['cum_bad'] = df_sorted[target_column].cumsum() / df_sorted[target_column].sum()
    df_sorted['ks'] = np.abs(df_sorted['cum_good'] - df_sorted['cum_bad'])
    ks_statistic = df_sorted['ks'].max()

    # Calcular AUC
    auc_value = roc_auc_score(df_sorted[target_column], df_sorted[score_column])

    # Calcular Gini
    gini = 2 * auc_value - 1

    # Dividir o score em 10 faixas
    df_sorted['decile'] = pd.cut(df_sorted[score_column], bins, labels=False)

    # Criar tabela detalhada
    table = df_sorted.groupby('decile').agg(
        min_score=pd.NamedAgg(column=score_column, aggfunc='min'),
        max_score=pd.NamedAgg(column=score_column, aggfunc='max'),
        event_rate=pd.NamedAgg(column=target_column, aggfunc='mean'),
        volume=pd.NamedAgg(column=target_column, aggfunc='size')
    ).reset_index()

    return ks_statistic, auc_value, gini, table

  ks_train, auc_train, gini_train, table_train = compute_metrics(train_df, score_column, target_column)
  ks_test, auc_test, gini_test, table_test = compute_metrics(test_df, score_column, target_column)

  # Plotando o gráfico de barras para Event Rate por Decil
  barWidth = 0.3
  r1 = np.arange(len(table_train))
  r2 = [x + barWidth for x in r1]

  plt.bar(r1, table_train['event_rate'], color='lightblue', width=barWidth, label='Train')
  plt.bar(r2, table_test['event_rate'], color='royalblue', width=barWidth, label='Test')

  plt.xlabel('Decile')
  plt.ylabel('Event Rate')
  plt.title('Event Rate by Decile')
  plt.xticks([r + barWidth for r in range(len(table_train))], table_train['decile'])
  plt.legend()
  plt.show()

  # Criando DataFrame para as métricas
  metrics_df = pd.DataFrame({
      'Metric': ['KS', 'AUC', 'Gini'],
      'Train Value': [ks_train, auc_train, gini_train],
      'Test Value': [ks_test, auc_test, gini_test]
  })

  return metrics_df, table_train, table_test


def calculate_ks(y_true, y_score):
  from sklearn.metrics import roc_auc_score, roc_curve
  """Calculate KS statistic."""
  fpr, tpr, _ = roc_curve(y_true, y_score)
  return 100*max(tpr - fpr)

def calculate_gini(y_true, y_score):
  from sklearn.metrics import roc_auc_score, roc_curve
  """Calculate Gini coefficient."""
  return (2 * roc_auc_score(y_true, y_score) - 1)*100


def plot_ks_gini_by_datref(df, target_col, score_col, datref_col,titulo='KS e GIni por Safra'):
  import pandas as pd
  import matplotlib.pyplot as plt
  df[datref_col] = df[datref_col].astype(str)
  unique_dates = sorted(df[datref_col].unique())
  ks_values = []
  gini_values = []

  for date in unique_dates:
      subset = df[df[datref_col] == date]
      y_true = subset[target_col]
      y_score = subset[score_col]

      ks_values.append(calculate_ks(y_true, y_score))
      gini_values.append(calculate_gini(y_true, y_score))

  plt.figure(figsize=(12, 6))
  plt.plot(unique_dates, ks_values, label='KS', marker='o')
  plt.plot(unique_dates, gini_values, label='Gini', marker='o')
  plt.xlabel(datref_col)
  plt.ylabel('Value')
  plt.ylim(0, 100)
  plt.title(titulo)
  plt.legend()
  plt.grid(True)
  plt.tight_layout()
  plt.show()

def apply_fillna(df):
    '''
    Esta função preenche os valores nulos do DataFrame com a média das colunas numéricas 
    e com a moda para as colunas categóricas.

    Parâmetros:
    - df: DataFrame a ser preenchido.

    Retorna:
    O DataFrame preenchido e dois dicionários:
    - means: contendo as médias das colunas numéricas.
    - modes: contendo as modas das colunas categóricas.
    '''

    # Preenchimento para colunas numéricas
    numerical_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
    means = {}
    for col in numerical_cols:
        means[col] = df[col].mean()
        df[col].fillna(means[col])

    # Preenchimento para colunas categóricas
    categorical_cols = df.select_dtypes(include=['object']).columns
    modes = {}
    for col in categorical_cols:
        modes[col] = df[col].mode()[0] if not df[col].mode().empty else 'VERIFICAR'
        df[col].fillna(modes[col])

    return df, means, modes


def apply_fillna_prod(df, means, modes):
    '''
    Esta função preenche os valores nulos do DataFrame em produção.

    Parâmetros:
    - df: DataFrame a ser preenchido.
    - means: Dicionário contendo as médias das colunas numéricas.
    - modes: Dicionário contendo as modas das colunas categóricas.

    Retorna:
    O DataFrame preenchido.
    '''

    # Preenchimento para colunas numéricas
    for col, mean_value in means.items():
        df[col].fillna(mean_value, inplace=True)

    # Preenchimento para colunas categóricas
    for col, mode_value in modes.items():
        if col in df.columns:
            df[col].fillna(mode_value, inplace=True)

    return df

def iv_category(df, target, variable, count):

    pivot_table = pd.pivot_table(df, columns=target, index=variable, values=count, aggfunc="count")
    pivot_table['%Sim'] = pivot_table['Sim'].apply(lambda x: round(x/pivot_table['Sim'].sum()*100,2))
    pivot_table['%Nao'] = pivot_table['Não'].apply(lambda x: round(x/pivot_table['Não'].sum()*100,2))
    pivot_table['%FreqLinha'] = (pivot_table['Sim'] / (pivot_table['Não'] + pivot_table['Sim']) * 100).round(2)
    pivot_table['WoE-ODDS'] = (pivot_table['%Sim'] / pivot_table['%Nao'])
    pivot_table['IV'] = ((pivot_table['%Sim']/100) - (pivot_table['%Nao']/100)) * np.log(pivot_table['WoE']).replace([np.inf, -np.inf], 0)
    
    return pivot_table


def iv_bin(df, target, variable, bins, count):

    bin_variable = pd.cut(x=df[variable], bins=bins)
    
    pivot_table = pd.pivot_table(df, columns=target, index=bin_variable, values=count, aggfunc="count")
    pivot_table['%Sim'] = pivot_table['Sim'].apply(lambda x: round(x/pivot_table['Sim'].sum()*100,2))
    pivot_table['%Nao'] = pivot_table['Não'].apply(lambda x: round(x/pivot_table['Não'].sum()*100,2))
    pivot_table['%FreqLinha'] = (pivot_table['Sim'] / (pivot_table['Não'] + pivot_table['Sim']) * 100).round(2)
    pivot_table['WoE'] = (pivot_table['%Sim'] / pivot_table['%Nao'])
    pivot_table['IV'] = ((pivot_table['%Sim']/100) - (pivot_table['%Nao']/100)) * np.log(pivot_table['WoE']).replace([np.inf, -np.inf], 'Não')

    return pivot_table

def metrics_calculate(nm_modelo, model, X_train, y_train, X_test, y_test):
    # Fazendo predições
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculando as métricas para o conjunto de treino
    accuracy_train = accuracy_score(y_train, y_train_pred)
    precision_train = precision_score(y_train, y_train_pred)
    recall_train = recall_score(y_train, y_train_pred)
    auc_roc_train = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])

    # Calculando o Índice Gini e Estatística KS para o conjunto de treino
    probabilities_train = model.predict_proba(X_train)[:, 1]
    df_train = pd.DataFrame({'true_labels': y_train, 'predicted_probs': probabilities_train})
    df_train = df_train.sort_values(by='predicted_probs', ascending=False)
    df_train['cumulative_true'] = df_train['true_labels'].cumsum() / df_train['true_labels'].sum()
    df_train['cumulative_false'] = (1 - df_train['true_labels']).cumsum() / (1 - df_train['true_labels']).sum()
    ks_statistic_train = max(abs(df_train['cumulative_true'] - df_train['cumulative_false']))
    gini_index_train = 2 * auc_roc_train - 1

    # Calculando as métricas para o conjunto de teste
    accuracy_test = accuracy_score(y_test, y_test_pred)
    precision_test = precision_score(y_test, y_test_pred)
    recall_test = recall_score(y_test, y_test_pred)
    auc_roc_test = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    # Calculando o Índice Gini e Estatística KS para o conjunto de teste
    probabilities_test = model.predict_proba(X_test)[:, 1]
    df_test = pd.DataFrame({'true_labels': y_test, 'predicted_probs': probabilities_test})
    df_test = df_test.sort_values(by='predicted_probs', ascending=False)
    df_test['cumulative_true'] = df_test['true_labels'].cumsum() / df_test['true_labels'].sum()
    df_test['cumulative_false'] = (1 - df_test['true_labels']).cumsum() / (1 - df_test['true_labels']).sum()
    ks_statistic_test = max(abs(df_test['cumulative_true'] - df_test['cumulative_false']))
    gini_index_test = 2 * auc_roc_test - 1

    # Criando o DataFrame com as métricas calculadas
    metrics_df = pd.DataFrame({
        'Algoritmo': [nm_modelo, nm_modelo],
        'Conjunto': ['Treino', 'Teste'],
        'Acuracia': [accuracy_train, accuracy_test],
        'Precisao': [precision_train, precision_test],
        'Recall': [recall_train, recall_test],
        'AUC_ROC': [auc_roc_train, auc_roc_test],
        'GINI': [gini_index_train, gini_index_test],
        'KS': [ks_statistic_train, ks_statistic_test]
    })
    return metrics_df

def fillna_numeric(df):
    '''
    Preenche os valores nulos das colunas numéricas do DataFrame com a média de cada coluna.

    Parâmetros:
    - df: DataFrame a ser preenchido.

    Retorna:
    O DataFrame preenchido e um dicionário contendo as médias das colunas numéricas.
    '''
    numerical_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
    means = {}
    for col in numerical_cols:
        means[col] = df[col].mean()
        df[col].fillna(means[col], inplace=True)
    return df, means

def fillna_categorical(df):
    '''
    Preenche os valores nulos das colunas categóricas do DataFrame com o modo de cada coluna ou com 'VERIFICAR' se não houver um modo.

    Parâmetros:
    - df: DataFrame a ser preenchido.

    Retorna:
    O DataFrame preenchido e um dicionário contendo os valores usados para preencher as colunas categóricas.
    '''
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    modes = {}
    for col in categorical_cols:
        modes[col] = df[col].mode()[0] if not df[col].mode().empty else 'VERIFICAR'
        df[col].fillna(modes[col], inplace=True)
    return df, modes

# Função para preenchimento dos valores nulos em produção.
def fillna_num_prod(df, means):
    '''
    Esta função preenche os valores nulos do DataFrame em produção.

    Parâmetros:
    - df: DataFrame a ser preenchido.
    - means: Dicionário contendo as médias das colunas numéricas.

    Retorna:
    O DataFrame preenchido.
    '''

    for col, mean_value in means.items():
      df[col].fillna(mean_value, inplace=True)

    return df

# Função para preenchimento dos valores nulos em produção.
def fillna_catg_prod(df, modes):
    '''
    Esta função preenche os valores nulos das colunas categóricas do DataFrame em produção
    com os valores fornecidos em um dicionário de modos.

    Parâmetros:
    - df: DataFrame a ser preenchido.
    - modes: Dicionário contendo os valores (modos) para preencher as colunas categóricas.

    Retorna:
    O DataFrame preenchido.
    '''
    for col, mode_value in modes.items():
        if col in df.columns:  # Verifica se a coluna existe no DataFrame
            df[col].fillna(mode_value, inplace=True)

    return df