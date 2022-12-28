import pandas as pd
import numpy as np

################################################################################################
# Método responsável para plotar os gráficos de comparação do dados de saída real e o previsto #
################################################################################################

def plotResult(y_test, media):
    from matplotlib import font_manager
    import matplotlib.pyplot as plt

    font1 = {'family':'serif','color':'black','size':15}
    font2 = {'family':'serif','color':'darkred','size':20}

    plt.figure(figsize=(16,4))

    plt.title("Saída de Real vs Teste", fontdict = font2)
    plt.ylabel("log_volume", fontdict = font1)
    plt.xlabel("Dias", fontdict = font1)

    plt.plot(y_test, color='gray')

    plt.plot(media, color='blue')

    plt.xlim(0, 250)

    plt.legend(['Conjunto de saída esperada', 'Conjunto de sáida prevista'])
    plt.show()

################################################################################################
# Método para plotar Histograma                                                                #
################################################################################################
    
def plot_histograma(titulo, titulox, df): 
    import matplotlib.pyplot as plt

    # Tamanho da imagem principal em polegadas
    plt.figure(figsize = ((16, 8)))
      
    plt.suptitle(titulo, fontsize = 20, color='black')

    #Gráfico 1:
    plt.subplot(1, 2, 1)
    plt.xlabel('VALOR PADRÃO', fontsize = 14)
    plt.hist(df['log_volatility'], color='darkorange')

    #Gráfico 2:
    plt.subplot(1, 2, 2)
    plt.hist(df['log_volatility_stand'])
    plt.xlabel(titulox, fontsize = 14)

    plt.show()

################################################################################################
# Método para atualização do valor do Beta                                                     #
################################################################################################

def UpdateBeta(beta):
    df = pd.read_csv('nyse_df.csv')    
    
    sigma_0 = np.exp(df['log_volatility'][1])
    sigma_0

    lista = [sigma_0]
    sigma = sigma_0
    for i in range(2,len(df)+1):
        r = df['DJ_return'][i]
        sigma = beta*sigma + (1.- beta)*(r*r)
        lista.append(sigma)

    n_volat = np.array(lista)    
    n_volat = np.log(n_volat)

    df['log_volatility'] = n_volat
    
    return df

################################################################################################
# Método para tratamento de dados categoricos, considerações sequênciais                       #
# A lógica é apenas mudar o nome (string) do dia da semana para um valor, no caso:             #
# 1 para mon(segunda), 2 para tues(terça) ..., 5 para fri(sexta)                               #
################################################################################################

def dia_sequencia(L_lag):

    df = pd.read_csv('nyse_df.csv')

    x_train, y_train, x_test, y_test = nysedf(df, L_lag)

    df['day_of_week'] = df['day_of_week'].map({'mon': 1,'tues': 2,'wed': 3,'thur': 4,'fri':5},na_action=None)
    df.reset_index(drop=True, inplace=True)

    df_train = df[df['train'] == True]
    df_test = df[df['train'] == False]

    df_train = df[df['train'] == True]
    df_test = df[df['train'] == False]

    x = df_train['day_of_week'][5:]
    x.reset_index(drop=True, inplace=True)
    x_train['day_of_week'] = x

    y = df_test['day_of_week'][5:]
    y.reset_index(drop=True, inplace=True)
    x_test['day_of_week'] = y
    
    return(x_train, y_train, x_test, y_test)

################################################################################################
# Método para tratamento de dados categoricos, considerações dia_seno_cosseno                  #
# Como são 5 dias da semana, pegamos o angulo total do circulo (360º) e dividimos por 5        #
# (5 dias da semana), resultando 72º, ou seja, será 72 graus para cada dia da semana,          #
# mutiplicamos o valor do dia por 72 e depois aplicamos a formula de seno e cosseno            #
################################################################################################

def dia_seno_cosseno(df):
    import math
    delta=72
    cosseno = []
    seno = []

    for i in df.day_of_week:
        cosseno.append(round(math.cos(math.radians(i*delta)), 3))
        seno.append(round(math.sin(math.radians(i*delta)), 3))
    df['cosseno'] = cosseno
    df['seno'] = seno
    df.drop(columns='day_of_week', inplace=True)

    return df

################################################################################################
# Método para tratamento de dados categoricos, considerações dia_get_dummies                   #
# Processo de codificação one-hot, para criar colunas de cada dia da semana e acescentando     #
# valor de 1 para a coluna do respectivo dia e zero para as demais.                            #
################################################################################################

def dia_get_dummies(df):
    df = pd.get_dummies(df, columns=['day_of_week'])
    return df

################################################################################################
# Método responsável para a separação dos dias sequencias para testes, no caso a variável      #
# L_lag é a determinação de quantos dias de atraso serão criados para os dados sequenciais.    #
################################################################################################

def nysedf(df, L_lag): 
        
    df_train = df[df['train'] == True]
    df_test = df[df['train'] == False]
    
    v_lagged_train = pd.concat([df_train['log_volume'].shift(i) for i in range(L_lag)], axis = 1)
    v_lagged_train.dropna(inplace = True), 
    v_lagged_train.reset_index(inplace=True, drop=True)

    r_lagged_train = pd.concat([df_train['DJ_return'].shift(i) for i in range(L_lag)], axis = 1)
    r_lagged_train.dropna(inplace = True), 
    r_lagged_train.reset_index(inplace=True, drop=True)

    z_lagged_train = pd.concat([df_train['log_volatility'].shift(i) for i in range(L_lag)], axis = 1)
    z_lagged_train.dropna(inplace = True), 
    z_lagged_train.reset_index(inplace=True, drop=True)

    x_train = pd.concat([v_lagged_train, r_lagged_train, z_lagged_train], axis = 1)

    y_train = df_train['log_volume'].shift(-L_lag)
    y_train.dropna(inplace=True)
    y_train.reset_index(inplace=True, drop=True)

    x_train.drop(labels=(len(x_train)-1), axis=0, inplace=True)
    
    ##################################################################################################
    # Dados de Teste
    ##################################################################################################
    
    v_lagged_test = pd.concat([df_test['log_volume'].shift(i) for i in range(L_lag)], axis = 1)
    v_lagged_test.dropna(inplace = True), 
    v_lagged_test.reset_index(inplace=True, drop=True)

    r_lagged_test = pd.concat([df_test['DJ_return'].shift(i) for i in range(L_lag)], axis = 1)
    r_lagged_test.dropna(inplace = True), 
    r_lagged_test.reset_index(inplace=True, drop=True)

    z_lagged_test = pd.concat([df_test['log_volatility'].shift(i) for i in range(L_lag)], axis = 1)
    z_lagged_test.dropna(inplace = True), 
    z_lagged_test.reset_index(inplace=True, drop=True)

    x_test = pd.concat([v_lagged_test, r_lagged_test, z_lagged_test], axis = 1)

    y_test = df_test['log_volume'].shift(-L_lag)
    y_test.dropna(inplace=True)
    y_test.reset_index(inplace=True, drop=True)

    x_test.drop(labels=(len(x_test)-1), axis=0, inplace=True)
    
    x_train.columns = ['Vt-5', 'Vt-4', 'Vt-3', 'Vt-2', 'Vt-1', 'Rt-5', 'Rt-4', 'Rt-3' , 'Rt-2' , 'Rt-1', 'Zt-5', 'Zt-4', 'Zt-3', 'Zt-2', 'Zt-1']
    x_test.columns = ['Vt-5', 'Vt-4', 'Vt-3', 'Vt-2', 'Vt-1', 'Rt-5', 'Rt-4', 'Rt-3' , 'Rt-2' , 'Rt-1', 'Zt-5', 'Zt-4', 'Zt-3', 'Zt-2', 'Zt-1']
    
    
    return (x_train, y_train, x_test, y_test)