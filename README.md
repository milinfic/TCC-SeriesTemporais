# ESTUDO COMPARATIVO ENTRE ALGORITMOS DE MACHINE LEARNING APLICADOS À PREVISÃO DE SÉRIES TEMPORAIS DO MERCADO FINANCEIRO.

>  No presente trabalho foi realizado um estudo comparativo da capacidade preditiva entre algoritmos de Aprendizado de Máquina aplicados a um problema de regressão chamado Previsão de Séries Temporais. Especificamente, foram utilizados dados que compõem uma Série Temporal de Mercado Financeiro, a partir dos quais tenta-se prever o volume de negociação do mercado que é um dos indicadores mais importantes em problemas financeiros. Foram utilizados três métodos de Aprendizado de Máquina: os métodos de Regressão Linear e Redes Neurais Convencionais adaptados ao contexto de Séries Temporais e o método das Redes Neurais Recorrentes. Todas as implementações foram realizadas usando pacotes da linguagem Python específicos para Aprendizado de Máquina, Aprendizagem Profunda e Análise e Tratamento de Séries Temporais. Utilizando a métrica estatística R2 foi possível realizar uma análise comparativa entre os três algoritmos sob algumas condições de tratamento dos dados, como a normalização da série temporal.

# **Analise de predição de dados de volumes financeiro da bolsa de Nova York.**

Dados obtidos são de mostra estatísticas históricas de negociação da Bolsa de Nova York Intercâmbio. São mostradas três séries temporais diárias cobrindo o período de 3 de dezembro de 1962 a 31 de dezembro de 1986:

# Dados do DataSet

> O dataframe possui uma coluna chamada **_train_**, nessa coluna possui os dados **true** ou **false**, sendo true os dados para treinamento e false os dados para teste.

>  Possui dados de entrada referenciando os dias da semana, no caso a coluna tem o nome de **_day_of_week_**

> Possui também três sequências de dados temporaís, no qual a coluna **_Dj_return_** é o valor do retorno dos ativos, **_log_volatility_** é a volatilidade e **_log_volume_** é o volume e também é a variável que será utilizada para ser predita, consequentemente fazendo esse trabalho se uma autoRegressão

<p align="center">
  <img alt="DataSet" title="#DataSet" src="./Imagens/Screenshot 2022-12-28 at 14-09-22 TCC Paulo.pdf.png">
</p>

> Os dados de predição (target) foi separado com a lógica de tempo de atraso, ou seja, os itens terão uma quantidade de dados de entrada, essa quantidade será determinada com a variável L_lag(atraso).

## Exemplo:

L_lag = 5, será 5 dias de dados de entrada

<table width="90%" height="70"border="1px" >
    <tr border="1">
    <th width="80" style="text-align:center;">Vt-5</th>
    <th width="80" style="text-align:center;">Vt-4</th>
    <th width="80" style="text-align:center;">Vt-3</th>
    <th width="80" style="text-align:center;">Vt-2</th>
    <th width="80" style="text-align:center;">Vt-1</th>
    <th width="80" style="text-align:center;">Vt</th>
    </tr>
</table>

<br>

# Foi efetudo três métodos para utilização de variáveis categóricas, no caso, a variável é um dia da semana

# **_Explicação de cada método:_**
> * ### ***dia_sequencia.***
>  A lógica é apenas mudar o nome (string) do dia da semana para um valor, no caso, 1 para mon(segunda), 2 para tues(terça) ..., 5 para fri(sexta)

 ## Exemplo:
<table width="40px" border="1px">
 <tr border="1" height="60">
            <th width="80" style="text-align:center;">Dias Da Semana</th>
</tr>
<tr border="1" height="40">
    <td width="80" style="text-align:center;">1</td>
    </tr>
<tr border="1" height="40">
    <td width="80" style="text-align:center;">2</td>
</tr>
<tr border="1" height="40">
    <td width="80" style="text-align:center;">3</td>
</tr>
<tr border="1" height="40">
    <td width="80" style="text-align:center;">4</td>
</tr>
<tr border="1" height="40">
    <td width="80" style="text-align:center;">5</td>
</tr>
</table>

 > * ### ***dia_seno_cosseno***
 > Foram criado duas colunas e eliminado a coluna da string do dia da semana, as colunas criadas são: seno e cosseno, como são 5 dias da semana, pegamos o angulo total do circulo (360º) e dividimos por 5 (5  dias da semana), resultando 72º, ou seja, será 72 graus para cada dia da semana,assim sendo, mutiplicamos o valor do dia por 72 e depois aplicamos a formula de seno e cosseno
 
 ## Exemplo:

<table width="90%" border="1px" >
    <tr border="1" height="60">
    <th width="80" style="text-align:center;">DIA</th>
    <th width="80" style="text-align:center;">SENO</th>
    <th width="80" style="text-align:center;">COSSENO</th>
    </tr>
    <tr border="1" height="40">
    <td width="80" style="text-align:center;">Segunda</td>
    <td width="80" style="text-align:center;">seno(1*72)</td>
    <td width="80" style="text-align:center;">cosseno(1*72)</td>
    </tr>
    <tr border="1" height="40">
    <td width="80" style="text-align:center;">Terça</td>
    <td width="80" style="text-align:center;">seno(2*72º)</td>
    <td width="80" style="text-align:center;">cosseno(2*72º)</td>
    </tr>
    <tr border="1" height="40">
    <td width="80" style="text-align:center;">Quarta</td>
    <td width="80" style="text-align:center;">seno(3*72º)</td>
    <td width="80" style="text-align:center;">cosseno(3*72º)</td>
    </tr>
    <tr border="1" height="40">
    <td width="80" style="text-align:center;">Quinta</td>
    <td width="80" style="text-align:center;">seno(4*72º)</td>
    <td width="80" style="text-align:center;">cosseno(4*72º)</td>
    </tr>
    <tr border="1" height="40">
    <td width="80" style="text-align:center;">Sexta</td>
    <td width="80" style="text-align:center;">seno(5*72º)</td>
    <td width="80" style="text-align:center;">cosseno(5*72º)</td>
    </tr>
</table>
  
>  ### ***dia_get_dummies***
> Processo de codificação one-hot, para criar colunas de cada dia da semana e acescentando 
valor de 1 para a coluna do respectivo dia e zero para as demais.

 ## Exemplo:

<table width="90%" border="1px" >
    <tr border="1" height="60">
    <th width="80" style="text-align:center;">SEGUNDA</th>
    <th width="80" style="text-align:center;">TERÇA</th>
    <th width="80" style="text-align:center;">QUARTA</th>
    <th width="80" style="text-align:center;">QUINTA</th>
    <th width="80" style="text-align:center;">SEXTA</th>
    </tr>
    <tr border="1" height="40">
    <td width="80" style="text-align:center;">1</td>
    <td width="80" style="text-align:center;">0</td>
    <td width="80" style="text-align:center;">0</td>
    <td width="80" style="text-align:center;">0</td>
    <td width="80" style="text-align:center;">0</td>
    </tr>
    <tr border="1" height="40">
    <td width="80" style="text-align:center;">0</td>
    <td width="80" style="text-align:center;">1</td>
    <td width="80" style="text-align:center;">0</td>
    <td width="80" style="text-align:center;">0</td>
    <td width="80" style="text-align:center;">0</td>
    </tr>
    <tr border="1" height="40">
    <td width="80" style="text-align:center;">0</td>
    <td width="80" style="text-align:center;">0</td>
    <td width="80" style="text-align:center;">1</td>
    <td width="80" style="text-align:center;">0</td>
    <td width="80" style="text-align:center;">0</td>
    </tr>
    <tr border="1" height="40">
    <td width="80" style="text-align:center;">0</td>
    <td width="80" style="text-align:center;">0</td>
    <td width="80" style="text-align:center;">0</td>
    <td width="80" style="text-align:center;">1</td>
    <td width="80" style="text-align:center;">0</td>
    </tr>
    <tr border="1"height="40">
    <td width="80" style="text-align:center;">0</td>
    <td width="80" style="text-align:center;">0</td>
    <td width="80" style="text-align:center;">0</td>
    <td width="80" style="text-align:center;">0</td>
    <td width="80" style="text-align:center;">1</td>
    </tr>
</table>

>  ### ***Normalização/Padronização dos dados***
> Esse processo reposiciona os dados com a média no valor de 0.

 ## Exemplo:

<p align="center">
  <img alt="Normalizacao" title="#Normalizacao" src="./Imagens/Screenshot 2022-12-28 at 14-08-19 TCC Paulo.pdf.png">
</p>
<br>

>  ### ***Quantidade de dados de entrada***
> Foi efetaudo teste para ver a melhor quantidade de dados para ser utilizado, conforme pode ver no gráfico abaixo, cinco dados de entrada é suficiente para fazer uma boa predição.
 ## Exemplo:
<p align="center">
  <img alt="Tela-Login" title="#Scores" src="./Imagens/Screenshot 2022-12-28 at 14-16-20 TratamentoIniciaisDeDadosLinearRegression - Jupyter Notebook.png">
</p>
<br>

>  ### ***Tabela de Resultados***
> Dados retirados dos testes de melhor predição, esses valores foram pego no colab, utilizei a plataforma para fazer predição com quantidade maior de época do que esta atribuída nesse trabalho compartilhado, porém, vocês podem copiar e subir no colab, redirecionando os dados de entrada do df para o caminho que corresponde ao local que vocês estiverem utilizando.
<p align="center">
  <img alt="Tela-Login" title="#TabelaResultado" src="./Imagens/Screenshot 2022-12-28 at 14-07-44 TCC Paulo.pdf.png">
</p>

# Resultado
* O trabalho mostrou que não precisamos ter todos os dados para melhor predição, no caso, a estratégia e utilizar a quantidade melhor para predição, nesse trabalho apenas 5 dias ja nos permite ter uma boa predição de valores.
* Tratamento de dados é importante para predição, no caso do trabalho, ao tratar os dados categoricos, que no caso foi os dias da semana, conseguimos melhorar a predição
* Uma simples regressão linear do sktlear teve melhor, ou igual, predição do que as outras utilizadas, consequêntemente nos mostrando ser mais importante o tratamento dos dados para essa predição.
