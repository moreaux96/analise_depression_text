# Projeto criado para calcular a precisão de tendencias depressivas em trechos da lingua portuguesa utlizando PLN 



![acurácia](./models/depre.png)



#Na sociedade contemporânea atual cada vez mais casos de depressão estão sendo diagnosticados, a partir disso foi pensando em um sistema de Inteligência Artificial que irá utilizar uma vertente da inteligência artificial que ajuda computadores a entender, interpretar e manipular a linguagem humana conhecida como PLN (Processamento de Linguagem Natural) para realizar a Análise de Sentimentos em textos, e identificar se o texto contém uma mensagem depressiva. Este trabalho tem o objetivo de desenvolver um protótipo de software que seja capaz de identificar padrões psicológicos depressivos em *textos livres da língua portuguesa. Para a realização deste trabalho, foi utilizada a linguagem Python com a biblioteca NLTK, a qual é alicerçada no teorema de Naive Bayes. Para criar a base de dados de teste, foram utilizados 2000 textos com tendências depressivas e 2000 textos randômicos coletados do Big Data do Twitter, que foram classificados manualmente. Em seguida foi realizado um tratamento dessa lista, composto pelas fases de remoção de palavras irrelevantes, palavras que não possuem peso na análise, e de extração de radical, para melhor aproveitamento do vocabulário. Com a base de treinamento já tratada, foi realizado o treinamento do algoritmo e obteve uma acurácia superior a 70% de probabilidades de tendências depressivas nos textos analisados
