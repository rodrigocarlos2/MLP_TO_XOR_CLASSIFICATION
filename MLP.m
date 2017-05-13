
clear; close all; clc;

X = [0 0 ;
     0 1 ;
     1 0 ;
     1 1]; 

D = [0 ;
     1 ;
     1 ;
     0];

neuronios_ocultos = 3;                    % Neurônios na camada oculta.
max_epocas        = 10000;                % Máximo de épocas permitidas.
nPadroes          = size(X,1);            % Número de Padrões;
X                 = [X ones(nPadroes,1)]; % Adicionando entrada unitária.
nEntradas         = size(X,2);            % Dimensão do vetor de entrada. 

% Taxa de Aprendizado:

alpha    = 0.7;     % (camada oculta)

MSE = zeros(1,max_epocas); % Vetor para armazenar os erros de cada época.

convergencia = 0;          % Variável que atesta a convergência (0: FALSO)

while (convergencia == 0)
    
    % Geraçãoo dos pesos: h? este la?o mais externo para caso n?o haja con-
    % vergência, o processo seja reiniciado com novos pesos.
    
    w_oculta = (rand(nEntradas,neuronios_ocultos) - 0.5)/10;   
    w_saida  = (rand(1,neuronios_ocultos) - 0.5)/10;
    
    for epoca = 1:max_epocas
    
        % Laço para a varredura de todos os padrões.
        for j = 1:nPadroes

            x = X(j,:); % Selecionando o padrão.
            d = D(j,1); % Selecionando o valor de saída desejado.

            s_oculta = x*w_oculta;          % Ativação da camada oculta, 
            y_oculta = (tanh(s_oculta))';   % usando uma função tanh(s).
            s_saida  = y_oculta'*w_saida';  % Ativação da camada de saída,
            y        = s_saida;             % usando uma função linear.
            erro     = d - y;               % Erro na saída.

            % Ajuste dos pesos da camada de saída:

            delta_w_saida  = erro.*alpha .*y_oculta;
            w_saida        = w_saida + delta_w_saida';

            % Ajuste dos pesos da camada oculta:

            delta_w_oculta = ...
            alpha.*erro.*w_saida'.*(1-(y_oculta.^2))*x;
            w_oculta = w_oculta + delta_w_oculta';

        end
        
        % Fim da época. Ao fim da época, os pesos são testados para que se
        % conheça o erro quadrático médio da época.

        Y = w_saida*tanh(X*w_oculta)'; % Saída da Rede;
        for i = 1:nPadroes
            if(Y(i) > 0.5)
                Y(i) = 1;
            else
                Y(i) = 0;
            end
        end
        E = D - Y';                    % Erros calculados;
        MSE(epoca) = ((E' * E)^0.5)/4;          % Erro Quadrático Médio.

        % Critério de parada [Erro Mínimo Alcançado]
        if MSE(epoca) < 0.001
            convergencia = 1;
            break 
        end

    end
end

% Apresentação em tela dos resultados:

disp('Treinamento da MLP para o operador lógico XOR - UFPI 2017');
plot(MSE(1:epoca))
grid on
title('MLP (Resultados): MSE x Épocas')
ylabel('MSE(Época)')
xlabel('Época')

disp('Resultados do Treinamento');
disp(' ');
disp(['Total de épocas (convergência): ',num2str(epoca)]);
disp(' ');
disp('Teste da Rede para os pesos encontrados: ')
for j=1:4
    disp(['X1 = ',num2str(X(j,1)),...
        '; X2 = ',num2str(X(j,2)),...
        '; D = ',num2str(D(j)),   ...
        '; Y rede = ',num2str(Y(j))]);
end