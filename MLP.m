
clear; close all; clc;

X = [0 0 ;
     0 1 ;
     1 0 ;
     1 1]; 

D = [0 ;
     1 ;
     1 ;
     0];

neuronios_ocultos = 3;                    % Neur�nios na camada oculta.
max_epocas        = 10000;                % M�ximo de �pocas permitidas.
nPadroes          = size(X,1);            % N�mero de Padr�es;
X                 = [X ones(nPadroes,1)]; % Adicionando entrada unit�ria.
nEntradas         = size(X,2);            % Dimens�o do vetor de entrada. 

% Taxa de Aprendizado:

alpha    = 0.7;     % (camada oculta)

MSE = zeros(1,max_epocas); % Vetor para armazenar os erros de cada �poca.

convergencia = 0;          % Vari�vel que atesta a converg�ncia (0: FALSO)

while (convergencia == 0)
    
    % Gera��oo dos pesos: h? este la?o mais externo para caso n?o haja con-
    % verg�ncia, o processo seja reiniciado com novos pesos.
    
    w_oculta = (rand(nEntradas,neuronios_ocultos) - 0.5)/10;   
    w_saida  = (rand(1,neuronios_ocultos) - 0.5)/10;
    
    for epoca = 1:max_epocas
    
        % La�o para a varredura de todos os padr�es.
        for j = 1:nPadroes

            x = X(j,:); % Selecionando o padr�o.
            d = D(j,1); % Selecionando o valor de sa�da desejado.

            s_oculta = x*w_oculta;          % Ativa��o da camada oculta, 
            y_oculta = (tanh(s_oculta))';   % usando uma fun��o tanh(s).
            s_saida  = y_oculta'*w_saida';  % Ativa��o da camada de sa�da,
            y        = s_saida;             % usando uma fun��o linear.
            erro     = d - y;               % Erro na sa�da.

            % Ajuste dos pesos da camada de sa�da:

            delta_w_saida  = erro.*alpha .*y_oculta;
            w_saida        = w_saida + delta_w_saida';

            % Ajuste dos pesos da camada oculta:

            delta_w_oculta = ...
            alpha.*erro.*w_saida'.*(1-(y_oculta.^2))*x;
            w_oculta = w_oculta + delta_w_oculta';

        end
        
        % Fim da �poca. Ao fim da �poca, os pesos s�o testados para que se
        % conhe�a o erro quadr�tico m�dio da �poca.

        Y = w_saida*tanh(X*w_oculta)'; % Sa�da da Rede;
        for i = 1:nPadroes
            if(Y(i) > 0.5)
                Y(i) = 1;
            else
                Y(i) = 0;
            end
        end
        E = D - Y';                    % Erros calculados;
        MSE(epoca) = ((E' * E)^0.5)/4;          % Erro Quadr�tico M�dio.

        % Crit�rio de parada [Erro M�nimo Alcan�ado]
        if MSE(epoca) < 0.001
            convergencia = 1;
            break 
        end

    end
end

% Apresenta��o em tela dos resultados:

disp('Treinamento da MLP para o operador l�gico XOR - UFPI 2017');
plot(MSE(1:epoca))
grid on
title('MLP (Resultados): MSE x �pocas')
ylabel('MSE(�poca)')
xlabel('�poca')

disp('Resultados do Treinamento');
disp(' ');
disp(['Total de �pocas (converg�ncia): ',num2str(epoca)]);
disp(' ');
disp('Teste da Rede para os pesos encontrados: ')
for j=1:4
    disp(['X1 = ',num2str(X(j,1)),...
        '; X2 = ',num2str(X(j,2)),...
        '; D = ',num2str(D(j)),   ...
        '; Y rede = ',num2str(Y(j))]);
end