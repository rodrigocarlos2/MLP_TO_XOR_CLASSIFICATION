
passo = .75;
alpha = 0.001;

it = 4000;
MSEmin = 1e-20;

X = [0 0 1 1;
     0 1 0 1];
D = [0 1 1 0];

[Wx,Wy,MSE]=MLP(X,D, passo, alpha, it,MSEmin);

semilogy(MSE);

disp(['D = [' num2str(D) ']']);

[p1 N] = size (X);

bias = -1;

X = [bias*ones(1,N) ; X];

V = Wx*X;
Z = 1./(1+exp(-V));

S = [bias*ones(1,N);Z];
G = Wy*S;

Y = 1./(1+exp(-G));
disp(['Y = [' num2str(Y) ']']);