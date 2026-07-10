close all
clear all

% All of this for the example --- in real dataset, you would not need this
% part
mbar       = 20;          % index of tick just below q
num_ticks = mbar + 2;    % indices 0 to mbar+1

tau = 5; eta = -2;
ticks_geo = zeros(1, num_ticks);
for i = 0:mbar+1
    ticks_geo(i+1) = 1.0001^(tau*(i-eta)); 
end

% this is where code will actually start for real dataset --- it will only
% need to know the following:
% tick prices and tick indices where q belongs
% tick prices and tick indices where q' belongs
% all the tick prices and tick indices between the above two

% for example, if q \in (t_2, t_3) and q' \in (t_14, t_15)
% the code requires t_2, t_3, ...., t_15

psi = 1;
B       = 1;     % JIT LP budget
alpha   = .9;   % fee rate

% in real dataset, q will be the initial pool price
q = ticks_geo(1) + (ticks_geo(mbar+2)-ticks_geo(1))*rand(1,1);

% k_q: (tick index in which q lies) + 1

% for example, if q \in (t_3, t_4), then k_q = 4

% in the real dataset, k_q will be known; here, I am computing for the
% given example with fixed number of total ticks

if q == ticks_geo(mbar+2)
    k_q = 21;
else 
    k_q = func_k(ticks_geo, mbar, q);
end

px      = 1.0;     % USD price of token X
py      = q;     % USD price of token Y

fprintf('q = %.6f, p_x = %f, p_y = %f\n\n', q, px, py);
fprintf('q lies in the interval (t_{%d}, t_{%d}) \n', k_q-1, k_q');

% t(i) returns t_i using paper index i
t_geo = @(i) ticks_geo(i+1);

% prints ticks
fprintf('           ');
for i = 0:mbar+1
    fprintf('t_%d      ', i);
end
fprintf('\n');
fprintf('%.4f   ', ticks_geo); fprintf('\n\n');


% choose some q' : final pool price without JIT liquidity
q_prime_0 = ticks_geo(1) + (ticks_geo(mbar+2)-ticks_geo(1))*rand(1,1);

% k_geo: (tick index where q' lies) + 1

% for example, if q' \in (t_3, t_4), then k_geo = 4
k_geo = func_k(ticks_geo, mbar, q_prime_0);  % this function will not be needed for real dataset

fprintf('q''(0) = %f lies in the interval (t_{%d}, t_{%d})\n', q_prime_0, k_geo-1, k_geo');

Delta_x = 2.5;     % total trade size between q and q'

P_vector = [rand(1, abs(k_q - k_geo) + 1)]; % liquidity vector by Passive LP in tick ranges
    
output = main_function_data(alpha, Delta_x, B, q, px, py, t_geo, q_prime_0, k_geo, P_vector, psi, k_q);

JIT_optimal_utility = output{1};
Passive_fees = output{3};
JIT_fees = output{4};
optimal_tick_range = output{5};
optimal_liquidity_allocated_by_JIT = output{6};

fprintf('Optimal utility of JIT LP = %f \n',  JIT_optimal_utility);
fprintf('Fees earned by JIT LP at optimal solution = %f \n', JIT_fees);
fprintf('Fees earned by Passive LP at JIT LP''s optimal solution = %f \n', Passive_fees);
fprintf('Optimal tick range for JIT LP = %d \n', optimal_tick_range);
fprintf('Optimal liquidity added by JIT LP = %f \n', optimal_liquidity_allocated_by_JIT);



