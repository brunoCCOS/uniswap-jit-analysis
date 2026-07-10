function [Rj, Cj, Aj, dx_j, L0j, Lmaxj, Linnerj] = function_residual_and_more_data(k, q,...
    Delta_x, t, P_vector, py, px, F, B, k_q, q_prime_0)
    % Precompute residual trades 
    % and store R, C, A, eps, L_max, L0 for each j (needed for regime (d))
    Rj     = zeros(1, abs(k_q-k)+1);
    Cj     = zeros(1, abs(k_q-k)+1);
    Aj     = zeros(1, abs(k_q-k)+1); 
    dx_j   = zeros(1, abs(k_q-k)+1); % remaining trade
    L0j    = zeros(1, abs(k_q-k)+1); 

    Lmaxj  = zeros(1, abs(k_q-k)+1); % = B / eps_mj
    Linnerj = zeros(1, abs(k_q-k)+1); % = (C*A) / (1 - A) - P
     
    k_star = abs(k- k_q)+1;

    % R, C, A,.... are all written in reverse order 
    index = 1;
    for j = 0:abs(k_q-k)
        dx = Delta_x;
        for i = 0:j-1
            dx = dx - P_vector(k_star-i)*(1/sqrt(t(max(k, k_q)-i-1)) - 1/sqrt(t(max(k, k_q)-i)));
        end
        
        dx_j(index) = dx;

        if j == 0 && q > q_prime_0
            Rj(index) = q * (py/px);
            Cj(index) = dx * sqrt(q);
        else
            Rj(index) = t(max(k, k_q)-j) * (py/px);
            Cj(index) = dx * sqrt(t(max(k, k_q)-j));
        end
        
        Aj(index) = sqrt((F/Rj(index)) * (P_vector(k_star - j)/(Cj(index)+P_vector(k_star - j))));
     
        eps_mj      = sqrt(t(max(k, k_q)-j)) - sqrt(t(max(k, k_q)-j-1));
        Lmaxj(index)  = B / eps_mj;
        Linnerj(index) = (Cj(index)*Aj(index)) / (1 - Aj(index)) - P_vector(k_star - j);
     
        if j == 0
            L0j(index) = max(0, Delta_x/(1/sqrt(t(max(k, k_q)-1)) - 1/sqrt(t(max(k, k_q)))) - P_vector(k_star));
        elseif j == abs(k_q-k)+1
            L0j(index) = 0;
        else 
            Dm_j = dx - P_vector(k_star - j)*(1/sqrt(t(max(k, k_q)-j-1)) - 1/sqrt(t(max(k, k_q)-j)));
            if Dm_j < 0
                fprintf('leftover trade < 0\n');
            end
            L0j(index) = max(0, Dm_j / (1/sqrt(t(max(k, k_q)-j)) - 1/sqrt(t(max(k, k_q)-j+1))));
        end

        index = index + 1;

    end

    % fprintf('R sequence: \n');
    % disp(Rj);
    % 
    % fprintf('C sequence: \n');
    % disp(Cj);
    % 
    % fprintf('A sequence: \n');
    % disp(Aj);
    % 
    % fprintf('remaining trade (Delta x_j) sequence: \n');
    % disp(dx_j);
    % 
    % fprintf('L_0 sequence: \n');
    % disp(L0j);
    % 
    % fprintf('L_inner (CA/(1-A))-P sequence: \n');
    % disp(Linnerj);
    % 
    % fprintf('L_max = B/eps_j sequence: \n');
    % disp(Lmaxj);
end