function output = main_function_data(alpha, Delta_x, B, q, px, py, t, ...
            q_prime_0, k, P_vector, psi, k_q)

    %For each j = 0,...,k apply Lemma 5.1 or 5.2

    k_geo = k;
    F = alpha + 1;

    k_star = abs(k- k_q)+1;

    [Rj, Cj, Aj, dx_j, L0j, Lmaxj, Linnerj] = function_residual_and_more_data(k, q,...
        Delta_x, t, P_vector, py, px, F, B, k_q, q_prime_0);
    
     %% =====================================================================
    %  SOLVE for each j
    %  =====================================================================
    V     = zeros(1, abs(k_q-k)+1);
    fees     = zeros(1, abs(k_q-k)+1);
    L_opt = cell(1, abs(k_q-k)+1);
     
    index = 1;
    for j = 0:abs(k_q-k_geo)

        L_vec = zeros(1, j+1);   
     
        if index == 1
            [L_vec(1), regime] = lemma_5_1(Rj(1), F, P_vector(k_star), Cj(1), ...
                Linnerj(1), L0j(1), Lmaxj(1));
            [V(1), fees(1)] = utility(L_vec(1), P_vector(k_star), dx_j(1), Rj(1), Cj(1), F, px, psi);
   
        else
            ratio = sqrt(t(k_star - j+2) / t(k_star-j+1));
     
            [L_vec(1), L_vec(2), regime] = lemma_5_2( ...
                Rj(index), F, P_vector(k_star-j), Cj(index), Linnerj(index), L0j(index), Lmaxj(index), ratio, ...
                Rj(index-1),   Cj(index-1),   dx_j(index-1),   L0j(index-1),   ...
                dx_j(index), px, psi);
     
            % V(j+1): utility comes from whichever tick is active
            if L_vec(2) > 0
                % regime (d)-upper: liquidity in tick mbar-j+1
                [V(index), fees(index)] = utility(L_vec(2), P_vector(k_star-j+1), dx_j(index-1), Rj(index-1), Cj(index-1), F, px, psi);
            else
                % all other regimes: liquidity in tick mbar-j
                [V(index), fees(index)] = utility(L_vec(1), P_vector(k_star-j), dx_j(index), Rj(index), Cj(index), F, px, psi);
            end
        end
     
        L_opt{index} = L_vec;
        % 
        % fprintf('(t_{%d}, t_{%d}) | regime=%-14s | L=[', max(k_q, k_geo)- j-1, ...
        % max(k_q, k_geo)-j, regime);
        % fprintf('%.4f ', L_vec);
        % fprintf('] | U=%.6f\n \n', V(index));
        index = index + 1;
    end
     
    %% =====================================================================
    %  OPTIMAL j*
    %  =====================================================================
    [V_star, j_star_idx] = max(V);
    fees_JIT = fees(j_star_idx);
    j_star = j_star_idx - 1;
     
    % fprintf('\n=== OPTIMAL SOLUTION ===\n');
    % fprintf('recall, q''(0) lied in the interval (t_%d, t_%d)\n\n', k_geo-1, k_geo)
    % fprintf('Optimal tick range (t_%d, t_%d)\n', max(k_q, k_geo)-j_star-1, max(k_q, k_geo)-j_star);
    % fprintf('Optimal allocation strategy = ['); 
    % fprintf('%.6f ', L_opt{j_star_idx}); fprintf(']\n');
    % fprintf('Optimal utility = %.6f\n', V_star);
    % fprintf('Fees incurred by JIT LP wrt optimal allocation = %f \n', fees_JIT);
    
    passive_fees = func_passive_utility_data(P_vector, L_opt{j_star_idx}, px, psi, dx_j, k_star, k_q, k, F);

    output = {V_star, L_opt{j_star_idx}, passive_fees, fees_JIT, max(k_q, k_geo)-j_star-1, sum(L_opt{j_star_idx})};
end
