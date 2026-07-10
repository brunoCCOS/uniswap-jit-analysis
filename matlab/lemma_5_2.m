function [L_low, L_up, regime] = lemma_5_2( ...
        R, F, P, C, L_inner, L0, L_max, ratio, ...
        R_up, C_up, dx_up, L0_up, ...
        dx, px, psi)
    % L_low = L_{mbar-j},  L_up = L_{mbar-j+1}
    L_up = 0;
 
    if R > F && P >= F*C/(R-F)
        % (a)
        L_low  = 0;
        regime = '5.2(a)';
 
    elseif R < F && F < R*ratio && P >= R*C/(F-R)
        % (b)
        L_low  = L_max;
        regime = '5.2(b)';
 
    elseif (R < F && F < R*ratio && P < R*C/(F-R)) || ...
           (R > F && P < F*C/(R-F))
        % (c)
        L_low  = max(L0, min(L_inner, L_max));
        regime = '5.2(c)';
 
    else
        % (d): F > R*ratio
        % Candidate 1: L0_up in tick mbar-j+1, utility uses R_up, C_up, dx_up
        [U_upper, fees_upper] = utility(L0_up, P, dx_up, R_up, C_up, F, px, psi); 
        % Candidate 2: L_cand in tick mbar-j, utility uses R, C, dx
        L_cand  = max(L0, min(L_inner, L_max));
        % fprintf('L0 = %f, L_inner = %f, L_max = %f, L_cand = %f \n', L0, ...
        % L_inner, L_max, L_cand);
        [U_lower, fees_lower] = utility(L_cand, P, dx, R, C, F, px, psi); 
 
        if U_upper > U_lower
            L_low  = 0;
            L_up   = L0_up;
            regime = '5.2(d) - upper';
        else
            L_low  = L_cand;
            regime = '5.2(d) - lower';
        end
    end
    L_low = max(0, L_low);
end