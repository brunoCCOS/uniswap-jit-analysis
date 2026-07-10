function fees = func_passive_utility(P_vec, JIT_opt_liq, px, psi, list_dx, k_star, k_q, k, F)

    fees = 0;
    for j = 0:abs(k_q-k)
        if j <=length(JIT_opt_liq)-1
            L = JIT_opt_liq(length(JIT_opt_liq)-j);
        else
            L = 0;
        end
        P = P_vec(k_star-j);
        dx = list_dx(j+1);
        const_psi =  px * (P^psi/(L^psi + P^psi)) * dx;

        fees = fees + const_psi*(F-1); 
     end
end
