function [V, fees] = utility(L, P, dx, R, C, F, px, psi)
    if L <= 0
        V = 0; fees = 0; return;
    end
    total = L + P;
    const = px * (L/total) * dx;
    const_psi = px * (L^psi/(L^psi + P^psi)) * dx;
    fees = const_psi * (F-1);
    V = fees + const_psi - const* R*total/(C + total);
end