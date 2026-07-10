
 
function [L_star, regime] = lemma_5_1(R, F, P, C, L_inner, L0, L_max)
    if R > F && P >= F*C/(R-F)
        L_star = L0;                              % (a)
        regime = '5.1(a)';
    elseif F > R && P >= R*C/(F-R)
        L_star = L_max;                           % (b)
        regime = '5.1(b)';
    else
        L_star = max(L0, min(L_inner, L_max));    % (c)
        if R > F && P < F*C/(R-F)
            regime = '5.1(c) - two';
        elseif F > R && P < R*C/(F-R)
            regime = '5.1(c) - one';
        end
    end
    L_star = max(0, L_star);
end