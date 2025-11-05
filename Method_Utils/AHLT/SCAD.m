function [f,df] = SCAD(x,gamma)
    a = 3.7;

    df = (a*gamma .* sign(x) - x) / (a-1);
    id = abs(x) > a*gamma;
    df(id,1) = 0;
    id = abs(x) <= gamma;
    df(id,1) = gamma .* sign(x(id,1));
    
    f = (2*a*gamma .* abs(x) - x.^2 - gamma^2) / (2*(a-1));
    id = abs(x) > a*gamma;
    f(id,1) = gamma^2*(a+1) / 2;
    id = abs(x) <= gamma;
    f(id,1) = gamma .*abs(x(id,:));
end