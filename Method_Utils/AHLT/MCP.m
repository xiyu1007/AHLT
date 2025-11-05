function [f,df] = MCP(x,gamma)
    f = abs(x) - ( (x.^2) ./ (2*gamma) );

    id = abs(x) >= gamma;
    term = gamma / 2;
    f(id,1) = term;
    
    id = abs(x) >= gamma;
    df = sign(x) - (x ./ gamma);
    df(id,1) = 0;
end