classdef AHLT
    properties
        name = 'AHLT';
        max_iter = 100;
        iter_tol = 1e-3;

        verbose = 0;

        info = '_';
        ncf = @SREP;

        % 优化变量
        % H; A; Z; D; E; U;
        % G; R;

        rho = 1e-4;
        rho_max = 1e8;
        delta = 2;
    end
    
    methods
        function obj = AHLT()
        end

        % Set parameters
        function [a,b,p,o,k,t] = setParams(~, params)
            a = params(1);
            b = params(2);
            p = params(3);
            o = params(4);
            k  = params(5);
            t  = params(6);
        end

        function [obj,pred,F,Loss] = run(obj,X,Y,params)
            %%
            % warning('off');
            [n, c, M, ~] = getDataInfo(X,Y);
            Loss = NaN(3,obj.max_iter);

            [a,b,p,o,k,t] = obj.setParams(params);

            %% INIT
            delta = obj.delta;
            rho = obj.rho;

            H = cell(M,1);

            if n >= 4000
                t = t * 10;
                anchors = DAS(X,t);
                SAG = cell(M,1); 
                for m=1:M
                    H{m} = KKT_Neighbor_Matrix(pd2(anchors{m},anchors{m},1),k);
                    SAG{m} = KKT_Neighbor_Matrix(pd2(anchors{m}, X{m}),k);
                end
                H = Diffusion(H, o);
                for m=1:M
                    H{m} = H{m} * SAG{m};
                end
                d = t;
            else
                if n > 166,  k = k * c; end
                t = t * c;
                for m=1:M
                    H{m} = KKT_Neighbor_Matrix(pd2(X{m},X{m},1),k);
                end
                H = Diffusion(H, o);
                d = n;
            end

            clear X Y SAG;

            H = cat(3, H{:});

            A = tensor(d, t, M, 'zeros');
            Z = tensor(t, n, M, 'zeros');
            D = Z;
            G = Z;
            R = G;
            E = tensor(d, n, M, 'zeros');
            U = E;

            %% 辅助变量
            sumD = sum(D, 3);   % t * n
            HEU = H - E + (U/rho);

            %% RUN
            for iter = 1:obj.max_iter
                %% Z
                term1 = pagemtimes(pagetranspose(A), HEU - pagemtimes(A,D)) + G + (R/rho);
                Z = term1 ./ 2;

                %% D
                term1 = pagemtimes(pagetranspose(A), HEU - pagemtimes(A,Z));
                rho2 = rho/2;
                for m=1:M
                    sumD = sumD - D(:,:,m);
                    term2 = rho2 * term1(:,:,m) - b* 0.5 *sumD;
                    D(:,:,m) =  term2 / ( rho2 + b);
                    sumD = sumD + D(:,:,m);
                end
                ZD = Z + D;

                %% E
                HEU = HEU + E;
                term1 = HEU - pagemtimes(A, ZD);
                norm2col = sqrt( sum( sum(term1.^2, 3) ,1) );  % 1 x n
                shrink_all = max(0, 1 - a ./ (rho * (norm2col + eps)));
                E = term1 .* reshape(shrink_all, [1, n, 1]);
                % 更新 HEU
                HEU = HEU - E;

                %% G
                ref = 1 / rho;
                term1 = Z - (R/rho);
                [G] = TNN(term1,ref,2,obj.ncf,p);

                %% A
                term1 = pagemtimes( HEU, pagetranspose(ZD) );
                [Ut, ~, Vt] = pagesvd(term1, 'econ');
                A = pagemtimes(Ut, pagetranspose(Vt));

                %% Lagrange
                term1 = G - Z;
                R = R + rho*(term1);
                HEU = HEU - (U/rho);
                term2 = HEU - pagemtimes(A, ZD);
                U = U + rho*(term2);
                rho = min(obj.rho_max,delta*rho);

                HEU = HEU + (U/rho);

                C1 = max(max(sum(abs(term1), 2), [], 1), [], 'all');
                C2 = max(max(sum(abs(term2), 2), [], 1), [], 'all');
                %% 
                Loss(1,iter) = C1;
                Loss(2,iter) = C2;
                err = max([C1,C2]);
                if err <  obj.iter_tol && iter >8 
                    break;
                end
            end
            catZ = reshape(permute(Z, [1, 3, 2]), [], n);
            Affinity = catZ'*catZ;
            [F,~] = eigs(Affinity, c, 'LA');
            rng('default')
            rng(42);
            [pred,F] = SpectralClustering(F, c, 1);
        end

        function parameter = init_param(~,fix)
            if nargin < 2
                fix = [Inf, Inf, Inf, Inf, Inf  Inf];
            end
            alphaSpace = [1e-4, 1e-3, 0.01, 0.1, 1];
            betaSpace = [1e-4, 1e-3, 0.01, 0.1, 1];
            pSpace = [0.01, 0.1, 1 10];
            oSpace = 2;
            kSpace = [1 3 5 7 9 ]; 
            tSpace = 3;

            if fix(1) ~= Inf
                alphaSpace = fix(1);
            end
            if fix(2) ~= Inf
                betaSpace = fix(2);
            end
            if fix(3) ~= Inf
                pSpace = fix(3);
            end
            if fix(4) ~= Inf
                oSpace = fix(4);
            end
            if fix(5) ~= Inf
                kSpace = fix(5);
            end
            if fix(6) ~= Inf
                tSpace = fix(6);
            end
            
            paramSpace = {alphaSpace,betaSpace, pSpace, oSpace, kSpace, tSpace};
            parameter = combvec(paramSpace{:})';
            parameter = sortrows(parameter, 'descend');
        end
    end
end