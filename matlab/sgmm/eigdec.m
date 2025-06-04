function [evals, evec] = eigdec(x, N)
    %EIGDEC	Sorted eigendecomposition
    %
    %	Description
    %	 EVALS = EIGDEC(X, N) computes the largest N eigenvalues of matrix X, sorted in descending order.
    %	 [EVALS, EVEC] = EIGDEC(X, N) also computes the corresponding eigenvectors.
    %
    %	See also
    %	PCA, PPCA
    %
    
    % Copyright (c) Ian T Nabney (1996-2001)
    
    x = double(x);
    N = double(N);
    
    if nargout == 1
       evals_only = true;
    else
       evals_only = false;
    end
    
    if N ~= round(N) || N < 1 || N > size(x, 2)
       error('Number of PCs must be integer, >0, < dim');
    end
    
    % Find the eigenvalues of the data covariance matrix
    if evals_only
       % Use eig function as always more efficient than eigs here
       temp_evals = eig(x);
    else
       % Use eig function unless fraction of eigenvalues required is tiny
       if (N/size(x, 2)) > 0.04
          [temp_evec, temp_evals] = eig(x);
       else
          options = struct('disp', 0);
          [temp_evec, temp_evals] = eigs(x, N, 'LM', options);
       end
       temp_evals = diag(double(temp_evals));
    end
    
    % Eigenvalues nearly always returned in descending order, but just
    % to make sure.....
    [evals, perm] = sort(-temp_evals);
    evals = -evals(1:N);
    if ~evals_only
       if isequal(evals, temp_evals(1:N))
          % Originals were in order
          evec = double(temp_evec(:, 1:N));
          return
       else
          % Need to reorder the eigenvectors
          evec = double(zeros(size(temp_evec, 1), N));
          for i = 1:N
             evec(:,i) = double(temp_evec(:, perm(i)));
          end
       end
    end
end