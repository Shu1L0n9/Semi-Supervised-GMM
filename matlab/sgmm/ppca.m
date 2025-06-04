function [var, U, lambda] = ppca(x, ppca_dim)
  %PPCA	Probabilistic Principal Components Analysis
  %
  %	Description
  %	 [VAR, U, LAMBDA] = PPCA(X, PPCA_DIM) computes the principal
  %	component subspace U of dimension PPCA_DIM using a centred covariance
  %	matrix X. The variable VAR contains the off-subspace variance (which
  %	is assumed to be spherical), while the vector LAMBDA contains the
  %	variances of each of the principal components.  This is computed
  %	using the eigenvalue and eigenvector  decomposition of X.
  %
  %	See also
  %	EIGDEC, PCA
  %
  
  %	Copyright (c) Ian T Nabney (1996-2001)
  
  
  ppca_dim = double(ppca_dim);
  if ppca_dim ~= round(ppca_dim) || ppca_dim < 1 || ppca_dim > double(size(x, 2))
      error('Number of PCs must be integer, >0, < dim');
  end
  
  [~, data_dim] = size(double(x));
  % Assumes that x is centred and responsibility weighted
  % covariance matrix
  [l, Utemp] = eigdec(double(x), double(data_dim));
  % Zero any negative eigenvalues (caused by rounding)
  l = double(l);
  l(l < 0) = 0;
  % Now compute the sigma squared values for all possible values
  % of q
  s2_temp = cumsum(double(l(end:-1:1))) ./ (double((1:data_dim))');
  % If necessary, reduce the value of q so that var is at least
  % eps * largest eigenvalue
  q_temp = min([double(ppca_dim); data_dim - double(min(find(s2_temp ./ double(l(1)) > eps)))]);
  if q_temp ~= double(ppca_dim)
      wstringpart = 'Covariance matrix ill-conditioned: extracted';
      wstring = sprintf('%s %d/%d PCs', ...
          wstringpart, double(q_temp), double(ppca_dim));
      warning(wstring);
  end
  if q_temp == 0
      % All the latent dimensions have disappeared, so we are
      % just left with the noise model
      var = double(l(1)) / double(data_dim);
      % lambda = double(var) * ones(1, double(ppca_dim));
  else
      var = double(mean(l(q_temp+1:end)));
  end  
  U = double(Utemp(:, 1:double(q_temp)));
  lambda = double(zeros(1, double(ppca_dim)));
  lambda(1:double(q_temp)) = double(l(1:double(q_temp)));
end