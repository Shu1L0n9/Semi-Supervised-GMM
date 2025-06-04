function out = normalise(in)
    % NORMALISE Normalize the input so that the sum of each row is 1
    % Input:
    %   in: Input matrix
    % Output:
    %   out: Normalized matrix
    
    in = double(in);  % Ensure the input is double precision
    
    % Compute the sum of each row
    row_sums = sum(in, 2);
    
    % If the sum is zero, set the result to zero
    row_sums(row_sums == 0) = 1;
    
    % Normalize
    out = in ./ row_sums;
end