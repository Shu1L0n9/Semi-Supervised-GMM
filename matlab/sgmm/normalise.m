function out = normalise(in)
    % NORMALISE 将输入矩阵的每一行归一化，使其和为 1
    % 输入:
    %   in: 输入矩阵
    % 输出:
    %   out: 归一化后的矩阵
    
    in = double(in);  % 确保输入为双精度
    
    % 计算每行的和
    row_sums = sum(in, 2);
    
    % 防止除以 0
    row_sums(row_sums == 0) = 1;
    
    % 归一化
    out = in ./ row_sums;
end