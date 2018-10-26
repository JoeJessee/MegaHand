function varnames = makeVariableNames(funs,n)
% Make variable names with functions in them

assert(n<10,'n must be less than 10');
varnames = strcat(repmat(funs,n,1),repelem(cellstr(('1':num2str(n))'),numel(funs),1));

end