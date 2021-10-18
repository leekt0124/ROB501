
The data are stored in cell arrays. 

http://www.mathworks.com/help/matlab/matlab_prog/access-data-in-a-cell-array.html

To access the measurement at time k, use curly brakcets
>> yk=y{k}; % This is a  3 x 1 vector of double precision real numbers

To access the model matrix at time k, also use curly brackets
>> Ck=C{k}; % This is a  3 x 100 vector of double precision real numbers

To access the actual state at time k
>> xk=x_actual{k}; % This is a 100 x 1 vector of double precision real numbers

K is the number of data points