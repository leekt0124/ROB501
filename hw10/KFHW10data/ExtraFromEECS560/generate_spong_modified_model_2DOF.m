% generate_spong_modified_model_2DOF.m
%
% File to automatically build up the .m-files needed for our simualtor
%

%load Mat/work_symb_model_abs;

fcn_name = 'dyn_mod_segway';

disp(['[creating ',upper(fcn_name),'.m]']);
fid = fopen([fcn_name,'.m'],'w');
n=max(size(q));
fprintf(fid,['function [D,C,G,B,J,R,F_bar,M_bar]=' ...
        ' %s(q,dq)\n'],fcn_name);
fprintf(fid,'%%%s\n\n',upper(fcn_name));
fprintf(fid,'%%%s\n\n',datestr(now));
fprintf(fid,'\n%s','%');
fprintf(fid,'\n%s','% Authors(s): Grizzle');
fprintf(fid,'\n%s','%');
fprintf(fid,'\n%s','%');
fprintf(fid,'\n%s','% Model NOTATION: Spong and Vidyasagar, page 142, Eq. (6.3.12)');
fprintf(fid,'\n%s','%                 D(q)ddq + C(q,dq)*dq + G(q) = B*tau');
fprintf(fid,'\n%s','%');
fprintf(fid,'\n%s','%');
% if n==3
%      fprintf(fid,'\n%s','[g,L1,L2,L3,m1,m2,m3,I1,I2,I3,s1,s2,s3]=modelParameters;');
% elseif n==2
%     fprintf(fid,'\n%s','[g,L1,L2,m1,m2]=modelParameters_acrobot;');
% end
fprintf(fid,'\n%s','%');
fprintf(fid,'\n%s','%');
fprintf(fid,'\n%s','%');
fprintf(fid,'\n%s','%');

fprintf(fid,'\n%s','phi=q(1);theta=q(2);');
fprintf(fid,'\n%s','dphi=dq(1);dtheta=dq(2);');

fprintf(fid,'\n%s','%');
fprintf(fid,'\n%s','%');
fprintf(fid,'\n%s','%');
fprintf(fid,'\n%s','%');
fprintf(fid,'\n%s',['D=zeros(',num2str(n),',',num2str(n),');']);
for i=1:n
    for j=1:n
        Temp0=D(i,j);
        if Temp0 ~= 0
            Temp1=char(Temp0);
            Temp2=['D(',num2str(i),',',num2str(j),')=',Temp1,';'];
            Temp3=fixlength(Temp2,'*+-',65,'         ');
            fprintf(fid,'\n%s',Temp3);
        end
    end
end
% fprintf(fid,'\n%s','%');
fprintf(fid,'\n%s','%');
fprintf(fid,'\n%s','%');
fprintf(fid,'\n%s','%');
fprintf(fid,'\n%s',['C=zeros(',num2str(n),',',num2str(n),');']);
for i=1:n
    for j=1:n
        Temp0=C(i,j);
        if Temp0 ~= 0
            %ttt = char(vectorize(jac_P(2)));
            Temp1=char(Temp0);
            Temp2=['C(',num2str(i),',',num2str(j),')=',Temp1,';'];
            Temp3=fixlength(Temp2,'*+-',65,'         ');
            fprintf(fid,'\n%s',Temp3);
        end
    end
end
fprintf(fid,'\n%s','%');
fprintf(fid,'\n%s','%');
fprintf(fid,'\n%s','%');
fprintf(fid,'\n%s','%');
fprintf(fid,'\n%s',['G=zeros(',num2str(n),',1);']);
for i=1:n
    Temp1=char(G(i));
    Temp2=['G(',num2str(i),')=',Temp1,';'];
    Temp3=fixlength(Temp2,'*+-',65,'         ');
    fprintf(fid,'\n%s',Temp3);
end
fprintf(fid,'\n%s','%');
fprintf(fid,'\n%s','%');
fprintf(fid,'\n%s','%');
fprintf(fid,'\n%s','%');
[n,m]=size(B);
fprintf(fid,'\n%s',['B=zeros(',num2str(n),',',num2str(m),');']);
for i=1:n
    for j=1:m
        Temp0=B(i,j);
        if Temp0 ~= 0
            Temp1=char(Temp0);
            Temp2=['B(',num2str(i),',',num2str(j),')=',Temp1,';'];
            Temp3=fixlength(Temp2,'*+-',65,'         ');
            fprintf(fid,'\n%s',Temp3);
        end
    end
end
fprintf(fid,'\n%s','%');
fprintf(fid,'\n%s','%');
fprintf(fid,'\n%s','%');
fprintf(fid,'\n%s','%');
fprintf(fid,'\n%s','%');
fprintf(fid,'\n%s','% From Reyhanoglu et al., Dynamics and Control of a Class of Underactuated Mechanical Systems');
fprintf(fid,'\n%s','% IEEE Transaction Auto Control, Vol. 44, No. 9, September, 1999, pp. 1663-1671');
fprintf(fid,'\n%s','%');
n=length(q);str_n=num2str(n);
fprintf(fid,'\n%s',['M_bar=D([2:',str_n,'],[2:',str_n,'])-D([2:',str_n,'],[1])*inv(D(1,1))*D([1],[2:',str_n,']);']);
fprintf(fid,'\n%s',['F=C*dq+G; F_bar=F([2:',str_n,'])-D([2:',str_n,'],[1])*inv(D(1,1))*F(1);']);
fprintf(fid,'\n%s','%');
fprintf(fid,'\n%s','J=-D(1,2)/D(1,1);');
fprintf(fid,'\n%s','R=-F(1)/D(1,1);');
fprintf(fid,'\n%s','%');
fprintf(fid,'\n%s','%u=F_bar+M_bar*v;');
fprintf(fid,'\n%s','%');
fprintf(fid,'\n%s','%');


fprintf(fid,'\n%s','return');
status = fclose(fid)
