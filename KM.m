
%K-Means Clustering
%=======================================================================
% (idx) is the Number of Clusters in K-Means
%=======================================================================
% (s) Structural Approach-the Silhouette Width index 
%=======================================================================

clc;clear;close all;format shortG
X=xlsread('C:\Users\Daviran\Desktop\GWOKM documents\Layer\input data_F2.xlsx',1);
P=X;
plot(X(:,1),X(:,2),'.');
title 'Data';
hold on;
disp('===========================================');

disp('K-means Clustering');
k=input('Enter the number of K:   ');

disp('===========================================');
opts = statset('Display','final');
[idx,C] = kmeans(X,k,'Distance','cityblock',...
    'Replicates',200,'Options',opts);

i=0;
figure;
hold on;
for i=1:k
plot(X(idx==i,1),X(idx==i,2),'.');
hold on;
plot(C(:,1),C(:,2),'kx',...
     'MarkerSize',5,'LineWidth',3)
 hold on;
title 'Cluster Assignments and Centroids'
 i=i+1;
end
figure;
silhouette(X,idx);
title('silhouette plot')
s = silhouette(X,idx,'cityblock');
Average=mean(s);

disp('===========================================')

disp('Sheet1=Centroide')

disp('===========================================')

disp('Sheet2=Average')

disp('===========================================')

disp('Sheet3=Cluster number')

disp('===========================================')

disp('Sheet4=s(i)')

disp('===========================================')
%***************************************************************
%Sheet1=Centroide

%Sheet2=Average

%Sheet3=Cluster number

%Sheet4=s(i)

%filename = 'Clusterexport.xlsx';
%xlswrite(filename,C,1)
%xlswrite(filename,Average,2)
%xlswrite(filename,idx,3)
%xlswrite(filename,s,4)
%% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%          Mehrdad Daviran & Reza Ghezelbash                 %
%         Amirkabir University of Technology                 %
%              Mehrdaddaviran@yahoo.com                      %
%              Rezaghezelbash@aut.ac.ir                      %      
%   Grey wolf optimizer with K-means Clustering (GWOKM)      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

