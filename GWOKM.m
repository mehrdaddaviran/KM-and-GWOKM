%% Hybridizing Grey wolf optimizer Algorithm with K-means Clustering GWOKM
%=======================================================================
% (Final_Class_Label) is the Number of Clusters in GWOKM
%=======================================================================
%%
clc
clear
close all
format shortG
%% Insert Data

M=xlsread('C:\Users\Daviran\Desktop\GWOKM documents\Layer\input data_F2.xlsx',1);
AA=input('Number of Cluster:    ');
BB=input('Number of search agents (wolves):    '); 
CC=input('Maximum iter:    ');
k = AA; % Number of clusters
X=M;
% Set the parameters for GWO
N = BB; % Number of search agents (wolves)
Max_iter = CC; % Maximum number of iterations
lb = min(X(:)); % Lower bound of centroid values
ub = max(X(:)); % Upper bound of centroid values
dim = size(X, 2) * k; % Dimension of the search space

% Initialize the positions of search agents
Positions = zeros(N, dim);
for i = 1:N
    Centroids = lb + (ub-lb).*rand(k, size(X, 2)); % create k x d matrix
    Positions(i,:) = reshape(Centroids, [], 1)';
end
%for i = 1:N
 %   Positions(i,:) = lb + (ub-lb).*rand(1, dim);
%end

% Calculate the fitness of each search agent
for i = 1:N
    %Centroids = reshape(Positions(i,:), [], k);
    [idx, ~] = kmeans(X, k, 'Start', Centroids);
    Fitness(i) = (sum(sum((X - Centroids(idx,:)).^2)))/100000;
end
%%
%%Fitness = zeros(Max_iter+1, 1); % Initialize Fitness vecto
%%% % Initialize the fitness vs iteration plot

figure(2);
plot(1, Fitness(1),'Marker','.', 'MarkerSize', 28);
hold on;
xlabel('Iteration');
ylabel('Fitness');

% Find the best search agent
[Best_Fitness, Best_index] = min(Fitness);
Best_Position = Positions(Best_index,:);

% Main loop
for t = 1:Max_iter
    a = 2 - 2 * t / Max_iter; % Parameter a
    % Update the position of each search agent
    for i = 1:N
        A = 2 * a * rand() - a;
        C = 2 * rand();
        D = abs(C * Best_Position - Positions(i,:));
        X1 = Best_Position - A * D;
        % Update the centroids based on the new position of the search agent
        Centroids = reshape(X1, [], k);
        Centroids=Centroids';
        [idx, ~] = kmeans(X, k, 'Start', Centroids);
        Fitness_new = (sum(sum((X - Centroids(idx,:)).^2)))/100000;
        % Update the fitness and position of the search agent
        if Fitness_new < Fitness(i)
            Fitness(i) = Fitness_new;
            Positions(i,:) = X1;
        end

    end
    % Find the best search agent
    [Best_Fitness, Best_index] = min(Fitness);
    Best_Position = Positions(Best_index,:);
       %%% % Plot the current fitness value
    plot(t + 1, Best_Fitness,'Marker','.','MarkerSize', 28);
    drawnow; 
end
%%% % Plot the final fitness vs iteration
%%figure(2);
%%G=length(Fitness);
%%plot(1:G, Fitness,'Marker','.', 'MarkerSize', 28);
hold off;
%%
% Get the best centroids for the K-Means algorithm
best_centroids = reshape(Best_Position, [], k);
best_centroids=best_centroids';
% Apply the K-Means algorithm with the best centroids on the dataset
[idx, C] = kmeans(X, k, 'Start', best_centroids);
%%
% Plot the results
figure(1);
scatter(X(:,1), X(:,2), [], idx);
hold on;
plot(C(:,1), C(:,2), 'kx', 'MarkerSize', 15, 'LineWidth', 3);
title('K-Means clustering with Grey Wolf Optimizer initialization');


%% Concolusion
Final_Class_Label= idx(:,1);
figure;
silhouette(X,Final_Class_Label);
title('silhouette plot')
s_GWOKM = silhouette(X,Final_Class_Label,'cityblock');
Average_GWOKM=mean(s_GWOKM);
%%
disp('================================================')
disp('Best Fitness                          ')
disp('================================================')
disp(Best_Fitness);
disp('================================================')
disp('Silhouette                 ')
disp('================================================')
disp(Average_GWOKM);
%%
disp('++++++++++++++++++++');
disp('Finished');

%% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%          Mehrdad Daviran & Reza Ghezelbash                 %
%         Amirkabir University of Technology                 %
%              Mehrdaddaviran@yahoo.com                      %
%              Rezaghezelbash@aut.ac.ir                      %      
%   Grey wolf optimizer with K-means Clustering (GWOKM)      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%