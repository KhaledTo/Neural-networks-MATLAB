%Charger la toolbox
addpath 'netlab3';

%Fichier options
foptions;
global options

data = load('Sunspots');
dataSunspots = data(:,2);

% 3 neurones cachés 
lancer_experiences (3, '3Neurones', dataSunspots);
% 20 neurones cachés 
lancer_experiences (20, '20Neurones', dataSunspots);
