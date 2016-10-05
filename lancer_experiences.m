function lancer_experiences (nb_n_hid, nom_exp, dataSunspots)

%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Expérience 1 à 4 : 3 neurones cachés
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%}

    %{
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Expérience 1 : 12 variables input
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %}

    %Comme en TP

    %Data input
    inExp(:,1) = dataSunspots(1:268);
    inExp(:,2) = dataSunspots(2:269);
    inExp(:,3) = dataSunspots(3:270);
    inExp(:,4) = dataSunspots(4:271);
    inExp(:,5) = dataSunspots(5:272);
    inExp(:,6) = dataSunspots(6:273);
    inExp(:,7) = dataSunspots(7:274);
    inExp(:,8) = dataSunspots(8:275);
    inExp(:,9) = dataSunspots(9:276);
    inExp(:,10) = dataSunspots(10:277);
    inExp(:,11) = dataSunspots(11:278);
    inExp(:,12) = dataSunspots(12:279);
    % Data output
    Sortie = dataSunspots(13:280);
    % Bases : Apprentissage, Validation et Test

    % Apprentissage
    DAppInput = inExp(1:209,:);
    DAppOutput = Sortie(1:209);
    % Validation
    DValInput = inExp(210:244,:);
    DValOutput = Sortie(210:244);
    % Test
    DTestInput = inExp(245:268,:); 
    DTestOutput = Sortie(245:268); 
    
    % On a 12 variables en entrée 
    nb_n_in = size(DAppInput,2);
    nb_n_out = size(DAppOutput,2);
   
    nom_fichier = strcat('Net1_', nom_exp, '.mat');
    %la fonction 'apprendre' retourne le meilleur réseau
    [BestNet] = apprendre(nb_n_in, nb_n_hid, nb_n_out, DAppInput, DAppOutput, DValInput, DValOutput, nom_fichier, 1);
  
    fprintf('\n\n------------------ Experience 1 ------------ %d neurones ------------------', nb_n_hid); 
    
    %Calcul des ARV par rapport à l'apprentissage sur l'echantillon 
    %d'apprentissage
    calc_ARV(BestNet, DAppInput, DAppOutput, dataSunspots);

    %Calcul de l'ARV par rapport à l'apprentissage sur l'echantillon de
    %validation
    calc_ARV(BestNet, DValInput, DValOutput, dataSunspots);
 
    %Calcul de l'ARV par rapport à l'apprentissage sur l'echantillon 
    %de test 
    calc_ARV(BestNet, DTestInput, DTestOutput, dataSunspots);
    
    %on réinitialise certaines variables
    clearvars minErr errVerif errApp errVal BestNet DAppInput DAppOutput DValInput DValOutput DTestInput DTestOutput
    
    %{
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Expérience 2 : 6 variables input
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %}   
    %Data input
    inExp(:,1) = dataSunspots(1:268);
    inExp(:,2) = dataSunspots(2:269);
    inExp(:,3) = dataSunspots(3:270);
    inExp(:,4) = dataSunspots(8:275);
    inExp(:,5) = dataSunspots(11:278);
    inExp(:,6) = dataSunspots(12:279);
    % Data output
    Sortie = dataSunspots(13:280);
    % Bases : Apprentissage, Validation et Test

    % Apprentissage
    DAppInput = inExp(1:209,:);
    DAppOutput = Sortie(1:209);
    % Validation
    DValInput = inExp(210:244,:);
    DValOutput = Sortie(210:244);
    % Test
    DTestInput = inExp(245:268,:); 
    DTestOutput = Sortie(245:268); 

    % On a 6 variables en entrée 
    nb_n_in = size(DAppInput,2);
    nb_n_out = size(DAppOutput,2);
 
    nom_fichier = strcat('Net2_', nom_exp, '.mat');
    [BestNet] = apprendre(nb_n_in, nb_n_hid, nb_n_out, DAppInput, DAppOutput, DValInput, DValOutput, nom_fichier, 1);
  
    fprintf('\n\n------------------ Experience 2 ------------ %d neurones ------------------', nb_n_hid); 
    %Calcul des ARV par rapport à l'apprentissage sur l'echantillon 
    %d'apprentissage
    calc_ARV(BestNet, DAppInput, DAppOutput, dataSunspots);

    %Calcul de l'ARV par rapport à l'apprentissage sur l'echantillon de
    %validation
    calc_ARV(BestNet, DValInput, DValOutput, dataSunspots);
 
    %Calcul de l'ARV par rapport à l'apprentissage sur l'echantillon 
    %de test 
    calc_ARV(BestNet, DTestInput, DTestOutput, dataSunspots);
    
    %on réinitialise certaines variables
    clearvars nb_n_in nb_n_out minErr errVerif errApp errVal BestNet DAppInput DAppOutput DValInput DValOutput DTestInput DTestOutput
    
    %{
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Expérience 3 : supprimer 6 variables de Net1 d'un coup
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %}
    
    %Charger le premier réseau 
    nom_fichier = strcat('reseaux/','Net1_', nom_exp, '.mat');
    load(nom_fichier,'Net');
    
    %réinitialiser le dataset 
    %Data input
    inExp(:,1) = dataSunspots(1:268);
    inExp(:,2) = dataSunspots(2:269);
    inExp(:,3) = dataSunspots(3:270);
    inExp(:,4) = dataSunspots(4:271);
    inExp(:,5) = dataSunspots(5:272);
    inExp(:,6) = dataSunspots(6:273);
    inExp(:,7) = dataSunspots(7:274);
    inExp(:,8) = dataSunspots(8:275);
    inExp(:,9) = dataSunspots(9:276);
    inExp(:,10) = dataSunspots(10:277);
    inExp(:,11) = dataSunspots(11:278);
    inExp(:,12) = dataSunspots(12:279);
    
    Sortie = dataSunspots(13:280);
    
    %Supprimer les 6 variables
    colSpr = [10, 9, 7, 6, 5, 4];
    inExp(:,colSpr) = [];
    
     % Apprentissage
    DAppInput = inExp(1:209,:);
    DAppOutput = Sortie(1:209);
    % Validation
    DValInput = inExp(210:244,:);
    DValOutput = Sortie(210:244);
    % Test
    DTestInput = inExp(245:268,:); 
    DTestOutput = Sortie(245:268); 
    
    %Réseaux : supprimer les neurones en entrée
    Net.nin = Net.nin - 6;
    % Ajuster nwts
    Net.nwts = Net.nwts - (Net.nhidden * 6);
    %Supprimer les poids qui partent des variables supprimées
    Net.w1(colSpr,:) = [];
    
    %entraîner à nouveau ce réseau après avoir pris en compte la
    %suppression des 6 variables
    nom_fichier = strcat('Net3_', nom_exp, '.mat');
    [BestNet] = apprendre(Net.nin, Net.nhidden, Net.nout, DAppInput, DAppOutput, DValInput, DValOutput, nom_fichier, 1);

    fprintf('\n\n------------------ Experience 3 ------------ %d neurones ------------------', nb_n_hid); 
    
    %Calcul des ARV par rapport à l'apprentissage sur l'echantillon 
    %d'apprentissage
    calc_ARV(BestNet, DAppInput, DAppOutput, dataSunspots);

    %Calcul de l'ARV par rapport à l'apprentissage sur l'echantillon de
    %validation
    calc_ARV(BestNet, DValInput, DValOutput, dataSunspots);
 
    %Calcul de l'ARV par rapport à l'apprentissage sur l'echantillon 
    %de test 
    calc_ARV(BestNet, DTestInput, DTestOutput, dataSunspots);
    
    clearvars Net minErr errVerif errApp errVal BestNet DAppInput DAppOutput DValInput DValOutput DTestInput DTestOutput Sortie inExp
    
    %{
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Expérience 4 : supprimer 6 variables de Net1 progressivement
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %}
    
        %{
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Ordre 1 x(t-10), x(t-9), x(t-7), x(t-6), x(t-5), x(t-4)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %}
    
        %Charger le premier réseau 
        nom_fichier = strcat('reseaux/','Net1_', nom_exp, '.mat');
        load(nom_fichier,'Net');

        %réinitialiser le dataset 
        %Data input
        inExp(:,1) = dataSunspots(1:268);
        inExp(:,2) = dataSunspots(2:269);
        inExp(:,3) = dataSunspots(3:270);
        inExp(:,4) = dataSunspots(4:271);
        inExp(:,5) = dataSunspots(5:272);
        inExp(:,6) = dataSunspots(6:273);
        inExp(:,7) = dataSunspots(7:274);
        inExp(:,8) = dataSunspots(8:275);
        inExp(:,9) = dataSunspots(9:276);
        inExp(:,10) = dataSunspots(10:277);
        inExp(:,11) = dataSunspots(11:278);
        inExp(:,12) = dataSunspots(12:279);

        Sortie = dataSunspots(13:280);
        
        %Supprimer les 6 variables une à une et entraîner le réseau à
        %chaque fois (on utilise une boucle)
        
        colSpr = [10, 9, 7, 6, 5, 4];
        
        nom_fichier = strcat('Net4_', nom_exp, '.mat');
        for i = 1:size(colSpr,2)
            inExp(:,colSpr(i)) = [];
            
            % Apprentissage
            DAppInput = inExp(1:209,:);
            DAppOutput = Sortie(1:209);
            % Validation
            DValInput = inExp(210:244,:);
            DValOutput = Sortie(210:244);
            % Test
            DTestInput = inExp(245:268,:); 
            DTestOutput = Sortie(245:268); 
       
            %Réseaux : supprimer les neurones en entrée un à un 
            Net.nin = Net.nin - 1;
            % Ajuster nwts 
            Net.nwts = Net.nwts - (Net.nhidden);
            %Supprimer les poids qui partent des variables supprimées
            Net.w1(colSpr(i),:) = [];

            %entraîner à nouveau ce réseau après avoir pris en compte la
            %suppression des 6 variables 
            %nb : pas de sortie graph (graph à 0)
            [BestNet] = apprendre(Net.nin, Net.nhidden, Net.nout, DAppInput, DAppOutput, DValInput, DValOutput, nom_fichier, 0);
        end 
        
        fprintf('\n\n------------------ Experience 4 ------------ %d neurones ------------------', nb_n_hid); 
        %Calcul des ARV par rapport à l'apprentissage sur l'echantillon 
        %d'apprentissage
        calc_ARV(BestNet, DAppInput, DAppOutput, dataSunspots);

        %Calcul de l'ARV par rapport à l'apprentissage sur l'echantillon de
        %validation
        calc_ARV(BestNet, DValInput, DValOutput, dataSunspots);

        %Calcul de l'ARV par rapport à l'apprentissage sur l'echantillon 
        %de test 
        calc_ARV(BestNet, DTestInput, DTestOutput, dataSunspots);
    
        clearvars Net minErr errVerif errApp errVal BestNet DAppInput DAppOutput DValInput DValOutput DTestInput DTestOutput Sortie inExp ;
        
        %{
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Ordre 2 x(t-4), x(t-5), x(t-6), x(t-7), x(t-9), x(t-10)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %}
        
        %Charger le premier réseau 
        nom_fichier = strcat('reseaux/','Net1_', nom_exp, '.mat');
        load(nom_fichier,'Net');

        %réinitialiser le dataset 
        %Data input
        inExp(:,1) = dataSunspots(1:268);
        inExp(:,2) = dataSunspots(2:269);
        inExp(:,3) = dataSunspots(3:270);
        inExp(:,4) = dataSunspots(4:271);
        inExp(:,5) = dataSunspots(5:272);
        inExp(:,6) = dataSunspots(6:273);
        inExp(:,7) = dataSunspots(7:274);
        inExp(:,8) = dataSunspots(8:275);
        inExp(:,9) = dataSunspots(9:276);
        inExp(:,10) = dataSunspots(10:277);
        inExp(:,11) = dataSunspots(11:278);
        inExp(:,12) = dataSunspots(12:279);

        Sortie = dataSunspots(13:280);
        
        %Supprimer les 6 variables une à une et entraîner le réseau à
        %Chaque fois (on utilise une boucle)
        %Explication : si on supprime la colonne 4 alors la 5 eme colonne
        %(colonne suivante à supprimer)
        %A pour index 4 etc 
        colSpr = [4, 4, 4, 4, 5, 5];
        
        %inExp(:,colSpr(2))= [];
        %Net.w1(colSpr(6),:) = [];
        nom_fichier = strcat('Net4_', nom_exp, '.mat');
        
        for i = 1:size(colSpr,2)
            inExp(:,colSpr(i)) = [];
         
            % Apprentissage
            DAppInput = inExp(1:209,:);
            DAppOutput = Sortie(1:209);
            % Validation
            DValInput = inExp(210:244,:);
            DValOutput = Sortie(210:244);
            % Test
            DTestInput = inExp(245:268,:); 
            DTestOutput = Sortie(245:268); 
       
            %Réseaux : supprimer les neurones en entrée un à un 
            Net.nin = Net.nin - 1;
            % Ajuster nwts 
            Net.nwts = Net.nwts - (Net.nhidden);
            %Supprimer les poids qui partent des variables supprimées
            Net.w1(colSpr(i),:) = [];

            %entraîner à nouveau ce réseau après avoir pris en compte la
            %suppression des 6 variables 
            %nb : pas de sortie graph (graph à 0)
            [BestNet] = apprendre(Net.nin, Net.nhidden, Net.nout, DAppInput, DAppOutput, DValInput, DValOutput, nom_fichier, 0);
        end 
        
        fprintf('\n\n------------------ Experience 5 ------------ %d neurones ------------------', nb_n_hid); 
         
        %Calcul des ARV par rapport à l'apprentissage sur l'echantillon 
        %d'apprentissage
        calc_ARV(BestNet, DAppInput, DAppOutput, dataSunspots);

        %Calcul de l'ARV par rapport à l'apprentissage sur l'echantillon de
        %validation
        calc_ARV(BestNet, DValInput, DValOutput, dataSunspots);

        %Calcul de l'ARV par rapport à l'apprentissage sur l'echantillon 
        %de test 
        calc_ARV(BestNet, DTestInput, DTestOutput, dataSunspots);
        
        clearvars Net minErr errVerif errApp errVal BestNet DAppInput DAppOutput DValInput DValOutput DTestInput DTestOutput Sortie inExp;     
    
        
%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Expérience 5 : 20 neurones cachés
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%}
    
    %{
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Expérience 5-1 : 12 variables input
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %}

    %on réinitialise certaines variables
    clearvars i minErr errVerif errApp errVal BestNet DAppInput DAppOutput DValInput DValOutput DTestInput DTestOutput Sortie inExp;
    
    %{
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Expérience 5-2 : 6 variables input
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %}   
   
    %{
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Expérience 5-3 : supprimer 6 variables de Net1 d'un coup
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %}
    
    %{
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Expérience 5-4 : supprimer 6 variables de Net1 progressivement
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %}
    
        %{
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Ordre 1 : x(t-10), x(t-9), x(t-7), x(t-6), x(t-5), x(t-4)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %}
       
        %{
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Ordre 2 : x(t-4), x(t-5), x(t-6), x(t-7), x(t-9), x(t-10)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %}
end
