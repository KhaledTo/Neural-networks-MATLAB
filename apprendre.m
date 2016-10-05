function BestNet = apprendre(nb_n_in, nb_n_hid, nb_n_out, DAppInput, DAppOutput, DValInput, DValOutput, nom_fichier, graph)
    %pour essayer de 'régler' le problème de l'initialisation des poids au
    %début et donc de la convergence 
    rand('twister', 0); 
    
    Net = mlp(nb_n_in, nb_n_hid, nb_n_out, 'linear');
    options = foptions;
    %on desactive les sorties (affichage de l'erreur à une itération
    %donnée)
    options(1) = -1;
    %nous allons gérer les itérations afin de comparer l'erreur en test et
    %en validation
    options(14) = 1;
    % Scaled conjugate gradient 
    algorithm = 'scg';
    %une première itération (nous gérons nous mêmes les itérations à l'aide
    %d'une boucle
    [Net, options, errApp] = netopt(Net, options, DAppInput, DAppOutput, algorithm);
    %Pour une fonction de transfert linéaire l'erreur retournée par mlperr correspond
    %à l'erreur quadratique
    errVal = mlperr(Net, DValInput, DValOutput);
    %initialisation du meilleur réseau 
    BestNet = Net;
    errVerif = 0;
    
    for i = 1:1000
        [Net, options, errAppi] = netopt(Net, options, DAppInput, DAppOutput, algorithm);
        %On ajoute ces erreurs d'apprentissage à notre vecteur
        errApp = [errApp; errAppi];
        errVali = mlperr(Net, DValInput, DValOutput);
       %On ajoute ces erreurs en test à notre vecteur
        errVal = [errVal; errVali];
        minErr = min(errVal);
        %On garde le réseau pour lequel l'erreur en test est minimale
        if errVali == minErr 
            BestNet = Net; 
            %disp('Get best net');
            %verifier que cette erreur est bien égale à 
            %l'erreur minimale sur l'échantillon de validation
            errVerif = errVali;
        %sinon si l'erreur augmente : on sort de la boucle et on arrête l'apprentissage 
        %elseif errVali > minErr 
            %break;
        end
    end

    %[Net1Val, options errVal] = netopt(Net1App, options, DvalInput, DvalOutput, algorithm);
    % Sauvegarde du réseau sur le disque dur 
    save(strcat('reseaux/',nom_fichier),'Net');
 
    %Afficher erreur en test et apprentissage (si graph = oui)
    if graph == 1
        plotErr(errApp, errVal);
    end
    %minErr = min(errVal);
    %verifier qu'on a bien sauvegardé le bon réseau
    %disp(minErr);
    %disp(errVerif);
   
    BestNet = BestNet;
end 