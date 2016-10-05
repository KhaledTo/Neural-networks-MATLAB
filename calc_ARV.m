function calc_ARV(BestNet, DIn, DOut, SOut)
	ARV = 0;
    %Calcul de l'ARV par rapport à l'apprentissage sur l'echantillon
    %d'apprentissage
    AppOutCalc = mlpfwd(BestNet, DIn);
    numerateur =  ((sum((DOut - AppOutCalc) .^2)) / size(DOut,1));
    %variance naturelle de la variable à prédire
    denominateur = ((sum((SOut - mean(SOut)) .^2)) / size(SOut,1));
    ARV = numerateur / denominateur ;
    fprintf('\n ARV : %g', ARV);
end
