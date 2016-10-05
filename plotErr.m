function plotErr(errlogApp, errlogVal)
	figure;   
    set(gca,'YScale','log');
	hold on;
        plot(errlogApp,'g-'); % Variation de l’erreur durant l’apprentissage
        plot(errlogVal,'r-'); % Variation de l’erreur en test
    drawnow;
    grid on;
    legend('erreur d''apprentissage','erreur en test');
    title('Evolution de l''erreur en test et de l''erreur en apprentissage');
end
