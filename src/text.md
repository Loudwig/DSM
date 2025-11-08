

Mettre des log sigma au lieu de sigma dans le réseau pour avoir une plage plus petites 
passer en niveau de bruit pas échelles linéaires pour avoir plus de niveau de bruit "faible"

affichage avec des rescales différents pour 


entrainement plus favorisé pour certaine échelles ? 
-> vérfier la loss pour chaque échelle


Ça marche pas très bien avec beacoup de bruit (10 bruits différents de 0.01 à 5)

métrique : visuel + on regarde l'erreur relative : MSE / E[target] avec E[target] = x_dim/sigma**2 


epsiolon loss : pour Être indépendant de sigma.

démarer avec sigma min haut et initialiser avec plus à chaque fois