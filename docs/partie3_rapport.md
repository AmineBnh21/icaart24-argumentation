# Partie 3 — Visualising Acceptability Degree Spaces

*Auteur : Mohamed Amine Benhamaid*

Code : `src/part3/` · Tests : `tests/test_part3_hcategorizer.py` · Résultats et figures : `results/part3/`

```bash
python -m pytest tests -q                                   # 19 tests
python -m src.part3.run_experiments                         # toutes les expériences (N = 100 000, ~30 min)
python -m src.part3.run_experiments --af "a,b,c : a>b, b>c, c>a"   # un framework (A, R) quelconque
python -m src.part3.run_experiments --quick                 # test rapide (N = 5 000)
```

---

## A. Sémantique weighted h-categorizer

### A.1 Définition et implémentation

Soit un framework pondéré (A, R, w) avec w : A → [0, 1]. Le degré d'acceptabilité est l'unique solution de

$$\mathrm{Deg}(a) = \frac{w(a)}{1 + \sum_{b \in \mathrm{Att}(a)} \mathrm{Deg}(b)}$$

calculée comme limite de la suite $f^0(a) = w(a)$, $f^{k+1}(a) = w(a) / (1 + \sum_{b \to a} f^k(b))$.
On arrête dès que $\max_a |f^{k+1}(a) - f^k(a)| < \varepsilon$ ($\varepsilon > 0$ est un paramètre ; `ValueError` si $\varepsilon \le 0$, `RuntimeError` si `max_iter` est atteint).

Implémentation (`src/part3/hcategorizer.py`) :

- `ArgumentationFramework(arguments, attacks)` stocke R sous forme de matrice $M$ ($M_{ba} = 1$ ssi $b$ attaque $a$). La somme des degrés des attaquants s'écrit alors simplement $x M$.
- `h_categorizer(af, weights, epsilon)` : un vecteur de poids (dict ou liste), renvoie un dict de degrés.
- `h_categorizer_batch(af, W, epsilon)` : version **vectorisée** sur une matrice $W$ de taille $N \times |A|$. Une itération vaut `X = W / (1 + X @ M)`. C'est ce qui rend l'échantillonnage de 100 000 vecteurs instantané (0,02 s à 0,4 s selon le framework).

### A.2 Tests sur de petits frameworks

Chaque test compare le résultat à une **forme close** (19 tests, tous verts) :

| Framework | Forme close vérifiée |
|---|---|
| a, b sans attaque | $\mathrm{Deg} = w$ |
| a → b | $\mathrm{Deg}(b) = w_b / (1+w_a)$ |
| a ↔ b | $x_a - x_b = w_a - w_b$ et $x_b(1+x_a) = w_b$ (équation du 2nd degré) ; pour $w = 1$ : $x = (\sqrt5-1)/2 \approx 0{,}618$ |
| a → a (auto-attaque) | $x = (-1+\sqrt{1+4w})/2$ |
| chaîne a→b→c, $w=1$ | $(1,\ 1/2,\ 2/3)$ |
| cycle à 3, $w = 1$ | nombre d'or $0{,}618$ pour chaque argument |
| étoile a,b → c | $\mathrm{Deg}(c) = w_c/(1+w_a+w_b)$ |
| 20 graphes aléatoires (6 arguments) | le résultat est bien un point fixe et $0 \le \mathrm{Deg} \le w$ |
| — | $w(a) = 0 \Rightarrow \mathrm{Deg}(a) = 0$ ; version batch = version unitaire ; parseur ; entrées invalides |

### A.3 Convergence et rôle de ε

![convergence](../results/part3/convergence.png)

| ε | 1e-2 | 1e-4 | 1e-6 | 1e-8 | 1e-10 | 1e-12 |
|---|---|---|---|---|---|---|
| a → b, chaîne à 5 (acycliques) | 2 / 5 | idem | idem | idem | idem | idem |
| a ↔ b, cycle à 3 | 5 | 10 | 15 | 20 | 24–25 | 29 |
| cycle à 6 | 5 | 10 | 14 | 19 | 24 | 28 |
| aléatoire 6 arg. / 10 att. | 6 | 12 | 17 | 23 | 29 | 34 |
| complet à 4 | 9 | 16 | 24 | 32 | 40 | 48 |

*(nombre d'itérations pour 20 000 vecteurs de poids traités ensemble.)*

Observations :

1. **Frameworks acycliques : convergence exacte en un nombre fini d'étapes.** Il en faut (profondeur du graphe + 1) : 2 pour a → b, 5 pour une chaîne de 5. En effet $f^k(a)$ est définitif dès que tous les ancêtres de $a$ le sont. L'erreur tombe alors à 0 en arithmétique flottante, quel que soit ε.
2. **Frameworks avec cycles : convergence géométrique (linéaire).** La courbe d'erreur est une droite en échelle log. Pour les cycles, on gagne 2 décimales toutes les 5 itérations, soit un taux d'environ $0{,}4 \approx \varphi^{-2} = 0{,}382$. C'est la dérivée de $x \mapsto 1/(1+x)$ au point fixe $\varphi^{-1}$, pour des poids égaux à 1 (le pire cas). Plus un argument a d'attaquants, plus la contraction est lente : environ 0,56 pour le graphe complet à 4.
3. **ε contrôle bien la précision.** L'erreur finale mesurée par rapport à une référence calculée avec ε = 1e-15 est toujours **inférieure à ε** (≈ 0,2 ε à 0,3 ε, cf. `convergence.json`). Le coût en itérations est logarithmique en 1/ε. Nous utilisons ε = 1e-10 pour la partie B.

---

## B. Espace des degrés d'acceptabilité

### B.1 Protocole

Pour un framework (A, R) donné (catalogue de 17 frameworks, ou `--af` en ligne de commande) :

1. tirer **N = 100 000** vecteurs $w_i \sim \mathcal U([0,1]^{|A|})$ (graine fixée) ;
2. calculer $x_i = \mathrm{HCat}(w_i)$ avec la version vectorisée ;
3. calculer l'**enveloppe convexe** des $x_i$ avec Qhull (`scipy.spatial.ConvexHull`), en dimension pleine jusqu'à |A| = 6 ;
4. construire les **heat-maps** : histogrammes 2D de 40 × 40 cases, en densité par unité de surface. En dimension 2, c'est le plan entier. En dimension 3 on ajoute un nuage 3D coloré par la densité et l'enveloppe 3D. Pour |A| ≥ 4 on trace la matrice des projections 2D, où la projection de l'enveloppe n-D est l'enveloppe 2D des points projetés ;
5. **résilience** : 10 ré-échantillonnages indépendants, puis comparaison des heat-maps et des volumes d'enveloppe ;
6. faire varier la structure du framework (sans attaque, attaque simple, attaque mutuelle, auto-attaque, chaîne, cycle, étoile, graphe complet, graphes aléatoires) et sa taille (|A| = 2 à 6).

### B.2 Résultat théorique : description exacte de l'espace S

En réécrivant l'équation du point fixe, on obtient $w_a = x_a\,(1 + \sum_{b\to a} x_b)$. L'application $w \mapsto x$ s'inverse donc **explicitement** par $g(x) = x \odot (1 + xM)$. Comme le point fixe est unique, on en déduit :

> **Proposition.** $S = \mathrm{HCat}([0,1]^{|A|}) = \{\, x \ge 0 \;:\; x_a (1 + \textstyle\sum_{b \to a} x_b) \le 1 \;\; \forall a \in A \,\}$.
>
> Avec $w$ uniforme, la densité des points $x_i$ est $p(x) = |\det J_g(x)|$ sur S, où $J_g(x)_{ac} = \delta_{ac}(1 + s_a) + x_a M_{ca}$.
>
> On en déduit $\mathrm{vol}(S) = \mathbb E_w\big[1/|\det J_g(\mathrm{HCat}(w))|\big]$, qu'on estime sans biais avec les mêmes échantillons.

Cela nous donne une **vérité terrain** pour juger l'enveloppe convexe (volume exact de S) et les heat-maps (densité exacte). Les tests vérifient que $g$ redonne exactement les poids, à 1e-10 près. Exemples :

- **a → b** : $S = \{x_b \le 1/(1+x_a)\}$, $p(x) = 1 + x_a$, $\mathrm{vol}(S) = \ln 2 = 0{,}6931$ (mesuré : $0{,}6936 \pm 0{,}0004$).
- **chaîne à 3 et étoile à 3** ont *exactement* le même volume, $3\ln 3 - 4 \ln 2 = 0{,}5232$ (mesuré : 0,5235 pour les deux), alors que leurs formes sont différentes.
- Pour un graphe acyclique, $J_g$ est triangulaire, donc $p(x) = \prod_a (1 + s_a) \ge 1$. Empiriquement on observe aussi $p \ge 1$ sur tous nos graphes cycliques (colonne `density_min` ≥ 1).

### B.3 Résultats globaux (N = 100 000, 10 ré-échantillonnages)

| AF | \|A\| | \|R\| | iter | vol(hull) | vol(S) exact | vide dans l'enveloppe | densité max | TV ré-échant. | Jaccard HDR50 | CV vol(hull) |
|---|---|---|---|---|---|---|---|---|---|---|
| isolated2 | 2 | 0 | 1 | 0.9997 | 1.0000 | 0.0% | 1.00 | 0.071 | (0.29)\* | 0.006 % |
| attack2 (a→b) | 2 | 1 | 2 | 0.7473 | 0.6936 | 7.3% | 2.00 | 0.059 | 0.66 | 0.13 % |
| mutual2 (a↔b) | 2 | 2 | 25 | 0.6162 | 0.5808 | 5.9% | 2.23 | 0.055 | 0.56 | 0.09 % |
| attack2_self | 2 | 2 | 25 | 0.5153 | 0.5057 | 2.0% | 2.82 | 0.051 | 0.63 | 0.05 % |
| chain3 | 3 | 2 | 3 | 0.5878 | 0.5235 | 11.3% | 3.00 | 0.062 | 0.57 | 0.16 % |
| cycle3 | 3 | 3 | 24 | 0.4720 | 0.4083 | 13.7% | 4.40 | 0.058 | 0.64 | 0.23 % |
| star3 | 3 | 2 | 2 | 0.5993 | 0.5235 | 12.9% | 3.00 | 0.061 | 0.55 | 0.49 % |
| complete3 | 3 | 6 | 34 | 0.3029 | 0.2684 | 11.4% | 6.64 | 0.052 | 0.49 | 0.27 % |
| chain4 | 4 | 3 | 4 | 0.4498 | 0.3882 | 14.4% | 4.96 | 0.062 | 0.56 | 0.19 % |
| cycle4 | 4 | 4 | 24 | 0.3669 | 0.3072 | 17.0% | 6.52 | 0.060 | 0.61 | 0.31 % |
| complete4 | 4 | 12 | 40 | 0.1238 | 0.1073 | 14.1% | 22.7 | 0.047 | 0.47 | 0.48 % |
| random4 | 4 | 3 | 25 | 0.4514 | 0.4024 | 11.5% | 4.43 | 0.063 | 0.57 | 0.14 % |
| chain5 | 5 | 4 | 5 | 0.3414 | 0.2890 | 16.9% | 7.72 | 0.063 | 0.55 | 0.24 % |
| cycle5 | 5 | 5 | 24 | 0.2761 | 0.2279 | 19.0% | 10.4 | 0.061 | 0.61 | 0.27 % |
| random5 | 5 | 8 | 25 | 0.1645 | 0.1372 | 18.4% | 19.7 | 0.056 | 0.59 | 0.76 % |
| cycle6 | 6 | 6 | 24 | 0.2031 | 0.1697 | 19.7% | 16.4 | 0.062 | 0.60 | 0.35 % |
| random6 | 6 | 10 | 29 | 0.1075 | 0.0880 | 21.7% | 41.4 | 0.055 | 0.59 | 0.68 % |

*« vide dans l'enveloppe » = part du volume de l'enveloppe qui n'appartient pas à S, estimée par Monte-Carlo (200 000 points testés avec la Proposition).*
*TV : distance en variation totale moyenne entre les heat-maps 40×40 de deux ré-échantillonnages (moyenne sur toutes les paires de projections).*
*HDR50 : ensemble minimal de cases qui contient 50 % de la masse, c'est-à-dire la « zone de forte densité ».*
*\* Pour une densité uniforme la HDR50 n'est pas définie de façon unique, donc ce Jaccard n'a pas de sens.*

### B.4 Analyse : l'enveloppe convexe

![a → b](../results/part3/space_attack2.png)

1. **L'espace S n'est pas convexe dès qu'il y a une attaque.** Les contraintes $x_a + x_a x_b \le 1$ sont bilinéaires. Pour a → b, la frontière $x_b = 1/(1+x_a)$ est une fonction convexe, et la région située *sous* elle n'est pas convexe. L'enveloppe est donc ici le trapèze $(0,0),(1,0),(1,\tfrac12),(0,1)$, d'aire 3/4, contre ln 2 pour S : **7,6 % de volume vide** en théorie, 7,3 % mesuré. L'enveloppe convexe **surestime** systématiquement l'ensemble des degrés atteignables.
2. **Ce volume vide augmente avec la dimension.** Il vaut 0 % sans attaque (S = cube), puis 2 à 7 % pour |A| = 2, 11 à 14 % pour |A| = 3, 11 à 17 % pour |A| = 4, 17 à 19 % pour |A| = 5 et 20 à 22 % pour |A| = 6. Chaque argument attaqué ajoute une face « creuse ». Cas particulier : l'auto-attaque seule donne un intervalle $[0, 0{,}618]$, convexe, d'où seulement 2 % de vide pour `attack2_self`.
3. **Le volume de S chute vite avec le nombre d'attaques.** Il vaut 1 sans attaque, 0,69 avec une attaque, 0,27 pour le graphe complet à 3, 0,11 pour le complet à 4, 0,09 pour random6. Les attaques **compriment** l'espace des degrés atteignables.
4. **Le coût explose avec la dimension.** Le nombre de sommets de l'enveloppe passe de ~20 (2D) à ~120 (3D), ~500 (4D), ~1 400 (5D) puis ~3 200 (6D). Le temps de Qhull passe de < 0,1 s à ~8 s en 6D, et l'estimation Monte-Carlo du vide devient elle aussi coûteuse (≈ 12 min par framework en 6D). Au-delà de 6 arguments, l'enveloppe en dimension pleine n'est plus praticable. Les **projections 2D** (enveloppe des points projetés = projection de l'enveloppe) restent la seule visualisation exploitable.
5. **Les sommets de l'enveloppe sont des points rares.** Ce sont les « coins » du type « un argument à 1, ses attaquants à 0 », qui ont une densité faible. Avec peu d'échantillons, l'enveloppe empirique est donc *plus petite* que l'enveloppe de S. Au test rapide (N = 5 000), on trouvait même vol(hull) < vol(S) en dimension 5 et 6. À N = 100 000, l'enveloppe pour a → b vaut 0,7473 contre 0,75 en théorie.

### B.5 Analyse : les heat-maps (zones de forte et faible densité)

![cycle à 3](../results/part3/space_cycle3.png)
![complet à 4](../results/part3/space_complete4.png)

1. **La densité est forte là où l'attaquant est fort.** Pour a → b, $p = 1 + x_a$ : la heat-map empirique reproduit le dégradé exact de 1 (à gauche) à 2 (à droite). Quand $\mathrm{Deg}(a)$ est grand, tous les poids $w_b \in [0,1]$ sont écrasés dans l'intervalle $[0, 1/(1+x_a)]$, qui est plus court, d'où une densité plus élevée. Les zones de forte densité sont donc les **régions comprimées par les attaques**. Les zones de faible densité sont les configurations où les arguments attaqués gardent un degré élevé, ce qui impose des attaquants faibles.
2. **Cycles et graphes complets : un « plateau » suivi d'une chute brutale.** Pour le cycle à 3, la densité de la projection $(a, b)$ est maximale autour de $\mathrm{Deg}(a) \approx 0{,}5$–$0{,}6$, puis s'effondre au-delà de 0,62 environ. Pour le graphe complet à 4, la masse se concentre dans le carré $[0, 0{,}6]^2$ et les marginales sont presque uniformes sur $[0, 0{,}55]$ puis décroissent rapidement. Les degrés supérieurs à 0,62 (le nombre d'or) ne s'atteignent que si les adversaires sont faibles : l'enveloppe les englobe mais ils sont quasiment vides. C'est l'illustration la plus nette de la différence entre « atteignable » (enveloppe) et « typique » (heat-map).
3. **Les degrés moyens suivent la topologie.** Un argument non attaqué a un degré moyen de 0,5. Un argument attaqué par un argument non attaqué a un degré moyen de $\tfrac12 \ln 2 = 0{,}347$, ce qui est observé (0,3465). Dans une chaîne, les degrés moyens oscillent ensuite autour de ≈ 0,37–0,38 (0,345 ; 0,381 ; 0,373 ; 0,375). Ce comportement alterné se retrouve dans les cycles, tous à ≈ 0,373. Dans les graphes complets, la moyenne baisse avec le nombre d'attaquants (0,32 pour 3 arguments, 0,28 pour 4).
4. **Les attaques mutuelles créent une corrélation négative.** Dans complete4, ρ = −0,21 entre chaque paire : un argument fort affaiblit son adversaire. Dans une chaîne, les arguments non reliés par une attaque restent quasi indépendants.
5. **La densité maximale augmente fortement avec |R|.** Elle vaut 2 pour une attaque, 6,6 pour complete3, 22,7 pour complete4 et 41 pour random6. La masse se concentre donc dans une partie de plus en plus petite d'un espace lui-même de plus en plus petit.

### B.6 Résilience de la heat-map au ré-échantillonnage

![ré-échantillonnage](../results/part3/resampling_mutual2.png)
![TV vs N](../results/part3/tv_vs_n.png)

1. **La structure macroscopique est stable.** Deux tirages indépendants de 100 000 points donnent des heat-maps visuellement identiques. La carte de différence ne montre que du bruit blanc, sans structure spatiale. La TV moyenne entre ré-échantillonnages reste entre 0,047 et 0,071 pour tous les frameworks.
2. **L'erreur décroît exactement en $1/\sqrt N$.** Pour les frameworks à 2 arguments, on compare à la densité *exacte* intégrée par case. Ailleurs, on compare à un échantillon de référence de 2 millions de points. La TV passe de ≈ 0,4 (N = 1 000) à ≈ 0,04 (N = 100 000) puis ≈ 0,024 (N = 300 000), avec une pente de −1/2 en log-log, comme prévu pour un bruit multinomial.
3. **Le volume de l'enveloppe est très stable.** Son coefficient de variation sur 10 tirages va de 0,05 % à 0,8 %. Il est un peu plus élevé en dimension plus grande, car les sommets sont des points rares (voir B.4.5).
4. **Le détail au niveau d'une case reste bruité.** Avec 40 × 40 cases et N = 100 000, une case dense contient environ 60 à 150 points, soit un bruit relatif de 8 à 13 %. L'écart relatif maximal entre deux tirages, sur l'ensemble des 1 600 cases, atteint 60 à 90 %. La zone de forte densité HDR50 ne se recouvre donc qu'à 47–66 % (Jaccard) d'un tirage à l'autre. La cause est que les sommets de densité sont des **plateaux** : beaucoup de cases ont presque la même densité, et le bruit change leur classement. Ce sont les *formes* des zones qui sont robustes, pas leurs contours case par case.
5. **Recommandations.** Avec N = 100 000, une grille de 40 × 40 suffit pour lire les zones de forte et faible densité. Pour des contours fins, il faut soit N ≥ 300 000, soit des cases plus grosses (l'erreur varie comme $\sqrt{\text{nb de cases}/N}$), soit un lissage (KDE). On peut aussi utiliser directement la **densité exacte** $|\det J_g|$ de la Proposition, qui ne dépend d'aucun tirage.

### B.7 Conclusion

- Le weighted h-categorizer s'implémente efficacement de façon vectorisée. Il converge en un nombre fini d'étapes sur les graphes acycliques et géométriquement (taux ≈ 0,38–0,56) sur les graphes cycliques. ε contrôle directement l'erreur finale.
- L'espace des degrés atteignables admet une **description exacte** : $x_a(1 + \sum_{b\to a} x_b) \le 1$. Sa densité vaut $|\det J_g|$, ce qui permet de valider les visualisations.
- L'**enveloppe convexe** donne une bonne idée de l'étendue de S, mais elle la surestime de 7 % (2 arguments) à 22 % (6 arguments) parce que S est non convexe. Son coût la limite à 6 arguments environ.
- La **heat-map** apporte une information que l'enveloppe ne donne pas : la masse se concentre là où les attaques compriment les degrés, et une grande partie de l'enveloppe (degrés > 0,62 dans les cycles) est presque vide. Cette information est robuste au ré-échantillonnage à l'échelle macroscopique, avec une erreur en $1/\sqrt N$ ≈ 4 % à N = 100 000.

Toutes les figures sont dans `results/part3/` : `space_<framework>.png` pour chacun des 17 frameworks, `resampling_*.png`, `tv_vs_n.png`, `convergence.png`. Les chiffres sont dans `summary.json` et `summary.md`.
