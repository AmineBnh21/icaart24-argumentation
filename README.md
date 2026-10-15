# icaart24-argumentation — Partie 3 : Visualising Acceptability Degree Spaces

Ce dépôt contient ma partie 3 du projet : l'implémentation de la sémantique **weighted h-categorizer**
et l'étude de l'**espace des degrés d'acceptabilité** qu'elle produit (enveloppe convexe, cartes de
densité, robustesse au ré-échantillonnage).

📄 Rapport : [version courte (5 pages)](docs/partie3/rapport_partie3.pdf) · [version longue](docs/partie3/rapport-version-longue.pdf)

---

## Sommaire

1. [La sémantique weighted h-categorizer](#1-la-sémantique-weighted-h-categorizer)
2. [Organisation du code](#2-organisation-du-code)
3. [Lancer le code](#3-lancer-le-code)
4. [Convergence et rôle de ε](#4-convergence-et-rôle-de-ε)
5. [L'espace des degrés : description exacte](#5-lespace-des-degrés--description-exacte)
6. [Résultats globaux](#6-résultats-globaux)
7. [Enveloppe convexe](#7-enveloppe-convexe)
8. [Cartes de densité](#8-cartes-de-densité)
9. [Résilience au ré-échantillonnage](#9-résilience-au-ré-échantillonnage)
10. [Conclusion](#10-conclusion)

---

## 1. La sémantique weighted h-categorizer

Un framework d'argumentation pondéré est un triplet $(A, R, w)$ : $A$ est un ensemble fini d'arguments,
$R \subseteq A \times A$ la relation d'attaque et $w : A \to [0,1]$ le poids initial de chaque argument.
La sémantique weighted h-categorizer (Amgoud *et al.*, IJCAI 2017) donne à chaque argument le degré

$$
\mathrm{Deg}(a) = \frac{w(a)}{1 + \sum_{b \in \mathrm{Att}(a)} \mathrm{Deg}(b)}
$$

Ce degré est l'unique point fixe de ce système. On le calcule par itération, à partir de $f^0 = w$ :

$$
f^{k+1}(a) = \frac{w(a)}{1 + \sum_{b \in \mathrm{Att}(a)} f^k(b)}
$$

On s'arrête dès que $\max_a |f^{k+1}(a) - f^k(a)| < \varepsilon$ (avec $\varepsilon > 0$ en paramètre).

## 2. Organisation du code

```
src/part3/
├── hcategorizer.py     # ArgumentationFramework + h_categorizer / h_categorizer_batch
├── frameworks.py       # catalogue (chaîne, cycle, étoile, complet, aléatoire) + parseur "a,b : a>b"
├── space.py            # échantillonnage, densité exacte, enveloppe convexe, heat-maps, métriques
└── run_experiments.py  # lance toutes les expériences et produit les figures
tests/test_part3_hcategorizer.py   # 19 tests unitaires
results/part3/                     # figures (.png) + tableau récapitulatif (summary.md)
docs/partie3/                      # rapports PDF
```

**Choix d'implémentation.** La relation $R$ est stockée dans une matrice $M \in \{0,1\}^{n\times n}$
($M_{ba} = 1$ si $b$ attaque $a$). La somme des degrés des attaquants s'écrit alors $xM$. On peut donc
itérer **en même temps** sur $N$ vecteurs de poids (les lignes d'une matrice $W$) :
`X ← W / (1 + X @ M)`. Le calcul des 100 000 vecteurs prend entre 0,02 s et 0,4 s selon le framework.

```python
from src.part3.frameworks import parse
from src.part3.hcategorizer import h_categorizer

af = parse("a,b,c : a>b, b>c, c>a")                     # cycle à 3
h_categorizer(af, {"a": 1, "b": 1, "c": 1}, epsilon=1e-10)
# -> 0.6180339887... pour a, b et c                    (1/φ, nombre d'or)
```

## 3. Lancer le code

Dépendances : `numpy`, `scipy`, `matplotlib`, `pytest`.

```bash
pip install numpy scipy matplotlib pytest

python -m pytest tests -q                                   # 19 tests
python -m src.part3.run_experiments                         # toutes les expériences (N = 100 000)
python -m src.part3.run_experiments --quick                 # test rapide (petit N)
python -m src.part3.run_experiments --af "a,b,c : a>b, b>c, c>a"   # framework quelconque
```

Options : `--n-samples` (100 000 par défaut), `--repeats` (10 ré-échantillonnages), `--epsilon` (1e-10).
Les figures et le tableau sont écrits dans `results/part3/`.

**Tests.** Ils comparent le résultat à une forme close sur de petits frameworks : sans attaque ($\mathrm{Deg} = w$),
$a \to b$ ($\mathrm{Deg}(b) = w_b/(1+w_a)$), attaque mutuelle, auto-attaque ($x = (-1+\sqrt{1+4w})/2$),
chaîne, cycle à 3 ($1/\varphi \approx 0{,}618$) et étoile. Ils vérifient aussi, sur 20 graphes aléatoires,
que le résultat est bien un point fixe avec $0 \le \mathrm{Deg} \le w$, et que $\varepsilon \le 0$ et les
poids hors de $[0,1]$ sont rejetés.

## 4. Convergence et rôle de ε

![Convergence](results/part3/convergence.png)

*À gauche : distance au point fixe selon l'itération (20 000 vecteurs de poids, échelle log).
À droite : nombre d'itérations nécessaires selon ε.*

| ε | 1e-2 | 1e-4 | 1e-6 | 1e-8 | 1e-10 | 1e-12 |
|---|---|---|---|---|---|---|
| a→b / chaîne à 5 (acycliques) | 2 / 5 | 2 / 5 | 2 / 5 | 2 / 5 | 2 / 5 | 2 / 5 |
| a↔b, cycle à 3 | 5 | 10 | 15 | 20 | 24–25 | 29 |
| cycle à 6 | 5 | 10 | 14 | 19 | 24 | 28 |
| aléatoire (6 arg., 10 att.) | 6 | 12 | 17 | 23 | 29 | 34 |
| complet à 4 | 9 | 16 | 24 | 32 | 40 | 48 |

- **Graphes acycliques** : la convergence est exacte en (profondeur + 1) itérations, quel que soit ε.
- **Graphes cycliques** : la convergence est géométrique, avec un taux d'environ $\varphi^{-2} \approx 0{,}38$
  pour les cycles (deux décimales gagnées toutes les 5 itérations) et d'environ 0,56 pour le graphe complet à 4.
- **ε contrôle la précision** : l'erreur finale vaut entre 0,2 ε et 0,3 ε, pour un coût logarithmique en 1/ε.
  On prend ε = 1e-10 dans la suite.

## 5. L'espace des degrés : description exacte

On tire $N = 100\,000$ vecteurs de poids $w \sim \mathcal U([0,1]^{|A|})$ et on étudie l'ensemble
$S = \mathrm{HCat}([0,1]^{|A|})$ des degrés obtenus. L'équation du point fixe s'inverse explicitement :
$w_a = x_a(1 + \sum_{b\to a} x_b)$. On en déduit :

> **Proposition.**
> $S = \{ x \ge 0 : x_a (1 + \sum_{b \in \mathrm{Att}(a)} x_b) \le 1 \ \ \forall a \}$.
> La densité des points échantillonnés est $p(x) = |\det J_g(x)|$, avec $g(x) = x \odot (1 + xM)$.

On dispose ainsi d'une **vérité terrain** pour juger les visualisations. Par exemple, pour $a \to b$ :
$p(x) = 1 + x_a$ et $\mathrm{vol}(S) = \ln 2 = 0{,}6931$ (valeur mesurée : 0,6936).

## 6. Résultats globaux

17 frameworks de 2 à 6 arguments, N = 100 000, 10 ré-échantillonnages.

| Framework | \|A\| | \|R\| | iter | vol(hull) | vol(S) | vide | dens. max | TV | Jaccard HDR50 | CV vol |
|---|---|---|---|---|---|---|---|---|---|---|
| isolated2 | 2 | 0 | 1 | 0.9997 | 1.0000 | 0.0 % | 1.00 | 0.071 | (0.29)* | 0.006 % |
| attack2 (a→b) | 2 | 1 | 2 | 0.7473 | 0.6936 | 7.3 % | 2.00 | 0.059 | 0.66 | 0.13 % |
| mutual2 (a↔b) | 2 | 2 | 25 | 0.6162 | 0.5808 | 5.9 % | 2.23 | 0.055 | 0.56 | 0.09 % |
| attack2_self | 2 | 2 | 25 | 0.5153 | 0.5057 | 2.0 % | 2.82 | 0.051 | 0.63 | 0.05 % |
| chain3 | 3 | 2 | 3 | 0.5878 | 0.5235 | 11.3 % | 3.00 | 0.062 | 0.57 | 0.16 % |
| cycle3 | 3 | 3 | 24 | 0.4720 | 0.4083 | 13.7 % | 4.40 | 0.058 | 0.64 | 0.23 % |
| star3 | 3 | 2 | 2 | 0.5993 | 0.5235 | 12.9 % | 3.00 | 0.061 | 0.55 | 0.49 % |
| complete3 | 3 | 6 | 34 | 0.3029 | 0.2684 | 11.4 % | 6.64 | 0.052 | 0.49 | 0.27 % |
| chain4 | 4 | 3 | 4 | 0.4498 | 0.3882 | 14.4 % | 4.96 | 0.062 | 0.56 | 0.19 % |
| cycle4 | 4 | 4 | 24 | 0.3669 | 0.3072 | 17.0 % | 6.52 | 0.060 | 0.61 | 0.31 % |
| complete4 | 4 | 12 | 40 | 0.1238 | 0.1073 | 14.1 % | 22.7 | 0.047 | 0.47 | 0.48 % |
| random4 | 4 | 3 | 25 | 0.4514 | 0.4024 | 11.5 % | 4.43 | 0.063 | 0.57 | 0.14 % |
| chain5 | 5 | 4 | 5 | 0.3414 | 0.2890 | 16.9 % | 7.72 | 0.063 | 0.55 | 0.24 % |
| cycle5 | 5 | 5 | 24 | 0.2761 | 0.2279 | 19.0 % | 10.4 | 0.061 | 0.61 | 0.27 % |
| random5 | 5 | 8 | 25 | 0.1645 | 0.1372 | 18.4 % | 19.7 | 0.056 | 0.59 | 0.76 % |
| cycle6 | 6 | 6 | 24 | 0.2031 | 0.1697 | 19.7 % | 16.4 | 0.062 | 0.60 | 0.35 % |
| random6 | 6 | 10 | 29 | 0.1075 | 0.0880 | 21.7 % | 41.4 | 0.055 | 0.59 | 0.68 % |

- **vide** : part du volume de l'enveloppe convexe qui n'appartient pas à $S$ (Monte-Carlo, 200 000 points).
- **TV** : distance en variation totale moyenne entre les heat-maps 40×40 de deux ré-échantillonnages.
- **HDR50** : plus petit ensemble de cases contenant 50 % de la masse (zone de forte densité).
- **CV vol** : coefficient de variation du volume de l'enveloppe entre ré-échantillonnages.
- \* Pour une densité uniforme, la HDR50 n'est pas définie de façon unique.

## 7. Enveloppe convexe

![a → b](results/part3/space_attack2.png)

*$a \to b$. À gauche : points, enveloppe convexe (rouge) et vraie frontière de $S$ (pointillés).
Au centre : densité empirique. À droite : densité exacte $1 + x_a$.*

- **$S$ n'est pas convexe dès qu'il y a une attaque.** Les contraintes $x_a + x_a x_b \le 1$ sont bilinéaires.
  Pour $a \to b$, l'enveloppe est un trapèze d'aire 3/4 alors que $\mathrm{vol}(S) = \ln 2$, soit **7,6 % de
  volume vide** en théorie (7,3 % mesuré). L'enveloppe **surestime** donc toujours l'espace atteignable.
- **Le volume vide augmente avec la dimension** : 0 % sans attaque, 2–7 % pour 2 arguments, 11–14 % pour 3,
  11–17 % pour 4, 17–19 % pour 5 et 20–22 % pour 6.
- **Les attaques compriment l'espace** : vol(S) vaut 1 sans attaque, 0,69 avec une attaque, 0,11 pour le
  graphe complet à 4 et 0,09 pour `random6`.
- **Le coût explose avec la dimension** : l'enveloppe a environ 20 sommets en 2D et environ 3 200 en 6D,
  et Qhull met jusqu'à environ 8 s en 6D. Au-delà de 6 arguments, seules les projections 2D restent utilisables.
- **Les sommets de l'enveloppe sont des points rares** (un argument à 1 et ses attaquants à 0). Avec peu
  d'échantillons, l'enveloppe empirique est donc trop petite.

## 8. Cartes de densité

![Cycle à 3](results/part3/space_cycle3.png)

*Cycle $a \to b \to c \to a$ : enveloppe 3D et points colorés par la densité exacte, puis densité des
3 projections 2D avec l'enveloppe projetée (cyan).*

<p align="center"><img src="results/part3/space_complete4.png" width="65%"></p>

*Graphe complet à 4 arguments. Sous la diagonale : densités projetées. Sur la diagonale : marginales.
Au-dessus : corrélations.*

- **La densité est forte là où l'attaquant est fort.** Pour $a \to b$, $p = 1 + x_a$ : quand $\mathrm{Deg}(a)$
  est grand, tous les $w_b \in [0,1]$ sont ramenés dans l'intervalle plus court $[0, 1/(1+x_a)]$.
- **Cycles et graphes complets : un plateau puis une chute brutale** au-delà de $1/\varphi \approx 0{,}62$.
  L'enveloppe englobe ces régions, mais elles sont presque vides. C'est la différence entre l'**atteignable**
  (l'enveloppe) et le **typique** (la heat-map).
- **Les degrés moyens suivent la topologie** : 0,5 pour un argument non attaqué, $\tfrac12 \ln 2 = 0{,}347$
  pour un argument attaqué par un argument non attaqué, environ 0,373 dans un cycle.
- **Les attaques mutuelles créent une corrélation négative** ($\rho = -0{,}21$ dans `complete4`).
- **La densité maximale augmente fortement avec |R|** : 2 pour une attaque, 22,7 pour `complete4`,
  41 pour `random6`.

## 9. Résilience au ré-échantillonnage

![Ré-échantillonnage](results/part3/resampling_mutual2.png)

*$a \leftrightarrow b$ : deux échantillons indépendants de 100 000 points et leur différence (bruit sans structure).*

<p align="center"><img src="results/part3/tv_vs_n.png" width="55%"></p>

*Distance TV à une référence selon N : la pente vaut −1/2.*

- **La structure macroscopique est stable** : la TV entre deux tirages reste entre 0,047 et 0,071.
- **L'erreur décroît en $1/\sqrt N$** : environ 0,4 pour N = 10³, 0,04 pour N = 10⁵ et 0,024 pour N = 3·10⁵.
- **Le volume de l'enveloppe est très stable** : son coefficient de variation va de 0,05 % à 0,8 %.
- **Le détail case par case reste bruité** : la zone HDR50 ne se recouvre qu'à 47–66 % d'un tirage à l'autre
  (Jaccard). Les *formes* des zones sont robustes, pas leurs contours exacts.
- **Recommandation** : une grille 40×40 avec N = 100 000 suffit pour lire les zones de forte et faible densité.
  Pour des contours fins, il faut augmenter N, prendre des cases plus grosses ou utiliser directement la
  densité exacte $|\det J_g|$.

## 10. Conclusion

- Le weighted h-categorizer s'implémente efficacement en version vectorisée. Il converge en un nombre fini
  d'étapes sur les graphes acycliques, et géométriquement sur les graphes cycliques.
- L'espace des degrés admet une description exacte, avec une densité $|\det J_g|$, ce qui permet de valider
  les visualisations.
- L'enveloppe convexe donne l'étendue de $S$, mais la surestime de 7 à 22 % et devient impraticable
  au-delà de 6 arguments.
- La heat-map montre où se trouve réellement la masse. Cette information est robuste au ré-échantillonnage,
  avec une erreur d'environ 4 % pour N = 100 000.

<details>
<summary><b>Figures des autres frameworks</b></summary>

**2 arguments** : sans attaque, attaque mutuelle, auto-attaque

![](results/part3/space_isolated2.png)
![](results/part3/space_mutual2.png)
![](results/part3/space_attack2_self.png)

**3 arguments** : chaîne, étoile, complet

![](results/part3/space_chain3.png)
![](results/part3/space_star3.png)
![](results/part3/space_complete3.png)

**4 arguments** : chaîne, cycle, aléatoire

![](results/part3/space_chain4.png)
![](results/part3/space_cycle4.png)
![](results/part3/space_random4_p0.4_s1.png)

**5 arguments** : chaîne, cycle, aléatoire

![](results/part3/space_chain5.png)
![](results/part3/space_cycle5.png)
![](results/part3/space_random5_p0.3_s2.png)

**6 arguments** : cycle, aléatoire

![](results/part3/space_cycle6.png)
![](results/part3/space_random6_p0.3_s3.png)

**Ré-échantillonnage** : a → b et cycle à 3

![](results/part3/resampling_attack2.png)
![](results/part3/resampling_cycle3.png)

</details>

---

**Référence** : L. Amgoud, J. Ben-Naim, D. Doder, S. Vesic. *Acceptability Semantics for Weighted
Argumentation Frameworks*. IJCAI 2017, p. 56–62.
