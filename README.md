
J'ai initialement compilé ces notes pour mon propre usage, afin d'acceder rapidement aux concepts en cas de besoin. Les concepts ne sont pas exhaustivement détaillés car j'ai mis l'accent sur les procédures MATLAB. J'invite ceux qui voudraient plus de détails à regarder le livre numérique 
<a href="https://github.com/madeenamadou/econ-comp/blob/master/%5BMario_J._Miranda%2C_Paul_L._Fackler%5D_Applied_Comput.pdf">Applied Computational Economics and Finance</a> de Mario J. Miranda & Paul L. Fackler.

J'ajouterai des ressources et codes MATLAB sur le meme theme avec le temps, donc n'hesitez pas à <a href="https://github.com/madeenamadou/">suivre</a> ou <a href="https://github.com/madeenamadou/econ-comp/fork">copier</a> le <a href="https://github.com/madeenamadou/econ-comp/">répertoire GitHub</a>. Le contenu de cette page a été écrit en <strong>Markdown</strong>.

Bonne lecture !!

<hr>


<h2>Tables des matieres</h2>
...



## Système d’équations linéaire : Ax = b
### Méthodes directes

Décomposition LU, Cholesky `x = A\b`

Méthode itérative (forme Qx = b + (Q − A)x)

- Gauss-jacobi : `gjacobi (A,b)`
- Gauss-seidel : `gseidel (A,b)`

## Systèmes d’équations non linéaires : Points fixes (f(x) = x) & Solutions racines (f(x) = 0)
### Méthode bisection, sur un interval [a,b]

Pour une fonction f,
```Matlab
bisect ('f',a,b)
```

### Méthode Newton : un ou plusieurs variables, avec des valeurs initiales, utilise le jacobien

Pour une fonction f à 2 variables, avec des valeurs initiales respectives x1 et x2 
```Matlab
newton('f',[x1;x2])
```

### Méthode Quasi-Newton : utilise une approximation du jacobien
- Secant Method : une variable
	
	Broyden Method : plusieurs variables, utilise une valeur initiale pour la racine, et une autre pour le Jacobien
 	Pour une fonction f à deux variables, et pour les valeurs initiales x1et x2 des variables `broyden('f',[x1;x2])`
	
	>**Note :** Pour ces méthodes, on peut ajouter une backstepping routine, pour éviter les divergences

### Méthodes exclusives pour Point-fixes
Méthode Itération de fonction, pour une valeur initiale x0
Pour une fonction g, 
```Matlab
fixpoint('g',x0)
```

### Complementary Method : utilise le jacobien
Pour résoudre f(x) = 0, pour *8a < x < b* ;  a et b peuvent être Inf

- Méthode semismooth
	
	Pour une fonction f, un intervalle [a,b], et une valeur initiale x0, 
	```Matlab
	ncpsolve('f',a,b,x)
	```
 
- Méthode minmax
	
	Spécifier d’abord l'option 'type'
	```Matlab
	optset('ncpsolve','type','minmax')
	```

## Problèmes d’optimisation (recherche du maximum et du minimum)
### Méthodes sans dérivées :
- Method Golden Search : une variable, optimum local sur un interval [a,b]
	
	Pour une fonction f, sur un interval [a,b]
	```Matlab
	golden('f',a,b)
	```

- Méthode Algorithm Nelder-Mead : plusieurs variables, avec des valeurs initiales pour les variables
	
	Pour une fonction f à deux variables, avec les valeurs initiales x1 et x2,
	```Matlab
	neldmead('f',[x1;x2])
	```

## Méthode d’intégration et de différentiation
### Méthode d’intégration

- Calcul de l'aire

	**Méthodes Newton-cotes** : calcul de l’aire sous la fonction
	
	Trapezoid rule : pour les fonctions discontinues ayant des points d’inflexion
	
	Pour n trapezes, sur un intervalle [a,b], *n* les nodes et *w* les weights,
	```Matlab
	[x,w] = qnwtrap(n,a,b)
	```
	
	Simpson rule : pour les fonctions plus lisses
	```Matlab
	[x,w] = qnwsimp(n,a,b)
	```
	
	>Si *w(x)=1*, on calcule l’aire sous la fonction
	
	**Méthodes Gaussian quadrature**
	
	Legendre quadrature, pour w(x) = 1
	```Matlab
	[x,w] = qnwlege(n,a,b)
	```

- Calcul de l’espérance

	**Méthodes Gaussian quadrature**
	
	Pour x suivant une **loi normale (mu, var)**, *n* les nodes gaussiens et *w* les weights gaussiens,
	```Matlab
	[x,w] = qnwnorm(n,mu, var)
	```
	>Si w(x) = *fonction de densité de probabilité* de *x*, on calcule l’espérance de la fonction par `Somme(w*f(x))`

	**Méthodes Intégration Monte-Carlo**
	
	Il faut générer pseudoaléatoirement *n* nodes *x* d’après la distribution ; les weights *w=1/n* étant identiques
	
	L’espérance de *f* est obtenue par
	```Matlab
	Somme(w*f(x))
	```

	**Méthodes Quasi-Monte Carlo**
	
	Ici les *n* nodes *x* sont déterministes, sur une intervalle [a,b] ;  les weights *w=(b-a)/n* étant identiques
	
	L’espérance de *f* est obtenue par
	```Matlab
	Somme(w*f(x))
	```
 
### Méthode de différentiation

Pour une fonction *f*, une approximation *O(h²)* de sa dérivée, autour du point *x0*, est de la forme:
>*f'(x0) = a*f(x0) + b*f(x0 + h) + c*f(x0 + alpha*h) [ + O(h²)]*

>où (x0 + h), (x0 + alpha*h) sont deux autres points,
>choisir, alpha = 2, et choisir h quelconque, les paramètres a, b et c, s’obtiennent en résolvant le système suivant :

```
a + b + c = 0
b + cλ = 1/h
b + cλ2 = 0
```

## Initial Value Problem (IVP)
Prend la forme d’une equation différentielle à résoudre pour une function solution, connaissant les valeurs initiales pour la fonction.

>e.g. à l’ordre 1  *y'(t) = f(t,y(t))*

>ou à l’ordre 2 *y''(t) = f(t,y(t),y'(t))* 

Il faut réécrire parfois le problème sous la forme d’une équation différentielle. Une équation différentielle d’ordre 2 peut se ramener à un système d’équation différentielle d’ordre 1.

Le principe de résolution est de dériver une approximation de taylor de  à l’ordre 1 (Méthode de Euler), ou 2 (Méthode Runge-Kutta 2), ou 4 (Runge-Kutta 4). On choisit un pas h pour subdiviser la période de temps ; plus petit h, mieux c’est en général.

>Méthode de Euler : *y(t+h) = y(t) + y'(t)h*

>Méthode Runge-Kutta 2 : *y(t+h) = y(t) + h[a_1k_1 + a_2k_2]*

Pour une bonne approximation, il faut que :

![](pic/maths/ivp1.gif)

On choisit a_2 puis on trouve ensuite les autres paramètres. Il existe trois méthodes communes pour évaluer les paramètres inconnus, connus comme: méthode de Heun, la méthode du point milieu et la méthode de Ralston.

**Heun’s Method** : (a_2,a_1,p_1,q_1) = (1/2,1/2,1,1)

**Midpoint’s Method** : (a_2,a_1,p_1,q_1) = (1,0,1/2,1/2)

**Ralston’s Method** : (a_2,a_1,p_1,q_1) = (2/3,1/3,3/4,3/4)

**Paul L. Fackler & Mario J. Miranda** : (a_2,a_1,p_1,q_1) = (3/4,1/4,2/3,2/3)

**Méthode Runge-Kutta 4** : ![](pic/maths/ivp2.gif)

>Avec
>
>![](pic/maths/ivp3.gif)

Pour un système d’équation différentiel « system », une période allant de t0 à tf, des valeurs initiales stockées dans la matrice «inits», la résolution numérique avec Runge-Kutta 4 :

```Matlab
[t,y] = ode45('system',[t0 tf],inits)
[t,y] = rk4('system',(t0:tf)',inits)
```

>Ou, avec un pas «h», des valeurs initiales stockées dans la matrice «Y0», la période initiale «t0», pour un système d’équation différentiel «f», la résolution numérique MuPAD avec Runge-Kutta 4 :
```Matlab
sol1 := numeric::odesolve2(f, t0, Y0, RK4, Stepsize=h)
```

## Boundary Value Problem
Il s’agit d’un boundary value problem quand on ne connait pas les valeurs initiales, pour la fonction. Il est souvent donné une boundary condition, comme par exemple la valeur terminale pour la fonction y_T.

On cherche le guess y_0 pour la valeur initiale pour la fonction tel que :

![](pic/maths/bvp1.gif)

On réécrit ce problème sous la forme d’un problème de recherche de solution racine, à résoudre avec **broyden**. Quand on trouve on peut résoudre l’IVP.

e.g: ici, on doit chercher la solution racine pour la fonction «shooting» avec **broyden**.

On crée d'abord une fonction de «shooting»

>Function F = shooting (y0)
>
>[t,y] = ode45('system',[t0,tf],y0);
>
>F = y(end) - y_end;

```Matlab
y0 = broyden('shooting',guess);
```

## Interpolation

Il s’agit d’obtenir une approximation ![](pic/maths/int1.gif) aussi correcte que possible d’une fonction f.

Pour un ensemble de _n_ interpolation nodes (x_i,f(x_i)), on veut que : ![](pic/maths/int1.gif)

De façon générale, ![](pic/maths/int1.gif) est construit à partir d’une combinaisons linéaires de plusieurs fonctions connues _ϕ_. Pour _m_ degrés d’interpolation (_m_ fonctions ϕ_j_), on a :

![](pic/maths/int3.gif)

Les c_j sont inconnues et pour les obtenir, on résous simultanément les équations linéaires suivantes :

Pour _i = 1...n_,

![](pic/maths/int4.gif)

Les méthodes d’interpolations diffèrent dans le choix des fonctions ϕ. Les ϕ_j sont généralement des polynômes.

On peut des polynômes Chebychev.

Pour un degré d’interpolation _m_, un intervalle de nodes <img src="https://latex.codecogs.com/gif.latex?x_i&space;\epsilon&space;[a,b]" vertical-align="middle"/>, on obtient les <img src="https://latex.codecogs.com/gif.latex?\phi_j" vertical-align="middle"/>
```Matlab
phi = fundefn('cheb', m, a, b);
```

Ensuite, on obtient les <img src="https://latex.codecogs.com/gif.latex?c_j" vertical-align="middle"/> pour une interpolation de f

```Matlab
c = funfitf(fhat, f);
```

Puis on calcule ![](pic/maths/int1.gif), pour “x” le vecteur-colonne des nodes <img src="https://latex.codecogs.com/gif.latex?x_i" vertical-align="middle"/>
```Matlab
fhat = funeval(c,phi,x)
```

On peut des polynômes splines. Le cubic splines est l’interpolation à l’ordre 3 avec cette famille de polynômes. En utilisant cubic splines <img src="https://latex.codecogs.com/gif.latex?(k=3)" vertical-align="middle"/>,

```Matlab
phi = fundefn('spli', m, a, b, k); 
phi(nodes) = funbas(basis);
c = funfitf(fhat, f); 
//ou
c = phi(nodes) \ v ; 
fhat = funeval(c,phi,x);
```

## Méthode Collocation
Il s’agit d’une méthode pour la fonction f solution du problème du type :

<img src="https://latex.codecogs.com/gif.latex?g(x,f(x))=0" vertical-align="middle"/> sur un intervalle <img src="https://latex.codecogs.com/gif.latex?[a,b]" vertical-align="middle"/>. Comme un système d’équation différentielle à l’ordre zéro. Le principe est d’utiliser une interpolation ![](pic/maths/int1.gif) de f.

Pour un ensemble de _n_ collocation nodes <img src="https://latex.codecogs.com/gif.latex?x_i&space;\epsilon&space;[a,b]" vertical-align="middle"/> le problème revient à résoudre simultanément les équations linéaires suivantes, pour <img src="https://latex.codecogs.com/gif.latex?c_j" vertical-align="middle"/> :

Pour <img src="https://latex.codecogs.com/gif.latex?i=1,...,n" vertical-align="middle"/>

<img src="https://latex.codecogs.com/gif.latex?g(x_i,&space;\sum_{j=1}^{m}c_j\phi_j(x_i))&space;=&space;0" vertical-align="middle"/>

On peut obtenir les collocations nodes à partir de nodes chebychev…
```Matlab
phi = fundefn('cheb', m, a, b);
x = funnode(phi);
```

Ensuite, on peut résoudre ce problème en utilisant broyden.

Pour un système d’équation «system», 
```Matlab
c = broyden('system',guess);
```

## Programmation Dynamique

Sur _N_ periodes <img src="https://latex.codecogs.com/gif.latex?t=1,...,N" vertical-align="middle"/>. L’horizon _N_ peut être fini ou infini.

On peut observer à chaque période, plusieurs états  possibles. L’espace des états _s_ peut être discret ou continu.

<img src="https://latex.codecogs.com/gif.latex?S=\left&space;\{&space;s_1,s_2,s_3,s_4,...\right&space;\}" vertical-align="middle"/>

Pour chaque état, il y a un ensemble d’actions disponibles x. L’espace des actions peut aussi être discret ou continu.

<img src="https://latex.codecogs.com/gif.latex?X=\left&space;\{&space;x_1,x_2,x_3,x_4,...\right&space;\}" vertical-align="middle"/>

### Discrete state space et discrete action space
* Horizons finis

À chaque période t, l’action x dans l’état s produit le reward f(x,s). S est un processus, soit deterministe, soit stochastique.

Pour S stochastique, on suppose qu’il s’agit d’un markov decision process, donc, la probabilité d’un état <img src="https://latex.codecogs.com/gif.latex?s'=s_{t+1}" vertical-align="middle"/> dépend seulement de l’état s et de l'action x a la periode precedente t. Quand il n’y a qu’une seule action disponible pour chaque état possible, on parle de processus chaine de markov.

La probabilité de transition <img src="https://latex.codecogs.com/gif.latex?P(s'=s_{t+1}|s=s_t,x=x_t)" vertical-align="middle"/>

L’objectif est de trouver un ensemble optimal d’action <img src="https://latex.codecogs.com/gif.latex?X^*_t={x^*(s),&space;s\epsilon&space;S}" vertical-align="middle"/> poru chaque <img src="https://latex.codecogs.com/gif.latex?t=1,...,N" vertical-align="middle"/>

Soit <img src="https://latex.codecogs.com/gif.latex?V_t(s)" vertical-align="middle"/> le reward maximum que l’agent peut atteindre à partir de _t_ jusqu’à la fin de période _T_. Pour chaque état  possible à _t_, il y a une valeur <img src="https://latex.codecogs.com/gif.latex?V_t(s)" vertical-align="middle"/>

Pour <img src="https://latex.codecogs.com/gif.latex?s&space;\epsilon&space;S" vertical-align="middle"/>,

<img src="https://latex.codecogs.com/gif.latex?V_t(s)&space;=&space;max&space;\left&space;\{&space;f(s,x)&space;&plus;&space;\delta&space;\sum_{s^{'}&space;\epsilon&space;S}&space;Prob&space;(s_{t&plus;1}&space;=&space;s^{'}|s&space;=&space;s_t,&space;x=x_t)V_{t&plus;1}(s^{'})\right&space;\}" vertical-align="middle"/>

C'est <strong>l’équation de Bellman</strong>.

Ensuite, on cherche <img src="https://latex.codecogs.com/gif.latex?x&space;\epsilon&space;X" vertical-align="middle"/> tel que :

<img src="https://latex.codecogs.com/gif.latex?x^{*}_t(s)&space;=&space;argmax(x){V_t(s),s\epsilon&space;S}" vertical-align="middle"/>

On impose une condition terminale, généralement <img src="https://latex.codecogs.com/gif.latex?V_{t+1}(s)=0" vertical-align="middle"/>. Avec cette condition, pour (+1) on va backward, en cherchant <img src="https://latex.codecogs.com/gif.latex?V_t(s)" vertical-align="middle"/> puis <img src="https://latex.codecogs.com/gif.latex?V_{t-1}(s)" vertical-align="middle"/> et ainsi de suite jusqu’à <img src="https://latex.codecogs.com/gif.latex?V_1(s)" vertical-align="middle"/>.

* Horizons infinis

Pour les horizons infinis, <img src="https://latex.codecogs.com/gif.latex?V(s)" vertical-align="middle"/> ne dépend plus de _t_ et le problème s’écrit :

<img src="http://latex.codecogs.com/gif.latex?V(s)&space;=&space;max&space;\left&space;\{&space;f(s,x)&space;&plus;&space;\delta&space;\sum_{s'&space;\epsilon&space;S}Prob(s'|s,x)V(s')&space;\right&space;\}\\&space;\\&space;x^{*}(s)&space;=&space;argmax(x\left&space;\{&space;V(s),s&space;\epsilon&space;S&space;\right&space;\})" vertical-align="middle"/>

<img src="https://latex.codecogs.com/gif.latex?V_{t+1}(s)=0" vertical-align="middle"/>

Pour <img src="https://latex.codecogs.com/gif.latex?s&space;\epsilon&space;S" vertical-align="middle"/>,

<img src="https://latex.codecogs.com/gif.latex?V_T(s)&space;=&space;max&space;\left&space;\{&space;f(s,x)\right&space;\}&space;\\&space;V_{t-1}(s)&space;=&space;max&space;\left&space;\{&space;f(s,x)&space;&plus;&space;\delta&space;\sum_{s'\epsilon&space;S}&space;Prob&space;(s_T=s^{'}|s_{T-1}&space;=&space;s,&space;x_{T-1}=x)V_T(s^{'})\right&space;\}\\" vertical-align="middle"/>

Et ainsi de suite….

On peut utiliser la méthode backward recursion.

Soit _n_ le nombre fini d’états possibles, _m_ le nombre fini d’actions disponibles. À la période _t_, la matrice des probabilités de transition vers l’état <img src="https://latex.codecogs.com/gif.latex?s_i&space;\epsilon&space;S" vertical-align="middle"/>

<img src="https://latex.codecogs.com/gif.latex?P(s_i)=\begin{bmatrix}&space;Prob(s_i|s_1,x_1)&space;&&space;...&space;&&space;Prob(s_i|s_n,x_1)\\&space;Prob(s_i|s_1,x_2)&space;&&space;...&space;&&space;Prob(s_i|s_n,x_2)\\&space;.&space;&&space;...&space;&&space;.\\&space;.&space;&&space;...&space;&&space;.\\&space;Prob(s_i|s_1,x_m)&space;&&space;...&space;&&space;Prob(s_i|s_n,x_m)&space;\end{bmatrix}" vertical-align="middle"/>

Il y a _n_ possibilités pour l’état <img src="https://latex.codecogs.com/gif.latex?s_i" vertical-align="middle"/> suivant :

<img src="https://latex.codecogs.com/gif.latex?P(m(currentaction)&space;\times&space;n(currentstate)&space;\times&space;n(futurestate))&space;=&space;\left&space;\{P(s_i),i=1,...n\rigth&space;\}" vertical-align="middle"/>

Pour une fonction reward «f», un horizon «N», un facteur d’escompte «delta»,  les probabilités de transition «P» dans une matrice _m x n x n_,

On crée la structure «model»
```Matlab
model.reward    = f;
model.transprob = P;
model.horizon   = N;
model.discount  = delta;
```

Ensuite on trouve la solution <img src="https://latex.codecogs.com/gif.latex?[v,s]&space;=&space;[V_t(s),&space;x_t^{*}(s)]" vertical-align="middle"/>
```Matlab
[v,x] = ddpsolve(model);
```

Rechercher les informations suivantes:
	* State space
	* Action space
	* Reward function (based on Action space)
	* Transition probability matrix (m x n x n)

Pour un <strong>continuous state space</strong> S, le next state est une fonction continue du current state, du current action et d’un terme d’erreur (déterministe ou stochastique) : <img src="https://latex.codecogs.com/gif.latex?S'&space;=&space;g(S,X,\epsilon)" vertical-align="middle"/>

On travaille avec une approximation de la objective function (avec cubic spline)

<img src="https://latex.codecogs.com/gif.latex?V(S)&space;=&space;\sum_{j=1}^{n}C_j\phi_j(S)" vertical-align="middle"/>, à partir de collocation nodes <img src="https://latex.codecogs.com/gif.latex?\left\{S_1,S_2,S_3,...,S_n\right\}" vertical-align="middle"/>

Pour chaque collocation nodes <img src="https://latex.codecogs.com/gif.latex?S_i" vertical-align="middle"/>, on veut résoudre :

<img src="http://latex.codecogs.com/gif.latex?\sum_{j=1}^{n}C_j\phi_j(S)&space;=&space;max&space;\left\(f(S_i,X)&space;&plus;&space;\delta\sum_{S'\epsilon&space;S}Prob\left\(g(S,X,\epsilon)|S,X\right\)\sum_{j=1}^{n}C_j\phi_j\left\(g(S,X,\epsilon)\right\)\right\)" vertical-align="middle"/>

Si <img src="https://latex.codecogs.com/gif.latex?\\epsilon" vertical-align="middle"/> est stochastique, on calcule l’espérance : 

<img src="http://latex.codecogs.com/gif.latex?\sum_{S'\epsilon&space;S}Prob\left\(g(S,X,\epsilon)|S,X\right\)\sum_{j=1}^{n}C_j\phi_j\left\(g(S,X,\epsilon)\right\)" vertical-align="middle"/>

avec une méthode d’intégration (comme gaussian quadrature). À partir de nodes <img src="http://latex.codecogs.com/gif.latex?\epsilon_k" vertical-align="middle"/> et des poids <img src="http://latex.codecogs.com/gif.latex?W_k" vertical-align="middle"/>, l’équation devient: 

<img src="http://latex.codecogs.com/gif.latex?\sum_{j=1}^{n}C_j\phi_j(S)&space;=&space;max&space;\left\(f(S_i,X)&space;&plus;&space;\delta\sum_{k=1}^{z}W_k\sum_{j=1}^{n}C_j\phi_j\left\(g(S,X,\epsilon)\right\)\right\)" vertical-align="middle"/>


Concrètement…

Pour une fonction de transition d’état <img src="http://latex.codecogs.com/gif.latex?g(S,X,\epsilon)" vertical-align="middle"/>, avec <img src="http://latex.codecogs.com/gif.latex?\epsilon" vertical-align="middle"/> une v.a. <img src="http://latex.codecogs.com/gif.latex?N(0,\sigma^2)" vertical-align="middle"/>, 

On génère les nodes <img src="http://latex.codecogs.com/gif.latex?\epsilon_k" vertical-align="middle"/> et les poids <img src="http://latex.codecogs.com/gif.latex?W_k" vertical-align="middle"/>

```Matlab
[e,w] = qnwnorm(m,0,sigma^2);
```

Ensuite on trouve les <img src="http://latex.codecogs.com/gif.latex?\phi_j" vertical-align="middle"/>

```Matlab
basis = fundefn('spli', n, Smin, Smax);    
p     = funnode(basis);      
```

Puis on trouve <img src="http://latex.codecogs.com/gif.latex?\left\[\sum_{j=1}^{n}\phi_j(S_i),&space;pour&space;i&space;=&space;1,...,N\right\]" vertical-align="middle"/>

```Matlab
phi   = funbas(basis); 
```

Ensuite, à partir d’un vecteur de valeurs initiales <img src="http://latex.codecogs.com/gif.latex?C_0&space;=&space;[C_{01},C_{02},C_{03},...,C_{0n}]" vertical-align="middle"/> on calcule :

<img src="http://latex.codecogs.com/gif.latex?V(S)&space;=&space;max&space;\left\(f(S_i,X)&space;&plus;&space;\delta\sum_{k=1}^{z}W_k\sum_{j=1}^{n}C_j\phi_j\left\(g(S,X,\epsilon)\right\)\right\)" vertical-align="middle"/>

```Matlab
c = zeros(n,1);
```

Et on recherche <img src="http://latex.codecogs.com/gif.latex?C&space;=&space;[C_{1},C_{2},C_{3},...,C_{n}]" vertical-align="middle"/> tel que 

<img src="http://latex.codecogs.com/gif.latex?\sum_{j=1}^{n}C_j\phi_j(S)&space;=&space;V(S)" vertical-align="middle"/>

L’objectif est de trouver <img src="http://latex.codecogs.com/gif.latex?C&space;=&space;[C_{1},C_{2},C_{3},...,C_{n}]" vertical-align="middle"/> qui vérifie :

<img src="http://latex.codecogs.com/gif.latex?\sum_{j=1}^{n}C_j\phi_j(S)&space;=&space;max&space;\left\(f(S_i,X)&space;&plus;&space;\delta\sum_{k=1}^{z}W_k\sum_{j=1}^{n}C_j\phi_j\left\(g(S,X,\epsilon)\right\)\right\)" vertical-align="middle"/>


```Matlab
for it=1:1000
    cold   = c;
    vprime = 0; 
    for k=1:m 
        vprime = vprime + w(k)*funeval(c,basis,beta*p+e(k));
    end
    v = max(exp(p)-K,delta*vprime);
    c = Phi\v;
    if norm(c-cold)<1.e-10, break, end
end

% on devrait avoir au final, 
funeval(c,basis,p) = v
```

Pour un <strong>continuous action space</strong> X

Action X, est une fonction du state S ; on a l’équation du problème : 

<img src="http://latex.codecogs.com/gif.latex?V(S)&space;=&space;max\left\(f(S,X(S))&space;\plus&space;\deltaE(V(S,X(S),\epsilon))\right\)" vertical-align="middle"/>

Action X peut être contrainte ou pas.

Sans contrainte, on trouve la solution à partir de la condition de premier ordre : 























