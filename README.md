
J'ai initialement compilé ces notes pour mon propre usage, afin d'acceder rapidement aux concepts et aux codes MATLAB en cas de besoin. Ces notes sont plus indiquées pour des personnes déjà introduites aux concepts, car les concepts sont sommairement présentés. Pour plus de détails, je recommande de regarder la lecture du livre numérique 
<a href="https://github.com/madeenamadou/econ-comp/blob/master/%5BMario_J._Miranda%2C_Paul_L._Fackler%5D_Applied_Comput.pdf">Applied Computational Economics and Finance</a> de Mario J. Miranda & Paul L. Fackler.

J'ajouterai des ressources et codes MATLAB sur le meme theme avec le temps, donc n'hesitez pas à <a href="https://github.com/madeenamadou/">suivre</a> ou <a href="https://github.com/madeenamadou/econ-comp/fork">copier</a> le <a href="https://github.com/madeenamadou/econ-comp/">répertoire GitHub</a>. Le contenu de cette page a été écrit en <strong>Markdown</strong>.

Bonne lecture !!

<hr>






## Système d’équations linéaire : Ax = b
### Méthodes directes

Décomposition LU, Cholesky `x = A\b`

Méthode itérative (forme Qx = b + (Q − A)x)

- Gauss-jacobi : `gjacobi (A,b)`
- Gauss-seidel : `gseidel (A,b)`

## Systèmes d’équations non linéaires : Points fixes (f(x0) = x0) & Solutions racines (f(x0) = 0)
### Méthode bisection, sur un interval [a,b]

Pour une fonction f,
```Matlab
bisect ('f',a,b)
```

### Méthode Newton : une ou plusieurs variables; avec des valeurs initiales; utilise le jacobien

Pour une fonction f à 2 variables, avec des valeurs initiales respectives x1 et x2 
```Matlab
newton('f',[x1;x2])
```

### Méthode Quasi-Newton : utilise une approximation du jacobien
- Secant Method : une variable
	
	Broyden Method : plusieurs variables; utilise une valeur initiale pour la racine, et une autre pour le Jacobien
 	Pour une fonction f à deux variables, et pour les valeurs initiales x1 et x2 des variables
	```Matlab
	broyden('f',[x1;x2])
	```
	
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
- Method Golden Search : une variable; optimum local sur un interval [a,b]
	
	Pour une fonction f, sur un interval [a,b]
	```Matlab
	golden('f',a,b)
	```

- Méthode Algorithm Nelder-Mead : plusieurs variables; avec des valeurs initiales pour les variables
	
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
	
	>Lorsque *w(x)=1*, on obtient l’aire sous la fonction
	
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
	>Si w(x) = *fonction de densité de probabilité* de *x*, on obtient l’espérance de la fonction par `Somme(w*f(x))`

	**Méthodes Intégration Monte-Carlo**
	
	Il faut générer pseudo-aléatoirement *n* nodes *x* d’après la distribution ; les weights *w=1/n* étant identiques
	
	L’espérance de *f* est obtenue par
	```Matlab
	Somme(w*f(x))
	```

	**Méthodes Quasi-Monte Carlo**
	
	Ici les *n* nodes *x* sont déterministes, sur un intervalle [a,b],  les weights *w=(b-a)/n* étant identiques.
	
	L’espérance de *f* est obtenue par
	```Matlab
	Somme(w*f(x))
	```
 
### Méthode de différentiation

Pour une fonction *f*, une approximation *O(h²)* de sa dérivée, autour du point *x0*, est de la forme:
><img src="http://latex.codecogs.com/gif.latex?af(x_0)&space;+&space;bf(x_0+h)&space;+&space;cf(x_0&space;+&space;\alphah)&space;[+O(h^2)]" vertical-align="middle"/>

>où <img src="http://latex.codecogs.com/gif.latex?(x_0&space;+&space;h)" vertical-align="middle"/>, <img src="http://latex.codecogs.com/gif.latex?(x_0&space;+&space;\alphah)" vertical-align="middle"/> sont deux autres points,
>choisir, alpha = 2, et choisir h quelconque, les paramètres a, b et c, s’obtiennent en résolvant le système suivant :

```
a + b + c = 0
b + cλ = 1/h
b + cλ2 = 0
```

## Initial Value Problem (IVP)
Prend la forme d’une equation différentielle à résoudre pour une function solution, connaissant les valeurs initiales pour la fonction.

>e.g. à l’ordre 1  <img src="http://latex.codecogs.com/gif.latex?y'(t)&space;=&space;f(t,y(t))" vertical-align="middle"/>

>ou à l’ordre 2 <img src="http://latex.codecogs.com/gif.latex?y''(t)&space;=&space;f(t,y(t),y'(t))" vertical-align="middle"/> 

Il faut réécrire parfois le problème sous la forme d’une équation différentielle. Une équation différentielle d’ordre 2 peut se ramener à un système d’équation différentielle d’ordre 1.

Le principe de résolution est de dériver une approximation de taylor de f(x+h) à l’ordre 1 (Méthode de Euler), ou 2 (Méthode Runge-Kutta 2), ou 4 (Runge-Kutta 4). On choisit un pas h pour subdiviser la période de temps ; plus petit h, mieux c’est en général.

>Méthode de Euler : <img src="http://latex.codecogs.com/gif.latex?y'(t+h)&space;=&space;y(t)&space;+&space;y'(t)h" vertical-align="middle"/>

>Méthode Runge-Kutta 2 : <img src="http://latex.codecogs.com/gif.latex?y'(t+h)&space;=&space;y(t)&space;+&space;h[a_1k_1&space;+&space;a_2k_2]" vertical-align="middle"/>, avec :

><img src="http://latex.codecogs.com/gif.latex?k_1&space;=&space;f(x,y(x))" vertical-align="middle"/>

><img src="http://latex.codecogs.com/gif.latex?k_2&space;=&space;f(x&space;+&space;p_1h,y(x)&space;+&space;q_1k_1h)" vertical-align="middle"/>

Pour une bonne approximation, il faut que :

<img src="https://latex.codecogs.com/gif.latex?\left\{\begin{matrix}&space;a_1&space;&plus;&space;a_2&space;=&space;1\\&space;a_2p_1&space;=&space;\frac{1}{2}\\&space;a_2q_1&space;=&space;\frac{1}{2}&space;\end{matrix}\right." vertical-align="middle"/>

On choisit <img src="https://latex.codecogs.com/gif.latex?a_2" vertical-align="middle"/> puis on trouve ensuite les autres paramètres. Il existe trois méthodes communes pour évaluer les paramètres inconnus ou connus : la méthode de Heun, la méthode du point milieu et la méthode de Ralston

**Heun’s Method** : <img src="https://latex.codecogs.com/gif.latex?(a_2,a_1,p_1,q_1)&space;=&space;\left&space;(\frac{1}{2},\frac{1}{2},1,1\right&space;)" vertical-align="middle"/>

**Midpoint’s Method** : <img src="https://latex.codecogs.com/gif.latex?(a_2,a_1,p_1,q_1)&space;=&space;\left&space;(1,0,\frac{1}{2},\frac{1}{2}\right&space;)" vertical-align="middle"/>

**Ralston’s Method** : <img src="https://latex.codecogs.com/gif.latex?(a_2,a_1,p_1,q_1)&space;=&space;\left&space;(\frac{2}{3},\frac{1}{3},\frac{3}{4},\frac{3}{4}\right&space;)" vertical-align="middle"/>

**Paul L. Fackler & Mario J. Miranda** : <img src="https://latex.codecogs.com/gif.latex?(a_2,a_1,p_1,q_1)&space;=&space;\left&space;(\frac{3}{4},\frac{1}{4},\frac{2}{3},\frac{2}{3}\right&space;)" vertical-align="middle"/>

**Méthode Runge-Kutta 4** : 

<img src="https://latex.codecogs.com/gif.latex?y(x&space;+&space;h)&space;=&space;y(x)&space;+&space;\frac{1}{6}[F_1&space;+&space;2(F_2&space;+&space;F_3)&space;+&space;F_4]" vertical-align="middle"/>

>Avec
>
><img src="https://latex.codecogs.com/gif.latex?F_1&space;=&space;hf(x,y(x))" vertical-align="middle"/>
>
><img src="https://latex.codecogs.com/gif.latex?F_2&space;=&space;hf(x&space;+&space;\frac{1}{2}h,y(x)&space;+&space;\frac{1}{2}F_1)" vertical-align="middle"/>
>
><img src="https://latex.codecogs.com/gif.latex?F_3&space;=&space;hf(x&space;+&space;\frac{1}{2}h,y(x)&space;+&space;\frac{1}{2}F_2)" vertical-align="middle"/>
>
><img src="https://latex.codecogs.com/gif.latex?F_4&space;=&space;hf(x&space;+&space;h,y(x)&space;+&space;F_3)" vertical-align="middle"/>

Pour un système d’équation différentiel « _system_ », une période allant de t0 à tf, des valeurs initiales stockées dans la matrice « _inits_ », la résolution numérique avec Runge-Kutta 4 se fait comme suit :

```Matlab
[t,y] = ode45('system',[t0 tf],inits)
[t,y] = rk4('system',(t0:tf)',inits)
```

>Ou, en utilisant MUPAD, avec un pas « _h_ », des valeurs initiales stockées dans la matrice « _Y0_ », la période initiale « _t0_ », un système d’équation différentiel « _f_ », la résolution numérique avec Runge-Kutta 4 se fait comme suit :
```Matlab
sol1 := numeric::odesolve2(f, t0, Y0, RK4, Stepsize=h)
```

## Boundary Value Problem
Il s’agit d’un boundary value problem quand on ne connait pas les valeurs initiales, pour la fonction. Il est souvent donné une boundary condition, comme par exemple la valeur terminale pour la fonction <img src="https://latex.codecogs.com/gif.latex?y_T" vertical-align="middle"/>.

On cherche un guess <img src="https://latex.codecogs.com/gif.latex?y_0" vertical-align="middle"/> pour la valeur initiale pour la fonction, tel que :

<img src="https://latex.codecogs.com/gif.latex?y_0(T)&space;-&space;y_T&space;=&space;0" vertical-align="middle"/>

On réécrit ce problème sous la forme d’un problème de recherche de solution racine, à résoudre avec **broyden**. Une fois trouvée <img src="https://latex.codecogs.com/gif.latex?y_0" vertical-align="middle"/>, on peut résoudre l’IVP.

e.g: ici, on doit chercher la solution racine pour la fonction « _shooting_ » avec **broyden**.

Définir d'abord la fonction

```Matlab
Function F = shooting (y0)
[t,y] = ode45('system',[t0 tf],y0);
F = y(end) - y_end;
```

puis,

```Matlab
y0 = broyden('shooting',guess);
```

## Interpolation

Il s’agit de trouver une approximation <img src="https://latex.codecogs.com/gif.latex?\hat{f}" vertical-align="middle"/> aussi précise que possible d’une fonction <img src="https://latex.codecogs.com/gif.latex?f" vertical-align="middle"/>.

Pour un ensemble de _n_ interpolation nodes <img src="https://latex.codecogs.com/gif.latex?(x_i,f(x_i))" vertical-align="middle"/>, on veut que : 

<img src="https://latex.codecogs.com/gif.latex?\hat{f}(x_i)&space;=&spacef(x_i);" vertical-align="middle"/>

De façon générale, <img src="https://latex.codecogs.com/gif.latex?\hat{f}" vertical-align="middle"/> est construite à partir d’une combinaison linéaire de plusieurs fonctions connues _ϕ_. Pour _m_ degrés d’interpolation (donc _m_ fonctions <img src="https://latex.codecogs.com/gif.latex?\phi_j" vertical-align="middle"/>), on a :

<img src="https://latex.codecogs.com/gif.latex?\hat{f}&space;=&space;\sum_{j=1}^{m}c_j\phi_j\\j&space;=&space;1,...,m" vertical-align="middle"/>

Les <img src="https://latex.codecogs.com/gif.latex?c_j" vertical-align="middle"/> sont inconnues et pour les obtenir, on résout simultanément les équations linéaires suivantes :

Pour _i = 1,...,n_,

<img src="https://latex.codecogs.com/gif.latex?\sum_{j=1}^{m}c_j\phi(x_i)&space;-&space;f(x_i)&space;=&space;0" vertical-align="middle"/>

Les méthodes d’interpolations diffèrent dans le choix des fonctions ϕ. Les <img src="https://latex.codecogs.com/gif.latex?\phi_j" vertical-align="middle"/> sont généralement des fonctions polynômes. On peut utiliser des polynômes Chebychev.

Pour un degré d’interpolation _m_, un intervalle de nodes <img src="https://latex.codecogs.com/gif.latex?x_i&space;\epsilon&space;[a,b]" vertical-align="middle"/>, on obtient les <img src="https://latex.codecogs.com/gif.latex?\phi_j" vertical-align="middle"/>
```Matlab
phi = fundefn('cheb', m, a, b);
```

Ensuite, on obtient les <img src="https://latex.codecogs.com/gif.latex?c_j" vertical-align="middle"/> pour une interpolation de <img src="https://latex.codecogs.com/gif.latex?f" vertical-align="middle"/>
```Matlab
c = funfitf(fhat, f);
```

Puis on calcule <img src="https://latex.codecogs.com/gif.latex?\hat{f}" vertical-align="middle"/>, pour _x_ le vecteur-colonne des nodes <img src="https://latex.codecogs.com/gif.latex?x_i" vertical-align="middle"/>
```Matlab
fhat = funeval(c,phi,x)
```

On peut également utiliser des polynômes splines. La méthode _cubic splines_ désigne l’interpolation à l’ordre 3 en utilisant cette famille de polynômes. En utilisant _cubic splines_ <img src="https://latex.codecogs.com/gif.latex?(k=3)" vertical-align="middle"/>, on procède comme suit : 
```Matlab
phi = fundefn('spli', m, a, b, k); 
phi(nodes) = funbas(basis);
c = funfitf(fhat, f); 
%ou
c = phi(nodes) \ v ; 
fhat = funeval(c,phi,x);
```

## Méthode Collocation
Il s’agit d’une méthode pour trouver la fonction f solution d'un problème du type :

<img src="https://latex.codecogs.com/gif.latex?g(x,f(x))=0" vertical-align="middle"/>,

sur un intervalle <img src="https://latex.codecogs.com/gif.latex?[a,b]" vertical-align="middle"/>. C'est comme résoudre un système d’équation différentielle d’ordre zéro. Le principe est d’utiliser une interpolation <img src="https://latex.codecogs.com/gif.latex?\hat{f}" vertical-align="middle"/> de f.

Pour un ensemble de _n_ collocation nodes <img src="https://latex.codecogs.com/gif.latex?x_i&space;\epsilon&space;[a,b]" vertical-align="middle"/> chercher la solution au problème revient à résoudre simultanément les équations linéaires suivantes, pour <img src="https://latex.codecogs.com/gif.latex?c_j" vertical-align="middle"/> :

Pour <img src="https://latex.codecogs.com/gif.latex?i=1,...,n" vertical-align="middle"/>

<img src="https://latex.codecogs.com/gif.latex?g(x_i,&space;\sum_{j=1}^{m}c_j\phi_j(x_i))&space;=&space;0" vertical-align="middle"/>

On peut obtenir les collocations nodes à partir de nodes chebychev…
```Matlab
phi = fundefn('cheb', m, a, b);
x = funnode(phi);
```

Ensuite, on peut résoudre ce problème en utilisant **broyden**.

Pour un système d’équation « _system_ », 
```Matlab
c = broyden('system',guess);
```

## Programmation Dynamique

Soient _**N**_ périodes <img src="https://latex.codecogs.com/gif.latex?t=1,...,N" vertical-align="middle"/> ; l’horizon _**N**_ pouvant être fini ou infini. On peut observer à chaque période, plusieurs états <img src="https://latex.codecogs.com/gif.latex?s" vertical-align="middle"/> possibles ; l’espace des états peut être discret ou continu.

<img src="https://latex.codecogs.com/gif.latex?S=\left&space;\{&space;s_1,s_2,s_3,s_4,...\right&space;\}" vertical-align="middle"/>

Pour chaque état, il y a un ensemble d’actions disponibles <img src="https://latex.codecogs.com/gif.latex?x" vertical-align="middle"/> ; l’espace des actions peut aussi être discret ou continu.

<img src="https://latex.codecogs.com/gif.latex?X=\left&space;\{&space;x_1,x_2,x_3,x_4,...\right&space;\}" vertical-align="middle"/>

### Discrete state space, Discrete action space
* Horizons finis

À chaque période _**t**_, l’action _**x**_ dans l’état _**s**_ produit le reward f(x,s). _**S**_ est un processus, soit déterministe, soit stochastique.

Pour S stochastique, on suppose qu’il s’agit d’un **markov decision process** ; la probabilité d’un état <img src="https://latex.codecogs.com/gif.latex?s'=s_{t+1}" vertical-align="middle"/> dépend donc seulement de l’état _**s**_ et de l'action _**x**_ à la periode precedente _**t**_. Quand il n’y a qu’une seule action disponible pour chaque état possible, on parle de **processus chaine de markov**.

Soit la probabilité de transition <img src="https://latex.codecogs.com/gif.latex?P(s'=s_{t+1}|s=s_t,x=x_t)" vertical-align="middle"/>

L’objectif est de trouver un ensemble optimal d’actions <img src="https://latex.codecogs.com/gif.latex?X^*_t={x^*(s),&space;s\epsilon&space;S}" vertical-align="middle"/> pour chaque <img src="https://latex.codecogs.com/gif.latex?t=1,...,N" vertical-align="middle"/>

Soit <img src="https://latex.codecogs.com/gif.latex?V_t(s)" vertical-align="middle"/> le reward maximum que l’agent peut atteindre, partant de _**t**_ jusqu’à la fin l'horizon _**T**_. Pour chaque état _**s**_ possible à _**t**_, il y a une valeur <img src="https://latex.codecogs.com/gif.latex?V_t(s)" vertical-align="middle"/>

Pour <img src="https://latex.codecogs.com/gif.latex?s&space;\epsilon&space;S" vertical-align="middle"/>,

<img src="https://latex.codecogs.com/gif.latex?V_t(s)&space;=&space;max&space;\left&space;\{&space;f(s,x)&space;&plus;&space;\delta&space;\sum_{s^{'}&space;\epsilon&space;S}&space;Prob&space;(s_{t&plus;1}&space;=&space;s^{'}|s&space;=&space;s_t,&space;x=x_t)V_{t&plus;1}(s^{'})\right&space;\}" vertical-align="middle"/>

C'est <strong>l’équation de Bellman</strong>.

Ensuite, on cherche <img src="https://latex.codecogs.com/gif.latex?x&space;\epsilon&space;X" vertical-align="middle"/> tel que :

<img src="https://latex.codecogs.com/gif.latex?x^{*}_t(s)&space;=&space;argmax(x){V_t(s),s\epsilon&space;S}" vertical-align="middle"/>

On impose une condition terminale, généralement <img src="https://latex.codecogs.com/gif.latex?V_{t+1}(s)=0" vertical-align="middle"/>. Avec cette condition, on progresse backward, en cherchant <img src="https://latex.codecogs.com/gif.latex?V_t(s)" vertical-align="middle"/> puis <img src="https://latex.codecogs.com/gif.latex?V_{t-1}(s)" vertical-align="middle"/> et ainsi de suite, jusqu’à trouver <img src="https://latex.codecogs.com/gif.latex?V_1(s)" vertical-align="middle"/>.

* Horizons infinis

Pour les horizons infinis, <img src="https://latex.codecogs.com/gif.latex?V(s)" vertical-align="middle"/> ne dépend plus de _t_ et le problème s’écrit :

<img src="http://latex.codecogs.com/gif.latex?V(s)&space;=&space;max&space;\left&space;\{&space;f(s,x)&space;&plus;&space;\delta&space;\sum_{s'&space;\epsilon&space;S}Prob(s'|s,x)V(s')&space;\right&space;\}\\&space;\\&space;x^{*}(s)&space;=&space;argmax(x\left&space;\{&space;V(s),s&space;\epsilon&space;S&space;\right&space;\})" vertical-align="middle"/>

<img src="https://latex.codecogs.com/gif.latex?V_{t+1}(s)=0" vertical-align="middle"/>

Pour <img src="https://latex.codecogs.com/gif.latex?s&space;\epsilon&space;S" vertical-align="middle"/>,

<img src="https://latex.codecogs.com/gif.latex?V_T(s)&space;=&space;max&space;\left&space;\{&space;f(s,x)\right&space;\}&space;\\&space;V_{t-1}(s)&space;=&space;max&space;\left&space;\{&space;f(s,x)&space;&plus;&space;\delta&space;\sum_{s'\epsilon&space;S}&space;Prob&space;(s_T=s^{'}|s_{T-1}&space;=&space;s,&space;x_{T-1}=x)V_T(s^{'})\right&space;\}\\" vertical-align="middle"/>

Et ainsi de suite….

On peut utiliser la méthode _backward recursion_.

Soit _n_ le nombre fini d’états possibles, _m_ le nombre fini d’actions disponibles. À la période _**t**_, soit la matrice des probabilités de transition vers l’état <img src="https://latex.codecogs.com/gif.latex?s_i&space;\epsilon&space;S" vertical-align="middle"/> : 

<img src="https://latex.codecogs.com/gif.latex?P(s_i)=\begin{bmatrix}&space;Prob(s_i|s_1,x_1)&space;&&space;...&space;&&space;Prob(s_i|s_n,x_1)\\&space;Prob(s_i|s_1,x_2)&space;&&space;...&space;&&space;Prob(s_i|s_n,x_2)\\&space;.&space;&&space;...&space;&&space;.\\&space;.&space;&&space;...&space;&&space;.\\&space;Prob(s_i|s_1,x_m)&space;&&space;...&space;&&space;Prob(s_i|s_n,x_m)&space;\end{bmatrix}" vertical-align="middle"/>

Il y a _n_ possibilités pour l’état <img src="https://latex.codecogs.com/gif.latex?s_i" vertical-align="middle"/> suivant :

<img src="https://latex.codecogs.com/gif.latex?P(m(currentaction)&space;\times&space;n(currentstate)&space;\times&space;n(futurestate))&space;=&space;\left&space;\{P(s_i),i=1,...n\rigth&space;\}" vertical-align="middle"/>

Pour une fonction reward « _f_ », un horizon « _N_ », un facteur d’escompte « _delta_ », les probabilités de transition « _P_ » dans une matrice _m x n x n_,

On définit la structure « _model_ »
```Matlab
model.reward    = f;
model.transprob = P;
model.horizon   = N;
model.discount  = delta;
```

Ensuite on recherche la solution <img src="https://latex.codecogs.com/gif.latex?[v,s]&space;=&space;[V_t(s),&space;x_t^{*}(s)]" vertical-align="middle"/>
```Matlab
[v,x] = ddpsolve(model);
```

 En résumé, les informations clés pour resoudre le problème :
	* State space
	* Action space
	* Reward function (based on Action space)
	* Transition probability matrix (m x n x n)

### Continuous state space
Pour un <strong>continuous state space</strong> _**S**_, le next _state_ est une fonction continue du _current state_, du _current action_ et d’un terme d’erreur (déterministe ou stochastique) : 

<img src="https://latex.codecogs.com/gif.latex?S'&space;=&space;g(S,X,\epsilon)" vertical-align="middle"/>

On travaille avec une approximation (de type _cubic splines_) de la fonction objectif

<img src="https://latex.codecogs.com/gif.latex?V(S)&space;=&space;\sum_{j=1}^{n}C_j\phi_j(S)" vertical-align="middle"/>, à partir de collocation nodes <img src="https://latex.codecogs.com/gif.latex?\left\{S_1,S_2,S_3,...,S_n\right\}" vertical-align="middle"/>

Pour chaque _collocation nodes_ <img src="https://latex.codecogs.com/gif.latex?S_i" vertical-align="middle"/>, on veut résoudre :

<img src="http://latex.codecogs.com/gif.latex?\sum_{j=1}^{n}C_j\phi_j(S)&space;=&space;max&space;\left\(f(S_i,X)&space;&plus;&space;\delta\sum_{S'\epsilon&space;S}Prob\left\(g(S,X,\epsilon)|S,X\right\)\sum_{j=1}^{n}C_j\phi_j\left\(g(S,X,\epsilon)\right\)\right\)" vertical-align="middle"/>

Si <img src="https://latex.codecogs.com/gif.latex?\\epsilon" vertical-align="middle"/> est stochastique, on obtient l’espérance : 

<img src="http://latex.codecogs.com/gif.latex?\sum_{S'\epsilon&space;S}Prob\left\(g(S,X,\epsilon)|S,X\right\)\sum_{j=1}^{n}C_j\phi_j\left\(g(S,X,\epsilon)\right\)" vertical-align="middle"/>

avec une méthode d’intégration (gaussian quadrature par exemple). À partir de nodes <img src="http://latex.codecogs.com/gif.latex?\epsilon_k" vertical-align="middle"/> et des poids <img src="http://latex.codecogs.com/gif.latex?W_k" vertical-align="middle"/>, l’équation devient: 

<img src="http://latex.codecogs.com/gif.latex?\sum_{j=1}^{n}C_j\phi_j(S)&space;=&space;max&space;\left\(f(S_i,X)&space;&plus;&space;\delta\sum_{k=1}^{z}W_k\sum_{j=1}^{n}C_j\phi_j\left\(g(S,X,\epsilon)\right\)\right\)" vertical-align="middle"/>


Concrètement…

Pour une fonction de transition d’état <img src="http://latex.codecogs.com/gif.latex?g(S,X,\epsilon)" vertical-align="middle"/>, avec <img src="http://latex.codecogs.com/gif.latex?\epsilon" vertical-align="middle"/> une variable aléatoire <img src="http://latex.codecogs.com/gif.latex?N(0,\sigma^2)" vertical-align="middle"/> 

On génère les nodes <img src="http://latex.codecogs.com/gif.latex?\epsilon_k" vertical-align="middle"/> et les poids <img src="http://latex.codecogs.com/gif.latex?W_k" vertical-align="middle"/>...

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

Et on recherche <img src="http://latex.codecogs.com/gif.latex?C&space;=&space;[C_{1},C_{2},C_{3},...,C_{n}]" vertical-align="middle"/> tel que :

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

### Continuous action space
_**X**_ est une fonction du state _**S**_. L’équation du problème devient : 

<img src="http://latex.codecogs.com/gif.latex?V(S)&space;=&space;max\left\(&space;f(S,X(S))&space;+&space;\delta&space;E(V(S,X(S),\epsilon))\right\)" vertical-align="middle"/>

>_**X**_ peut être contrainte ou non

Sans contrainte, on trouve la solution à partir de la condition de premier ordre : 

<img src="http://latex.codecogs.com/gif.latex?V'(S)&space;=&space;f_s&space;&plus;&space;f_X&space;\frac{\partial&space;X}{\partial&space;S}&space;&plus;&space;\delta&space;E&space;\left&space;(&space;V'(S')&space;\left&space;(&space;g_s&space;&plus;&space;g_X&space;\frac{\partial&space;X}{\partial&space;S}&space;\right&space;)&space;\right&space;)&space;\\&space;\\&space;V'(S)&space;=&space;\left&space;(&space;f_X&space;&plus;&space;\delta&space;E&space;\left&space;(&space;V'(S')&space;\right&space;)&space;g_X&space;\right&space;)&space;\frac{\partial&space;X}{\partial&space;S}&space;&plus;&space;f_S&space;&plus;&space;\delta&space;E&space;\left&space;(&space;V'(S')&space;\right&space;)&space;g_S&space;\\&space;\\&space;f_X&space;&plus;&space;\delta&space;E&space;\left&space;(V('(S')\right&space;)g_X&space;=&space;0" vertical-align="middle"/>

Donc <img src="http://latex.codecogs.com/gif.latex?V'(S)&space;=&space;f_S&space;+&space;\deltaE(V'(S'))g_s&space;=&space;0" vertical-align="middle"/>























