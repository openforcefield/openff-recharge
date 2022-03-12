(theory_chapter)=
# Theory

This page aims to outline the key theoretical aspects of the framework, ranging from how the supported charge models are
applied to molecules up to how the framework constructs the objective ( / loss) functions used to train them.

## Supported charge models

A key feature of this framework is being able to apply different types of charge model to a molecule. There are 
currently three types of parameter available based upon the [SMIRNOFF] specification: library charges, bond charge
corrections, and virtual sites.

### Library charges 

A library charge is used to assign a set of specific partial charges to each atom in a molecule. It is a combination of 
a fully indexed SMILES pattern that describes the molecule the charges should be applied to, and a set of charges to 
apply to each atom in that SMILES pattern. A collection of these can be stored in a flat vector 

$$ \vec{p_{lib}} = 
  \begin{bmatrix}
     p_{1, 1} & \dots & p_{1, n_1} & \dots & p_{\gamma_{lib}, 1} & p_{\gamma_{lib}, n_{\gamma_{lib}}} \\
  \end{bmatrix} ^ T
$$ 

where here $\gamma_{lib}$ is the total number of library parameters in the model, $n_i$ is the number of atoms in the 
molecule represented by parameter $i$, and $p_{i, j}$ corresponds to the charge associated with atom $j$ in the molecule 
represented by parameter $i$.

### Bond charge corrections

A bond charge correction (BCC) is used to perturb the charge on two atoms involved in a given bond by an equal but 
opposite amount {cite}`jakalian2000fast,jakalian2002fast` such that the total charge on the molecule remains the same. 
It is a combination of a SMIRKS pattern that has exactly two bonded atoms tagged with an index (e.g. `[#6:1]-[#1:2]`) 
and the value of the charge correction to apply. A collection of these can be stored in a flat vector

$$ \vec{p_{bcc}} = 
  \begin{bmatrix}
     p_{1} & \dots & p_{\gamma_{bcc}} \\
  \end{bmatrix} ^ T
$$

where here $\gamma_{bcc}$ denotes the number of BCC parameters in the model and $p_i$ corresponds to the $i$'th BCC 
parameter.

The value of the charge correction will be added to the first indexed atom matched by the SMIRKS pattern, while the 
same value of the charge correction but with an opposite sign will be added to the second indexed atom matched by the 
SMIRKS pattern.

### Virtual sites

A virtual site (v-site) parameter is used to create a dummy particle on a molecule that is positioned relative to a 
number of parent atoms and carries a slight partial charge. Such particles are useful for, for example, representing
lone pairs of electrons or sigma holes that are not well represented by charges localised on atom centers.

Unlike the other two parameter types, a virtual site is a combination of both a set of charge increment values that 
shift charge from a parent atom onto the virtual site *and* a set of parameters that describe where the virtual site 
should be placed relative to the parent atoms.

Like the other parameter types we can store the charge increments in a flat vector

$$ \vec{p_{vsite}} = 
  \begin{bmatrix}
     p_{1, 1} & \dots & p_{1, n_1} & \dots & p_{\gamma_{vsite}, 1} & p_{\gamma_{vsite}, n_{\gamma_{vsite}}} \\
  \end{bmatrix} ^ T
$$ 

where here $\gamma_{vsite}$ is the total number of v-site parameters in the model, $n_i$ is the number of parent atoms 
that v-site parameter $i$ will be positioned relative to, and $p_{i, j}$ corresponds to the charge that will be added 
to parent atom $j$ and subtracted from the total v-site charge.

The coordinate parameters that must be provided will depend on the type of virtual site to be placed, but will include 
at least one of namely a distance $d$, out-of-plane angle $\theta_{out}$, and in-plane-angle $\theta_{in}$ defined 
relative to the parent atoms.

% TODO: positioning v-sites

## Applying charge models

In general, the vectors of charge parameters described above can be applied to a given molecule through the construction
of an *assignment matrix* {cite}`jakalian2000fast`. The assignment matrix

$$
    \mathbf{T}=\begin{bmatrix}
      T_{1,1} & \dots  & T_{1,\gamma_{\square}} \\
      \vdots  & \ddots & \vdots  \\
      T_{N,1} & \dots. & T_{N,\gamma_{\square}} \\
    \end{bmatrix}
$$

is a matrix that yields a vector of partial charges on each atom in a molecule when right multiplied by a flat vector of 
charge parameters as described above

$$ \vec{q_{\square}} = \mathbf{T}_{\square}\vec{p}_{\square} $$

where $N$ is the number of atoms and $\gamma_{\square}$ the number of charge parameters of type $\square$. 

In the case of library charges the assignment must have exactly one entry of 1 in each row. As an example consider
applying a library charge with a SMIRKS of `[C:1]([H:2])([H:3])([H:4])([H:5])` to methane

$$ 
    \vec{q_{lib}} =  
    \begin{bmatrix}
      1 & 0 & 0 & 0 & 0 \\
      0 & 1 & 0 & 0 & 0 \\
      0 & 0 & 1 & 0 & 0 \\
      0 & 0 & 0 & 1 & 0 \\
      0 & 0 & 0 & 0 & 1 \\
    \end{bmatrix}
    \begin{bmatrix}
       0.100 \\
      -0.025 \\
      -0.025 \\
      -0.025 \\
      -0.025 \\
    \end{bmatrix}
$$

In contrast, for bond charge correction parameters that assigment matrix can have rows that have any entry so long as
the constraint that the sum of each column must be 0 is satisfied. This ensures that the total charge on the molecule 
will remain 0 after the corrections have been applied. Considering for example the simple case of applying a BCC with 
SMIRKS of `[C:1][H:2]` to methane

$$ 
    \vec{q_{bcc}} =  
    \begin{bmatrix}
      4 \\
     -1 \\
     -1 \\
     -1 \\
     -1 \\
    \end{bmatrix}
    \begin{bmatrix}
      0.01 \\
    \end{bmatrix}
$$

For a given charge model that is some combination of library charges, BCCs, and v-sites the total charge on a molecule
will be

$$ 
\vec{q_{atom}} = \mathbf{T}_{lib}\vec{p_{lib}} + \mathbf{T}_{bcc}\vec{p_{bcc}} + \mathbf{T}_{vsite}\vec{p_{vsite}}
               = \begin{bmatrix} \mathbf{T}_{lib} & \mathbf{T}_{bcc} & \mathbf{T}_{vsite} \end{bmatrix} \begin{bmatrix} \vec{p_{lib}} \\ \vec{p_{bcc}} \\ \vec{p_{vsite}} \end{bmatrix}
$$

Alternatively a model may not contain any library charge parameters, such as the AM1BCC charge model, opting to instead
derive a base set of charges $q_{base}$, to be perturbed from some QM calculation. In such a case, the total charge will
instead be

$$ 
\vec{q_{atom}} = q_{base} + \mathbf{T}_{bcc}\vec{p_{bcc}} + \mathbf{T}_{vsite}\vec{p_{vsite}}
               = q_{base} + \begin{bmatrix} \mathbf{T}_{bcc} & \mathbf{T}_{vsite} \end{bmatrix} \begin{bmatrix} \vec{p_{bcc}} \\ \vec{p_{vsite}} \end{bmatrix}
$$

## Training to electronic property data

The charge parameters described above can be trained against electronic property data, namely the electrostatic 
potential (ESP) and the electric field computed on a grid of points, by constructing an objective ( / loss) function
to be minimized in the usual ways.

In particular, the calculated electronic property of interest $O\left(\vec{q}\right)$ can be inserted into the standard
least squares (L2) loss function such that 

$$
    \Delta = \sum^K_{i=1}\left||O_{ref,i} - O\left(\vec{q_i}\right)\right||^2
$$

where the summation is over $K$ molecules in a given conformation and $O_{ref}$ is the target value of the electronic 
property computed using some reference (normally an accurate QM) method.

The value of $O$ can be written more generally as

$$
    O\left(\vec{q}\right) = \mathbf{X} \vec{q} = \mathbf{X}\begin{bmatrix} \mathbf{T}_{lib} & \mathbf{T}_{bcc} & \mathbf{T}_{vsite} \end{bmatrix} \begin{bmatrix} \vec{p_{lib}} \\ \vec{p_{bcc}} \\ \vec{p_{vsite}} \end{bmatrix}
$$

In the case of training against ESP data that has been computed on a grid of $M$ points

$$
    X = \begin{bmatrix}
    \dfrac{1}{r_{1,1}} & \dots  & \dfrac{1}{r_{1,N}} \\
    \vdots             & \ddots & \vdots             \\
    \dfrac{1}{r_{M,1}} & \dots. & \dfrac{1}{r_{M,N}} \\
\end{bmatrix}
$$

where $r_{i,j}$ is the distance between grid point $i$ and atom $j$, while when training against electric field data

$$
    X = \begin{bmatrix}
    \dfrac{\hat{r}_{1,1}}{|\vec{r}_{1,1}|} & \dots  & \dfrac{\hat{r}_{1,N}}{|\vec{r}_{1,N}|} \\
    \vdots                                 & \ddots & \vdots                                  \\
    \dfrac{\hat{r}_{M,1}}{|\vec{r}_{M,1}|} & \dots. & \dfrac{\hat{r}_{M,N}}{|\vec{r}_{M,N}|} \\
\end{bmatrix}
$$

Adopting the language of [regression analysis], we refer in this framework to the combination of $\mathbf{X}$ and the 
assignment matrices as a **design matrix**.

:::{bibliography}
:style: unsrt
:::

[SMIRNOFF]: https://openforcefield.github.io/standards/standards/smirnoff/
[regression analysis]: https://en.wikipedia.org/wiki/Design_matrix