# casadi-ccopt
This is a tiny package that helps use Opti to construct problems for a solver with explicit complementarity support such as [Marble](https://roboticexplorationlab.org/Marble/) or CCopt. These solvers rely on indices of which constraints are complementarity to each other, which Opti abstracts away.

There are two pieces
- Use complementarity(opti, left, right) to add constraints of the form 0 <= left ⟂ right >= 0 
- Use construct_ccopt_solver(opti, opt=None) to initialize the solver (in this case CCOpt)

Internally, complementarity defines slack variables for each side and records them. Then construct_ccopt_solver finds the indices corresponding to those vectors in opti.x and records them as cc_pairs for the solver, as well as producing lbx that has 0 for the lower bounds for those slacks. If left or right is already a registered opti variable, you can avoid creating
a redundant slack by passing optional args left_var = True or right_var = True respectively.

## Installation and environment

Requires Python 3.9+, CasADi with the `ccopt` `nlpsol` plugin, CCOpt, and compatible LibMad
libraries available to the runtime. Plain `pip install casadi` is not necessarily sufficient.
Install and configure the solver libraries separately, including library search paths such as
`LD_LIBRARY_PATH` where needed.

With that environment active, install this module:

```bash
pip install --no-deps -e .
```

## Usage

```python
import casadi as ca
from casadi_ccopt import complementarity, construct_ccopt_solver

# Construct opti problem as normal
opti = ca.Opti()
x = opti.variable()
y = opti.variable()
a = opti.parameter()
b = opti.parameter()
opti.minimize((x - a)**2 + (y - b)**2)

# Add complementarity between x and y, 0 <= left ⟂ right >= 0 (creates slack variables stored in opti._cc_var)
complementarity(opti, x, y) 
# Optionally can do complementarity(opti, x, y, left_var = True, right_var = True) if either
# is a registered opti.variable to avoid creating a slack

# Construct the solver (uses opti._cc_var to find complementarity indices and passes them to CCOpt).
solver, lbx = construct_ccopt_solver(opti)

opti.set_value(a, 1.0)
opti.set_value(b, 2.0)
result = solver(
    x0=opti.value(opti.x, opti.initial()),
    p=opti.value(opti.p),
    lbx=lbx,
    lbg=opti.value(opti.lbg),
    ubg=opti.value(opti.ubg),
)
```
