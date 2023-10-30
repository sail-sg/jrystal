# Jrystal

Plane-wave density functional theory for crystals in JAX

## Getting started




## Installation
Get source code from github

```sh
git clone git@github.com:sail-sg/jrystal.git
```

```sh
pip install .
```


#### Development mode
```sh
pip install --editable .
```

This allows to modify your source code and have the changes take effect without you having to rebuild and reinstall.


## Developer notes:

#### The Jrystal philosophy:

- Opt for an ``nn.module`` object when defining operations:
  - Involve trainable parameters, which necessitate persistent state across different stages of training.
  - Require numerous internal stateless variables that must be defined at initialization and remain constant during function calls, such as `black_wave`. While it is possible to implement such behavior using custom decorators, this approach often lacks generality. Consequently, utilizing an nn.module object is a more robust and preferable solution.
- Opt for an pure function when:
  - the function contains no trainable parameters, and has only a few function arguments or local stateful variables.

- Prefer duplicating code over a bad abstraction.

- A helpful error message is crucial.

- It's better for an abstraction to be small and specific than to be large and general.
