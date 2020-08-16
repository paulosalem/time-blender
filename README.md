# TimeBlender
A programmatic and compositional time series generator.


## Introduction

TimeBlender is a programmatic compositional time series generator. By *programmatic* it is meant that series are
specified through programming; and, by *compositional*, that the programming structures used to this end
can be combined (or, well, *blended*) to achieve complex results.

This software has a dual purpose:

  - to allow the author to study time series from a generative point of view, either by implementing existing
    concepts or researching new ones.
  - and to produce artificial time series of practical interest.
  
On the one hand, these two objectives are antagonistic in the sense that early research might result in inadequate or not 
optimal ways of achieving results (i.e., because new ways are being sought, and some of these might prove pointless), 
which hinders practical uses.  On the other hand, they are complementary in the sense that good results are not easy to 
achieve without good foundations, and these require research. Obviously, the present software is being developed 
because the latter aspect seems stronger, owing to a perception that in industrial practice we lack good synthetic time 
series generation. For example, while general ARMA-based generators are easy to find, the author could not locate 
generators for user behavior in financial applications such as banking (see model examples below).

Target applications include:

  - Permit data scientists to work with artificial, but realistic, data while access to real data is not available
    (either because it does not exist yet, or because bureaucratic procedures create unreasonable delays).
  
  - Data augmentation, particularly for RNN models.
  
  - Artificial stress scenarios simulation (e.g., a market crash).
  
**Please note that this is a very early prototype, therefore API stability cannot be guaranteed, as 
pretty much anything could change.**


## Features

Main features:

  - Event-based: each time point is generated based on an *event* class. Several standard such events are provided,
                 and it is easy to add more.
  - Programmatic: events can be specified by arbitrary (Python) programs, hence they are not limited to traditional
                  statistical techniques. For example, agent-based models could be defined to simulate
                  market data. 
  - Compositional: events can be composed to obtain complex events.
  - Pandas-based: Pandas is used in various parts to allow convenient post-generation processing options and integration 
                  with other tools.


(Even more) experimental features:

  - Learning:  Rudimentary and early support for learning from observations is provided
               through black-box parameter optimization 
               (currently using the [hyperopt](https://hyperopt.github.io/hyperopt/) library). 
               The idea here is that one can learn from real time series in order to be able
               to generate similar ones automatically. The objective *is not* forecasting, although that might
               be possible eventually. At present, it is unclear how this aspect will evolve.
               

Standard models, which work both as examples and as a basic model library, include:

  - AR, MA and ARMA.
  - Seasonal effects.
  - Banking behavior of salary earners.
  - Kondratiev business cycle.
  
Please note that some of the above are provided as rather naive implementations. It is hoped that more sophisticated
models take their place as the library improves.

## Installation

TimeBlender is developed in Python and provided as a PIP package:

```bash
pip install time-blender
```

It can also be installed directly from GitHub:
```bash
pip install git+https://github.com/paulosalem/time-blender#egg=time-blender
```

## Use

TimeBlender is designed mainly for programmatic use. However, for convenience, a command-line interface is also 
provided and allows access to pre-defined models.

### Programmatic Use

A Jupyter notebook is provided with many examples on how to use TimeBlender. Here, let us take a look at some simple
examples. 

The following would generate a wave of period 30 and amplitude 3, summed with normal noise of mean 0 and standard 
deviation 1:

```python
  import pandas as pd
  from time_blender.core import *
  from time_blender.random_events import *
  from time_blender.deterministic_events import *
  from time_blender.coordination_events import *
  
  norm = NormalEvent(0, 1)
  we = WaveEvent(30, 3)
  
  compos = norm + we
  
  data = Generator(model).generate(pd.Timestamp(2018, 1, 1),  pd.Timestamp(2018, 31, 1), n=1)
  
  # data[0] contains the generated series.
```

Some models are predefined for convenience. For instance, a random ARMA(4, 2) model can be defined as:

```python
  from time_blender.models import ClassicModels
  
  model = ClassicModels.arma(4, 2)
  data = Generator(model).generate(pd.Timestamp(2018, 1, 1),  pd.Timestamp(2018, 31, 1), n=1)
  
  # data[0] contains the generated series.

```


### Command-Line Use

The PIP package installs a script named `time_blender`. To see the available options, models and their parameters, type:

```bash
  time_blender -h
```
 

## License

MIT License

Copyright (c) 2018 Paulo Salem

Please see the attached LICENSE file for details.
