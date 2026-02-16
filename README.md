# add all configurations 


### First add package finder

```toml
[tool.setuptools.packages.find]
include = ["specsyns*", "baselines*", "config*"]
```
### Install the package in editible mode 

```bash 
pip install -e .
```