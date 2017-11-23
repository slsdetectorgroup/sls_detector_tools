# sls detector tools
A collection of tools for detector related tasks in Python

### Documentation

[https://slsdetectorgroup.github.io/sls_detector_tools/](https://slsdetectorgroup.github.io/sls_detector_tools/)


### Build and install using conda

```bash
#Clone the repo
git clone git@github.com:slsdetectorgroup/sls_detector_tools.git

#Build and install
conda-build sls_detector_tools
conda install --use-local sls_detector_tools
```

### Interacting with instruments and measurements setups

 * **SourceMeter** - Keithly SourceMeter using serial to USB adapter
 * **AgilentMultiMeter** - Agilent multimeter over telnet
 * **XrayBox** - Big X-ray box using command line calls to the client

```python

from sls_detector_tools import SourceMeter

```
