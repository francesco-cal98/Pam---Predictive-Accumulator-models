# Pyton-PAM 

---

A python implementation of the PAM framework. The structure and methodologies are inspired from the HGF toolbox, open source code available as part of the TAPAS software collection. The python implementation of the HGF toolbcx is forked from https://github.com/mesoScopic-Computational-AuditioN-lab/HGF by Jorie van Haren jjg.vanharen@maastrichtuniversity.nl

The repository is structured in the following way: 

- HGF folder: contains the scripts relative to the HGF implementation.
- PAM: contains the scripts relative to the PAM framework
- Demos: contains three notebooks that serves as an example of usage. 

Original reference:

Frässle, S., et al. (2021). TAPAS: An Open-Source Software Package for
Translational Neuromodeling and Computational Psychiatry. Frontiers in
Psychiatry, 12:680811. https://doi.org/10.3389/fpsyt.2021.680811

----

## Dependencies

| Required | Package           | Remarks         |
| ---------|-------------------|-----------------|
| Yes      | [Python 3]        |                 |
| Yes      | [numpy]           |                 |
| Yes      | [statsmodels.api] |                 |
| Yes      | [scipy]           | Opitimization   |
| No       | [pandas]          | Plotting        |
| No       | [seaborn]         | Plotting        |
| No       | [matplotlib]      | Plotting        |

----

## Installation

1. Clone the latest release and unzip it.
2. Change directory in your command line:
```
cd /path/to/HGF
```
3. Create a Virtual Environment
```
python<version> -m venv <virtual-environment-name>
```
4. Activate Virtual Environment
```
.venv\Scripts\activate
```
5. Install dependencies:
```
pip install -r requirments.txt
```
