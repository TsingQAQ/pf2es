# -*- coding: utf-8 -*-
# %% [markdown]
# # Parallel Pareto Frontier Entropy Search

# %% [markdown]
# ## Related Work

# %% [markdown]
# - [MESMO](https://par.nsf.gov/servlets/purl/10145801) paper, NIPS 2019, Belakaria et al
# - [PFES](http://proceedings.mlr.press/v119/suzuki20a.html), ICML 2020, Shinya Suzuki et al
#
#
# Multi-Fidelity Part
# - [MF-OSEMO](file:///C:/Users/Administrator/Downloads/6561-Article%20Text-9786-1-10-20200519.pdf) paper, 2020 AAAI, Belakaria et al
#
# Constraint Part
# - [MESMOC](https://arxiv.org/pdf/2009.01721.pdf) paper, NIPS 2020 Workshop, Belakaria et al
# - [MESMOC+](https://arxiv.org/pdf/2011.01150.pdf) paper, AISTATS, Daniel Fernández-Sánchez (Daniel Hernández-Lobato)
#
# Uncatogrized
# - [iMOCA](https://arxiv.org/pdf/2009.05700.pdf) paper, NIPS 2020 Workshop, Belakaria et al

# %% [markdown]
# -----------

# %% [markdown]
# # Aquisition Function Description

# %% [markdown]
# ## MESMO

# %% [markdown]
# **Note**: this is the oldest MESMO paper:

# %% [markdown]
# Consider minimization, the conditional distribution given pareto front in MESMO is approximated as (recall Eq. 4.10 ):

# %% [markdown]
# \begin{equation}
# \begin{aligned}
# H[\boldsymbol{f}_x \vert D, x, \mathcal{F^*}] &\approx \sum_{j=1}^K H[y^j \vert D, x, max\{z_1^j, ..., z_m^j\}] \quad \text{Independent Assumption on each obj}\\& = \sum_{j=1}^K \left[ \frac{\gamma(\boldsymbol{x})\phi(\gamma)}{2\Phi(\gamma(\boldsymbol{x}))} - ln \Phi(\gamma(\boldsymbol{x})) \right] \quad \text{Same Formulation as Zi Wang's MES}
# \end{aligned}
# \end{equation}

# %% [markdown]
# ## PFES

# %% [markdown]
# \begin{equation}
# \begin{aligned}
# \alpha(x) &= H[\mathcal{F^*} \vert D] - \mathbb{E}_{f_x}H[\mathcal{F^*} \vert D, \{x, \boldsymbol{f}_x\}] \\& = H[\boldsymbol{f}_x\vert D] - \mathbb{E}_{\mathcal{F^*}}H[\boldsymbol{f}_x \vert D, x, \mathcal{F^*}]
# \end{aligned}
# \end{equation}

# %% [markdown]
# MESMO approximation:

# %% [markdown]
# \begin{equation}
# H[\boldsymbol{f}_x \vert D, x, PF] \approx \sum_{j=1}^K H[y^j \vert D, x, max\{z_1^j, ..., z_m^j\}]
# \end{equation}

# %% [markdown]
# Where $\{z_1,..., z_m\}$ are sampled pareto front points

# %% [markdown]
# ---------------

# %% [markdown]
# Recall the definition of acquisition function in PFES paper [1]:
# \begin{equation}
# \alpha(\boldsymbol{x}) = H[p(\boldsymbol{f}_x) \vert D] - \frac{1}{|PF|} \Sigma_{\mathcal{F^*}\in PF} H[p(\boldsymbol{f}_x \vert D, \boldsymbol{f}_x \prec \mathcal{F^*})] \tag{1}
# \end{equation}

# %% [markdown]
# Where $\mathcal{F^*}$ denotes the sampled pareto pront as a discrete approximation of pareto frontier. 

# %% [markdown]
# $p(\boldsymbol{f}_x \vert D, \boldsymbol{f}_x \prec \mathcal{F^*})$ can be rewritten as:  
#
# \begin{equation}
# p(\boldsymbol{f}_x \vert D, \boldsymbol{f}_x \prec \mathcal{F^*}) = \left\{
# \begin{aligned}
# &\frac{1}{Z}p(\boldsymbol{f}_x \vert D)\quad  \{\boldsymbol{f}_x \in R^M: \boldsymbol{f}_x \prec \mathcal{F^*}\}\\
# &0 \quad \quad else\\
# \end{aligned}\tag{2}
# \right.
# \end{equation}  
# Where $Z = \int_\mathcal{F} p(\boldsymbol{f}_x) d\boldsymbol{f}_x$ is the normalization constant, $\mathcal{F}$ is defined as the dominated objective space.

# %% [markdown]
# Assume the dominated space can been partitioned into $M$ cells, with statistical independence (for simplicity derivation, might not necessary) assumption on different obj, we have:  
# \begin{equation}
# \begin{aligned}
# Z & = \sum_{m=1}^{M}\left[\prod_{i=1}^{L}\left(\Phi(\frac{u_m^i - \mu_i(x)}{\sigma_i(x)}) - \Phi(\frac{l_m^i - \mu_i(x)}{\sigma_i(x)})\right)\right] \\& = \sum_{m=1}^{M} Z_m
# \end{aligned} \tag{3}
# \end{equation}
# and 
# \begin{equation}
# Z_m  = \prod_{i=1}^L Z_{m_i} \tag{4}
# \end{equation}
#
# where $u_m^i$, $l_m^i$ denotes the upper and lower bound of cell $m$ at dimension $i$. 

# %% [markdown]
# -------

# %% [markdown]
# ### Deriviation of the differential entropy based on conditional distribution 

# %% [markdown]
# Thorem 3.1 of PFES paper reveals the calculation of $H[p(\boldsymbol{f}_x \vert D, \boldsymbol{f}_x \prec \mathcal{F^*})]$ for a single query points:

# %% [markdown]
# \begin{equation}
# \begin{aligned}
# H[p(\boldsymbol{f}_x \vert D, \boldsymbol{f}_x \prec \mathcal{F^*})] &= - \int_\mathcal{F}  \frac{p(\boldsymbol{f}_x \vert D)}{Z} log \frac{p(\boldsymbol{f}_x \vert D)}{Z} d \boldsymbol{f}_x
# \\ &=  - \int_\mathcal{F}  \frac{p(\boldsymbol{f}_x \vert D)}{Z} log p(\boldsymbol{f}_x \vert D) d \boldsymbol{f}_x + \frac{log Z}{Z} \int_\mathcal{F}  p(\boldsymbol{f}_x \vert D)  d \boldsymbol{f}_x \\ &= -  \frac{1}{Z}\int_\mathcal{F}  p(\boldsymbol{f}_x \vert D) log p(\boldsymbol{f}_x \vert D) d \boldsymbol{f}_x + log Z \\&= -  \frac{1}{Z} \sum_{m=1}^M \int_{\mathcal{F}_m}p(\boldsymbol{f}_x \vert D) log p(\boldsymbol{f}_x \vert D) d \boldsymbol{f}_x + log Z
# \end{aligned} \tag{5}
# \end{equation}

# %% [markdown]
# Define auxilary random variable (truncated normal)  $\boldsymbol{h}_{x_{\mathcal{F}_m}} := \boldsymbol{f}_{x} \cdot I\{\boldsymbol{f}_{x} \in \mathcal{F}_m\}$, then:

# %% [markdown]
# \begin{equation}
# p(\boldsymbol{h}_{x_{\mathcal{F}_m}}) = \left\{
# \begin{aligned}
# &\frac{1}{Z_m}p(\boldsymbol{f}_x \vert D)\quad  \{\boldsymbol{f}_x \in \mathcal{F}_m\}\\
# &0 \quad else\\
# \end{aligned} \tag{6}
# \right.
# \end{equation}

# %% [markdown]
# Then we could write its differential entropy as:
# \begin{equation}
# \begin{aligned}
# \mathbb{H}[\boldsymbol{h}_{x_{\mathcal{F}_m}}] &= -\int_{\mathcal{F}_m} p({h}_{x_{\mathcal{F}_m}})log p({h}_{x_{\mathcal{F}_m}}) d{h}_{x_{\mathcal{F}_m}} \\ &= - \int_{\mathcal{F}_m} \frac{1}{Z_m}p(\boldsymbol{f}_x \vert D)log [\frac{1}{Z_m}p(\boldsymbol{f}_x \vert D)] d\boldsymbol{f}_x  \\&= - \int_{\mathcal{F}_m} \frac{1}{Z_m} p(\boldsymbol{f}_x \vert D)log p(\boldsymbol{f}_x \vert D) d\boldsymbol{f}_x + \frac{1}{Z_m} log Z_m \int_{\mathcal{F}_m}  p(\boldsymbol{f}_x \vert D) d\boldsymbol{f}_x\\& = - \frac{1}{Z_m} \int_{\mathcal{F}_m} p(\boldsymbol{f}_x \vert D)log p(\boldsymbol{f}_x \vert D)d\boldsymbol{f}_x + log Z_m
# \end{aligned} \tag{7}
# \end{equation}
# So we have:
# \begin{equation}
# \int_{\mathcal{F}_m} p(\boldsymbol{f}_x \vert D)log p(\boldsymbol{f}_x \vert D)d\boldsymbol{f}_x = Z_mlog Z_m - Z_m\mathbb{H}[\boldsymbol{h}_{x_{\mathcal{F}_m}}]  \tag{8}
# \end{equation}

# %% [markdown]
# We plug in $\boldsymbol{h}_x$ into the original differential entropy expression, i.e., substitute Eq. 8 into Eq. 5:

# %% [markdown]
# \begin{equation}
# \begin{aligned}
# \mathbb{H}[p(\boldsymbol{f}_x \vert D, \boldsymbol{f}_x \prec \mathcal{F^*})] &= -  \frac{1}{Z} \sum_{m=1}^M \int_{\mathcal{F}_m}p(\boldsymbol{f}_x \vert D) log p(\boldsymbol{f}_x \vert D) d \boldsymbol{f}_x + log Z \\ &=   -  \frac{1}{Z} \sum_{m=1}^M \left[ Z_mlog Z_m - Z_m\mathbb{H}[\boldsymbol{h}_{x_{\mathcal{F}_m}}]  \right]+ log Z 
# \end{aligned}\tag{9}
# \end{equation}

# %% [markdown]
# Where $M$ denotes the partitioned cell total number 

# %% [markdown]
# ----------------------

# %% [markdown]
# ### Single Query Point Case (PFES):

# %% [markdown]
# \begin{equation}
# \begin{aligned}
# H[p(\boldsymbol{f}_x \vert D, \boldsymbol{f}_x \prec \mathcal{F^*})] & =  -  \frac{1}{Z} \sum_{m=1}^M \left[ Z_mlog Z_m - Z_m\mathbb{H}[\boldsymbol{h}_{x_{\mathcal{F}_m}}]  \right]+ log Z  \\ & = - \frac{1}{Z} \sum_{m=1}^M \left[ Z_mlog Z_m  + Z_m\int_{l1}^{u1}\int_{l2}^{u2}...\int_{lL}^{uL} p(\boldsymbol{h}_x)logp(\boldsymbol{h}_x)d\boldsymbol{h}_x\right] + log Z\\ & = - \frac{1}{Z} \sum_{m=1}^M \left[ Z_mlog Z_m  + Z_m \int_{l1}^{u1}\int_{l2}^{u2}...\int_{lL}^{uL} \prod_{i=1}^L p(\boldsymbol{h}_{x_i}) \sum_{i=1}^L logp(\boldsymbol{h}_{x_i})d\boldsymbol{h}_x\right] + log Z\\ & = - \frac{1}{Z} \sum_{m=1}^M \left[ Z_m log Z_m   - Z_m \sum_{i=1}^L \left(  \int_{lj}^{uj}\prod_{j\neq i}^{L}p(\boldsymbol{h}_{x_j}) d\boldsymbol{h}_{x_j}\cdot \underbrace{- \int_{li}^{ui}  p(\boldsymbol{h}_{x_i}) log p(\boldsymbol{h}_{x_i})d\boldsymbol{h}_{x_i}}_{\text{entropy of 1d truncated normal}}\right) \right] + log Z\\ & = - \frac{1}{Z} \sum_{m=1}^M \left[ Z_m log Z_m  - Z_m \sum_{i=1}^L \underbrace{\left(log(\sqrt{2 \pi e}\sigma_i Z_{mi}) + \frac{l_i \phi(l_i) - u_i \phi(u_i)}{2Z_{mi}} \right)}_{\text{1d truncated differential entropy}}  \right]+ log Z \\ & = - \frac{1}{Z} \sum_{m=1}^M \left[ Z_m log Z_m  - Z_m \sum_{i=1}^L log(\sqrt{2 \pi e}\sigma_i Z_{mi})  - Z_m \sum_{i=1}^L \frac{l_i \phi(l_i) - u_i \phi(u_i)}{2Z_{mi}} \right]+ log Z \\&=   \sum_{m=1}^M \left[ \frac{Z_m}{Z} \sum_{i=1}^L \frac{l_i \phi(l_i) - u_i \phi(u_i)}{2Z_{mi}} \right]+ log Z - \frac{1}{Z} \sum_{m=1}^M \left[ Z_m log Z_m  - Z_m log({2 \pi e}^\frac{L}{2} Z_m \prod_{i=1}^L \sigma_i )  \right] \\& = \sum_{m=1}^M \left[ \frac{Z_m}{Z} \sum_{i=1}^L \frac{l_i \phi(l_i) - u_i \phi(u_i)}{2Z_{mi}} \right]+ log Z + \frac{1}{Z} \sum_{m=1}^M \left[ Z_m  log ({2 \pi e})^\frac{L}{2} \prod_{i=1}^L \sigma_i  \right] \\& = \sum_{m=1}^M \left[ \frac{Z_m}{Z} \sum_{i=1}^L \frac{l_i \phi(l_i) - u_i \phi(u_i)}{2Z_{mi}} \right]+  log ({2 \pi e})^\frac{L}{2} Z \prod_{i=1}^L \sigma_i 
# \end{aligned} \tag{10}
# \end{equation}

# %% [markdown]
# Where $L$ denotes the objective numbers, $Z_{mi} = \Phi(\frac{u_m^i - \mu_i(x)}{\sigma_i(x)}) - \Phi(\frac{l_m^i - \mu_i(x)}{\sigma_i(x)})$

# %% [markdown]
# ------------------

# %% [markdown]
# ### Batch Case by GIBBON

# %% [markdown]
# In the most simple case, we assume noise free and single fidelity condition. i.e., $C_i$ = $A_i$. 

# %% [markdown]
# \begin{equation}
# \begin{aligned}
# & H[p(\boldsymbol{f}_\boldsymbol{x} \vert D, \boldsymbol{f}_\boldsymbol{x} \prec \mathcal{F^*})] \\ & = -  \frac{1}{Z} \sum_{m=1}^M \left[ Z_mlog Z_m - Z_m\mathbb{H}[\boldsymbol{h}_{x_{\mathcal{F}_m}}]  \right]+ log Z \quad  (Eq. 9)\\& = -  \frac{1}{Z} \sum_{m=1}^M \left[ Z_mlog Z_m - Z_m \sum_{i=1}^L \mathbb{H}[\boldsymbol{f}_{\boldsymbol{x}}^i \vert \boldsymbol{f}_{\boldsymbol{x}}^i \in \mathcal{F}_m^i]  \right]+ log Z \\& = -  \frac{1}{Z} \sum_{m=1}^M (Z_mlog Z_m)+ log Z + \frac{Z_m}{Z}\sum_{m=1}^M \sum_{i=1}^L \mathbb{H}[\boldsymbol{f}_{\boldsymbol{x}}^i \vert \boldsymbol{f}_{\boldsymbol{x}}^i \in \mathcal{F}_m^i] \\ & \\ &\leq -  \frac{1}{Z} \sum_{m=1}^M (Z_mlog Z_m)+ log Z + \frac{Z_m}{Z}\sum_{m=1}^M \sum_{i=1}^L \sum_{j=1}^B \mathbb{H}[\boldsymbol{f}_{x_j}^i \vert \boldsymbol{f}_{\boldsymbol{x}}^i \in \mathcal{F}_m^i]  \quad{\text{information-theoretic inequality}}\\ & = -  \frac{1}{Z} \sum_{m=1}^M (Z_mlog Z_m)+ log Z + \frac{Z_m}{Z}\sum_{m=1}^M \sum_{i=1}^L \sum_{j=1}^B \mathbb{H}[\boldsymbol{f}_{x_j}^i \vert \boldsymbol{f}_{x_j}^i \in \mathcal{F}_m^i]  \quad {\text{conditional independence}} 
# \end{aligned}\tag{11}
# \end{equation}

# %% [markdown]
# ## Constraints

# %% [markdown]
# Related paper: 
# - [CMES: Constrained Bayesian Optimization with Max-Value Entropy Search](https://arxiv.org/abs/1910.07003)
# - CMES-IBO [Sequential- and Parallel- Constrained Max-value Entropy Search via Information Lower Bound](https://arxiv.org/abs/2102.09788)
#

# %% [markdown]
# $$\boldsymbol{h}_{x}:=(f_1, f_2, ..., f_L, g_1, ...g_C)$$

# %% [markdown]
# We do 2 different derivation: the 1st derivation follows the CMES (and conventional MES) style. The 2nd derivation follows the CMES-IBO, which directly starts from the mutual information. We see if there is a difference on formulation and result.

# %% [markdown]
# ![image info](./A_definition.png)
#
# **Notation/Definition**
# - Denote Feasible Pareto Frontier as $\mathcal{F_{Fea}^*}$
# - $g > 0$ is represented as feasible
# - Let $A \subset R^d$, such that $\forall \{\boldsymbol{f}, \boldsymbol{g}\} \in A: \boldsymbol{f}\succ \mathcal{F}_{Fea}^*$ and $\boldsymbol{g}>0$

# %% [markdown]
# ### CMES Derive

# %% [markdown]
# \begin{equation}
# \begin{aligned}
# \alpha(x) &= H[\mathcal{F_{Fea}^*} \vert D] - \mathbb{E}_{h_x}H[\mathcal{F_{Fea}^*} \vert D, \{x, \boldsymbol{h}_x\}] \\& = H[\boldsymbol{h}_x\vert D] - \mathbb{E}_{\mathcal{F_{Fea}^*}}H[\boldsymbol{h}_x \vert D, x, \mathcal{F_{Fea}^*}] 
# \end{aligned}
# \end{equation}
#
# 1. In case having feasible pareto samples
#
# \begin{equation}
# p(\boldsymbol{h}_x \vert D, x, \mathcal{F}_{Fea}^*)  = \left\{
# \begin{aligned}
# &\frac{p(\boldsymbol{h}_x \vert D, x)}{Z} \quad h_x \in \overline{A}\\
# &0 \quad else\\
# \end{aligned} \tag{12}
# \right.
# \end{equation}
#
# Then we have:
# \begin{equation}
# Z := \int_\overline{A} p(\boldsymbol{h}_x \vert D, x)dx = 1- \int_A p(\boldsymbol{h}_x \vert D, x)dx
# \end{equation}
#
# Now the left is deriving the differential entropy expression in $\overline{A}$. 

# %% [markdown]
# Reuse Eq. 5 with $\overline{A}$: suppose $\overline{A}$ can be partitioned into $\mathcal{F}_m$ cells, we have:

# %% [markdown]
# \begin{equation}
# \begin{aligned}
# H[\boldsymbol{h}_x \vert D, x, \mathcal{F_{Fea}^*}] &= - \int_\overline{A}  \frac{p(\boldsymbol{h}_x \vert D)}{Z} log \frac{p(\boldsymbol{h}_x \vert D)}{Z} d \boldsymbol{h}_x
# \\ &=  - \int_\overline{A}  \frac{p(\boldsymbol{h}_x \vert D)}{Z} log p(\boldsymbol{h}_x \vert D) d \boldsymbol{h}_x + \frac{log Z}{Z} \int_\overline{A}  p(\boldsymbol{h}_x \vert D)  d \boldsymbol{h}_x \\ &=  log Z -  \int_\overline{A}\frac{p(\boldsymbol{h}_x \vert D) }{Z}  log p(\boldsymbol{h}_x \vert D) d \boldsymbol{h}_x \\&= log Z -  \frac{1}{Z} \sum_{m=1}^M \int_{\mathcal{F}_m}p(\boldsymbol{h}_x \vert D) log p(\boldsymbol{h}_x \vert D) d \boldsymbol{h}_x  \quad \text{linearity of expectation}
# \end{aligned} 
# \end{equation}

# %% [markdown]
# Reuse Eq. 6-8, we still have:

# %% [markdown]
# \begin{equation}
# \int_{\mathcal{F}_m} p(\boldsymbol{h}_x \vert D)log p(\boldsymbol{h}_x \vert D)d\boldsymbol{h}_x = Z_mlog Z_m - Z_m\mathbb{H}[\boldsymbol{h}_{x_{\mathcal{F}_m}}]  
# \end{equation}

# %% [markdown]
# We plug in $\boldsymbol{h}_x$ into the original differential entropy expression, i.e., substitute Eq. 8 into Eq. 5:

# %% [markdown]
# \begin{equation}
# \begin{aligned}
# H[\boldsymbol{h}_x \vert D, x, \mathcal{F_{Fea}^*}] &= log Z -  \frac{1}{Z} \sum_{m=1}^M \int_{\mathcal{F}_m}p(\boldsymbol{h}_x \vert D) log p(\boldsymbol{h}_x \vert D) d \boldsymbol{h}_x  \\ &= log Z  -  \frac{1}{Z} \sum_{m=1}^M \left[ Z_mlog Z_m - Z_m\mathbb{H}[\boldsymbol{h}_{x_{\mathcal{F}_m}}]  \right]
# \end{aligned}
# \end{equation}

# %% [markdown]
# Eventually, we have the following expression:

# %% [markdown]
# \begin{equation}
# \begin{aligned}
# H[\boldsymbol{h}_x \vert D, x, \mathcal{F_{Fea}^*}] & = \sum_{m=1}^M \left[ \frac{Z_m}{Z} \sum_{i=1}^{L+C} \frac{l_i \phi(l_i) - u_i \phi(u_i)}{2Z_{mi}} \right]+  log ({2 \pi e})^\frac{L}{2} Z \prod_{i=1}^{L+C} \sigma_i 
# \end{aligned} 
# \end{equation}

# %% [markdown]
# The main difficulty, now we have, is of partitioning $\overline{A}$ to hyper-rectangles again so that we can calculate the box entropy within each of the rectangles (i.e., cells, or $\mathcal{F}_m$). Idealy, $\overline{A}$ could be partitioned into two main chatogories: $\overline{A} = \overline{A_{infea}}\cup \overline{A_{fea}}$, where:
# 1. $\overline{A_{infea}}$ represents at least one of the constraint has been violated (left part of the figure). In this case, there is no constraint of $p(\boldsymbol{f})$'s distribution. However, even leave out $p(\boldsymbol{f})$, $\overline{A_{infea}}$ is not a hyper-rectangle, so the computation is somewhat complicated: a naive partition will result at most $2^C$ cells, again, divide and conqure method could be of use.
# 2. $\overline{A_{fea}}$ represents all of the constraint has been satisfied (lower right part of the figure), in this case, we ask $\boldsymbol{f}_\boldsymbol{x}$ must be dominated by current pareto frontier, this is exactly the same as PFES expression. In this case, the expression is much more easier: just an extention of equation given above.

# %% [markdown]
# ----------------------

# %% [markdown]
# 2. In case having **no feasible** pareto samples
# $F_{Fea}^*:= \boldsymbol{g}_{fea}: = \text{reference point (or even -inf)}$  
# This can result the acquisition function to work similar like PoF

# %% [markdown]
# ### CMES-IBO Start

# %% [markdown]
# Here: Inspired by CMES IBO, we try to derive a version of CPFES-IBO, the main difference of CMES-IBO derive fashion compared with the one above, is that it directly starts from mutual information and provide the approximation of the acquisition function as an lower bound of it, hence, claimed as "*rigorous lower bound*" in the original paper. We start with this sort of derivation and see if there are any useful insight we could gain from it. 

# %% [markdown]
# CPFES-IBO starst directly by selecting the next point to maximize the mutual information between the query $\boldsymbol{h}_*$ and constraint Pareto frontier $\mathcal{F_{Fea}^*}$:  

# %% [markdown]
# \begin{equation}
# \begin{aligned}
# \alpha(x) &= H[\mathcal{F_{Fea}^*} \vert D] - \mathbb{E}_{h_x}H[\mathcal{F_{Fea}^*} \vert D, \{x, \boldsymbol{h}_x\}] \\& = H[\boldsymbol{h}_x\vert D] - \mathbb{E}_{\mathcal{F_{Fea}^*}}H[\boldsymbol{h}_x \vert D, x, \mathcal{F_{Fea}^*}] d\boldsymbol{h}_x\} d \mathcal{F_{Fea}^*}
# \end{aligned}
# \end{equation}

# %% [markdown]
# \begin{equation}
# \begin{aligned}
# I(\mathcal{F_{Fea}^*}; \boldsymbol{h}_* ) :&= \int_{\mathcal{F_{Fea}^*}} \int_{\boldsymbol{h}_x} p(\boldsymbol{h}_x, \mathcal{F_{Fea}^*})\text{log} \frac{ p(\boldsymbol{h}_x, \mathcal{F_{Fea}^*})}{p(\boldsymbol{h}_x)p(\mathcal{F_{Fea}^*})}d\mathcal{F_{Fea}^*}d\boldsymbol{h}_x \\ & = \int_{\mathcal{F_{Fea}^*}} p(\mathcal{F_{Fea}^*}) \left[\int_{\boldsymbol{h}_x} p(\boldsymbol{h}_x \vert \mathcal{F_{Fea}^*})\text{log} \frac{ p(\boldsymbol{h}_x\vert F_{Fea}^*)}{p(\boldsymbol{h}_x)}d\boldsymbol{h}_x\right] d\mathcal{F_{Fea}^*} \\ & = \mathbb{E}_{\mathcal{F_{Fea}^*}}\left[\int_{\boldsymbol{h}_x}p(\boldsymbol{h}_x \vert \mathcal{F_{Fea}^*})\text{log} \frac{ p(\boldsymbol{h}_x\vert F_{Fea}^*)}{p(\boldsymbol{h}_x)}d\boldsymbol{h}_x\right] \text{Not Analytical}\\ & = \mathbb{E}_{\mathcal{F_{Fea}^*}}\left[\int_{\boldsymbol{h}_x}p(\boldsymbol{h}_x \vert \mathcal{F_{Fea}^*})\text{log} \frac{ p(\boldsymbol{h}_x\vert F_{Fea}^*)q(\boldsymbol{h}_x)}{p(\boldsymbol{h}_x)q(\boldsymbol{h}_x)}d\boldsymbol{h}_x\right]  \text{VI trick}\\ & = \mathbb{E}_{\mathcal{F_{Fea}^*}}\left[\int_{\boldsymbol{h}_x}p(\boldsymbol{h}_x \vert \mathcal{F_{Fea}^*})\text{log} \frac{ q(\boldsymbol{h}_x)}{p(\boldsymbol{h}_x)}d\boldsymbol{h}_x + D_{KL}(p(\boldsymbol{h}_x \vert \mathcal{F_{Fea}^*}) || q(\boldsymbol{h}_x))\right]  \\ & \geq \mathbb{E}_{\mathcal{F_{Fea}^*}}\left[\int_{\boldsymbol{h}_x}p(\boldsymbol{h}_x \vert \mathcal{F_{Fea}^*})\text{log} \frac{ q(\boldsymbol{h}_x)}{p(\boldsymbol{h}_x)}d\boldsymbol{h}_x \right]  \text{Lower Bound}
# \end{aligned}
# \end{equation}

# %% [markdown]
# Once $q(\boldsymbol{h}_x)$ exactly equals $p(\boldsymbol{h}_x \vert \mathcal{F_{Fea}^*})$ the Lower Bound is tight.  

# %% [markdown]
# **The approximation of $p(\boldsymbol{h}_x \vert \mathcal{F_{Fea}^*})$**

# %% [markdown]
# Here we reuse the approximation of Eq. 12 for $q(\boldsymbol{h}_x)$ as, we can have:

# %% [markdown]
# \begin{equation}
# \begin{aligned}
#     I(\mathcal{F_{Fea}^*}; \boldsymbol{h}_* ) &\approx \mathbb{E}_{\mathcal{F_{Fea}^*}}\left[\int_{\boldsymbol{h}_x}p(\boldsymbol{h}_x \vert \mathcal{F_{Fea}^*})\text{log} \frac{ q(\boldsymbol{h}_x)}{p(\boldsymbol{h}_x)}d\boldsymbol{h}_x \right] \\ & = \mathbb{E}_{\mathcal{F_{Fea}^*}}\left[\int_{\boldsymbol{h}_x}p(\boldsymbol{h}_x \vert \mathcal{F_{Fea}^*})\text{log} \frac{p(\boldsymbol{h}_x ) }{Zp(\boldsymbol{h}_x)}d\boldsymbol{h}_x \right] 
# \end{aligned}
# \end{equation}

# %% [markdown]
# By making use of the fact that $\text{lim}_{x \rightarrow 0} x log x = 0$, we can skip the part that $\boldsymbol{h}_x \in A$, hence result the following:

# %% [markdown]
# \begin{equation}
# \begin{aligned}
#     I(\mathcal{F_{Fea}^*}; \boldsymbol{h}_* ) & \approx \mathbb{E}_{\mathcal{F_{Fea}^*}}\left[\text{log} \frac{1 }{Z} \int_{\boldsymbol{h}_x \in \overline{A}}p(\boldsymbol{h}_x \vert \mathcal{F_{Fea}^*})\right]  \\ &= -\mathbb{E}_{\mathcal{F_{Fea}^*}}\left[\text{log} Z \right] \quad \text{Note no approx for}\ p(\boldsymbol{h}_x \vert \mathcal{F_{Fea}^*})\ \text{made} \\ & = -\mathbb{E}_{\mathcal{F_{Fea}^*}}\left[\text{log}\left(1 - \int_A p(\boldsymbol{h}_x \vert D, x)dx \right)\right] \\ & =  - \mathbb{E}_{  \mathcal{F_{Fea}^*}}\left[\text{log}\left(1 -\text{Pr}(\boldsymbol{f}_x \succ \mathcal{F_{Fea}^*})\prod_{c=1}^C PoF_c \right)\right] \\ & \approx - \frac{1}{K}\Sigma_{i=1}^K \left[\text{log}\left(1 -\text{Pr}(\boldsymbol{f}_x \succ \mathcal{F_{{Fea}_i}^*})\prod_{c=1}^C PoF_c \right)\right]
# \end{aligned}\tag{13}
# \end{equation}

# %% [markdown]
# Note: 
# - (Work on non-constraint version) Eq. 13 also works with non-constraint version by simply removing the right product part
# - (More easy handling with Partition): Eq. 13 needs partition of non-dominated region, hence any of existing partition based method is useful.
# - (Computation Complexity): is $\text{Pr}(\boldsymbol{f}_x \succ \mathcal{F_{{Fea}}^*})$ able to be approxiated by rejecting sampling? (Not exact, but in this case we may have lower computation complexity)

# %% [markdown]
# #### Links with existing acquisition function

# %% [markdown]
# Besides the constraint, we would like to elaborate some link with existing multi-objective based acquisition function as well.

# %% [markdown]
# From Eq. 13, So the inner part is just the PI plus the PoF product, in case $\mathcal{F_{Fea}}$ has been approximated by Pareto points and the a hypervolume based partition has been done there, this can deemded exactly as an extended version of HVPI

# %% [markdown]
# ## Decoumposed Setting

# %% [markdown]
# In case 

# %% [markdown]
# ## Parallaization

# %% [markdown]
# Here, we mainly try to derive from the conditional mutual information, motivated from the MES-IBO paper

# %% [markdown]
# Denote $H_q = {\boldsymbol{h}_1, ..., \boldsymbol{h}_q}$

# %% [markdown]
# \begin{equation}
#  MI(F_{Fea}^*, H_q) = MI(F_{Fea}^*, H_{q-1}) + CMI(\boldsymbol{h}_q ; F_{Fea}^* \vert H_{q-1})
# \end{equation}

# %% [markdown]
# ## Implementation Details

# %% [markdown]
# ### Visual Check of partition on dominated region

# %% [markdown]
# ##### 2D Case

# %%
# %matplotlib notebook

# %%
import tensorflow as tf
from matplotlib import pyplot as plt
from trieste.acquisition.multi_objective.partition import ExactPartition2dDominated
from trieste.acquisition.multi_objective.pareto import Pareto
import matplotlib.patches as patches

objs = tf.constant([[1, 0], [0.3, 0.7], [0.7, 0.3], [0., 1.]])

lb, ub = ExactPartition2dDominated(objs).partition_bounds(tf.constant([-10., -10.]), tf.constant([3., 3.]))
fig, ax = plt.subplots()
for i in range(lb.shape[0]):
    rect = patches.Rectangle(lb[i], (ub - lb)[i, 0], (ub - lb)[i, 1], linewidth=1, edgecolor='r')
    ax.add_patch(rect)
plt.scatter(Pareto(objs).front[:, 0], Pareto(objs).front[:, 1], 10, label='Pareto Front')
plt.scatter(tf.constant([3., 3.])[0], tf.constant([3., 3.])[1], 10, label='Worst Point')
plt.legend(fontsize=15)
plt.title('2D Demo of partition the dominated region')
plt.show()

# %% [markdown]
# ##### 3D Case

# %%
# %matplotlib notebook

# %%
import tensorflow as tf
from trieste.acquisition.multi_objective.partition import FlipTrickPartitionNonDominated
objs = tf.constant([[1.0, 0.0, 2.0], [-1.0, 1.0, 3.0], [1.5, -1.0, 2.5]])
# objs = tf.random.normal(shape=(5, 3))
ideal_point = tf.constant([-2.0, -1.0, -1.0])
worst_point = tf.constant([5.0, 5.0, 5.0])
lb, ub = FlipTrickPartitionNonDominated(objs, ideal_point, worst_point).partition_bounds()

# %%
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import matplotlib.pyplot as plt

def cuboid_data2(o, size=(1,1,1)):
    X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    X = np.array(X).astype(float)
    for i in range(3):
        X[:,:,i] *= size[i]
    X += np.array(o)
    return X

def plotCubeAt2(positions,sizes=None,colors=None, alpha=None, **kwargs):
    if not isinstance(colors,(list,np.ndarray)): colors=["C0"]*len(positions)
    if not isinstance(sizes,(list,np.ndarray)): sizes=[(1,1,1)]*len(positions)
    g = []
    for p,s,c in zip(positions,sizes,colors):
        g.append( cuboid_data2(p, size=s) )
    return Poly3DCollection(np.concatenate(g),  
                            facecolors=np.repeat(colors,6), alpha=alpha, **kwargs)

fig = plt.figure()
ax = fig.gca(projection='3d')
for pos, size in zip(lb.numpy().tolist(), (ub-lb).numpy().tolist()):
    pc = plotCubeAt2([pos],[size], edgecolor="k", alpha=0.2, colors='#1f77b4')
    ax.add_collection3d(pc)    
ax.scatter(Pareto(objs).front[:, 0], Pareto(objs).front[:, 1], Pareto(objs).front[:, 2], s=20, color='r', label='Pareto Front')
ax.scatter(*worst_point, s=20, color='g', label='Worst Point')
plt.title('3D Demo of partition the dominated region')
plt.legend()
plt.show()

# %% [markdown]
# ### Paper plot usage

# %%
#ffe699

# %%
# %matplotlib notebook

# %%
import tensorflow as tf
from trieste.objectives.multi_objectives import VLMOP2
from trieste.acquisition.multi_objective.partition import HypervolumeBoxDecompositionIncrementalDominated, FlipTrickPartitionNonDominated
from trieste.acquisition.multi_objective import Pareto
# objs = VLMOP2().gen_pareto_optimal_points(5)
objs = tf.constant(
      [[0.9460075 , 0.        ],
       [0.9460075, 0.22119921],
       [0.83212055, 0.43212055],
       [0.22119921, 1.0],
       [0.        , 1.5 ]], dtype=tf.float32)
# true_obsrv = tf.concat([objs, tf.constant([[0.3], [0.4], [0.5], [0.0], [0.8]])], 1)
true_obsrv = tf.concat([objs, tf.constant([[0.4], [0.4], [0.5], [0.0], [0.0]])], 1)
projected_feasible_obsrv = tf.concat([objs, tf.zeros(shape=(5, 1))], 1)

worst_point = tf.constant([3.0, 3.0, 1.0])
lb, ub = HypervolumeBoxDecompositionIncrementalDominated(projected_feasible_obsrv, worst_point).partition_bounds()

# %%
objs

# %%
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import FancyArrowPatch
import numpy as np
import matplotlib.pyplot as plt


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)
        
        
def cuboid_data2(o, size=(1,1,1)):
    X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    X = np.array(X).astype(float)
    for i in range(3):
        X[:,:,i] *= size[i]
    X += np.array(o)
    return X

def plotCubeAt2(positions,sizes=None,colors=None, alpha=None, **kwargs):
    if not isinstance(colors,(list,np.ndarray)): colors=["C0"]*len(positions)
    if not isinstance(sizes,(list,np.ndarray)): sizes=[(1,1,1)]*len(positions)
    g = []
    for p,s,c in zip(positions,sizes,colors):
        g.append( cuboid_data2(p, size=s) )
    return Poly3DCollection(np.concatenate(g),  
                            facecolors=np.repeat(r'#ffe699',6), alpha=alpha, **kwargs)

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


# produce figure
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d', proj_type = 'ortho')
arrow_prop_dict = dict(mutation_scale=20, arrowstyle='-|>', color='k', shrinkA=0, shrinkB=0)

# ax.text(1, 1, 1.8, r'Area:$A$', fontsize=20)
# add feasible surface
x = np.linspace(-1,3,10)
y = np.linspace(-1,3,10)
X,Y = np.meshgrid(x,y)
Z=0.0 * X + 0.0 * Y
surf = ax.plot_surface(X, Y, Z, alpha=0.1)


for pos, size in zip(lb.numpy().tolist(), (ub-lb).numpy().tolist()):
    pc = plotCubeAt2([pos],[size], edgecolor='k', alpha=0.2, colors=r'#ffe699') # #1f77b4
    ax.add_collection3d(pc)

true_obsrv =  tf.constant(
       [[0.9460075, 0.43212055, 0.4],
       [0.83212055, 1.0, 0.5],
       [0.22119921     , 1.5 , 0.0]], dtype=tf.float32)
projected_feasible_obsrv = tf.concat([true_obsrv[..., :-1], tf.zeros(shape=(3, 1))], 1)
ax.scatter(true_obsrv[:, 0], true_obsrv[:, 1], true_obsrv[:, 2], s=100, color='g', label='Observations', zorder=10)
ax.scatter(projected_feasible_obsrv[:, 0], projected_feasible_obsrv[:, 1], 
           projected_feasible_obsrv[:, 2], s=100, color='tab:red', label='Feasible Pareto Frontier', marker="*", 
          edgecolors='k',linewidths=1, zorder=12)
# ax.scatter(Pareto(projected_feasible_obsrv).front[1:-1, 0], Pareto(projected_feasible_obsrv).front[1:-1, 1], 
#            Pareto(projected_feasible_obsrv).front[1:-1, 2], s=50, color='tab:red', label='Feasible Pareto Frontier', marker="*", 
#           edgecolors='k',linewidths=1)
ax.scatter(worst_point[0], worst_point[1], worst_point[2], s=100, color='tab:cyan', label='Ideal Point', marker="^", 
          edgecolors='k',linewidths=1)
ax.scatter(0, 0, -1, s=100, color='tab:cyan', label='Anti Ideal Point', marker="s", 
          edgecolors='k',linewidths=1)

## constraint axes
a = Arrow3D([-0.5, -0.5], [-0.5, -0.5], [0.0, 3], **arrow_prop_dict)
ax.add_artist(a)
b = Arrow3D([-0.5, 2.0], [-0.5, -0.5], [0.0, 0], **arrow_prop_dict)
ax.add_artist(b)
c = Arrow3D([-0.5, -0.5], [-0.5, 2.0], [0.0, 0], **arrow_prop_dict)
ax.add_artist(c)
# text_options = {'horizontalalignment': 'center',
#                 'verticalalignment': 'center',
#                 'fontsize': 14}
ax.text(1.5, -1, -0.9, r'Objective 2', (1, 0, 0), fontsize=15)
ax.text(-1, 0.0, 0.0, r'Objective 1', (0, 1, 0), fontsize=15)
ax.text(-0.5, -0.3, 0.5, r'Constraint', (0, 0, 1), fontsize=15, zorder=20)

line1_xs = [0.9460075, 0.0]
line1_ys = [0.0, 0.0]
line1_zs = [0.0, 0.0]
ax.plot(line1_xs, line1_ys, line1_zs, color='k', linestyle="--")

line2_xs = [0.0, 0.0]
line2_ys = [0.0, 1.5]
line2_zs = [0.0, 0.0]
ax.plot(line2_xs, line2_ys, line2_zs, color='k',  linestyle="--")

line3_xs = [0.0, 0.0]
line3_ys = [0.0, 0.0]
line3_zs = [0.0, -1.0]
ax.plot(line3_xs, line3_ys, line3_zs, color='k',  linestyle="--")
# make the panes transparent
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# make the grid lines transparent
ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.set_box_aspect([1,1,1])
# 
# ax.scatter(*worst_point, s=20, color='g', label='Worst Point')
plt.axis('off')
ax.set_zlim([-2, 4])
# ax.view_init(elev=-27, azim=44)
ax.view_init(elev=-23, azim=52)
ax.dist = 6.5
ax.legend(bbox_to_anchor=(1.1, 1.05), fontsize=30)
plt.tight_layout()
plt.show()
# plt.savefig('Legend.png', dpi=1000)

# %%
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def Rx(phi):
    return np.array([[1, 0, 0],
                     [0, np.cos(phi), -np.sin(phi)],
                     [0, np.sin(phi), np.cos(phi)]])

def Ry(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])

def Rz(psi):
    return np.array([[np.cos(psi), -np.sin(psi), 0],
                     [np.sin(psi), np.cos(psi), 0],
                     [0, 0, 1]])

# define origin
o = np.array([0,0,0])

# define ox0y0z0 axes
x0 = np.array([1,0,0])
y0 = np.array([0,1,0])
z0 = np.array([0,0,1])

# define ox1y1z1 axes
psi = 20 * np.pi / 180
x1 = Rz(psi).dot(x0)
y1 = Rz(psi).dot(y0)
z1 = Rz(psi).dot(z0)

# define ox2y2z2 axes
theta = 10 * np.pi / 180
x2 = Rz(psi).dot(Ry(theta)).dot(x0)
y2 = Rz(psi).dot(Ry(theta)).dot(y0)
z2 = Rz(psi).dot(Ry(theta)).dot(z0)

# define ox3y3z3 axes
phi = 30 * np.pi / 180
x3 = Rz(psi).dot(Ry(theta)).dot(Rx(phi)).dot(x0)
y3 = Rz(psi).dot(Ry(theta)).dot(Rx(phi)).dot(y0)
z3 = Rz(psi).dot(Ry(theta)).dot(Rx(phi)).dot(z0)

# produce figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
arrow_prop_dict = dict(mutation_scale=20, arrowstyle='-|>', color='k', shrinkA=0, shrinkB=0)


# plot ox1y1z1 axes
a = Arrow3D([o[0], x1[0]], [o[1], x1[1]], [o[2], x1[2]], **arrow_prop_dict)
ax.add_artist(a)
a = Arrow3D([o[0], y1[0]], [o[1], y1[1]], [o[2], y1[2]], **arrow_prop_dict)
ax.add_artist(a)
a = Arrow3D([o[0], z1[0]], [o[1], z1[1]], [o[2], z1[2]], **arrow_prop_dict)
ax.add_artist(a)


# show figure
ax.view_init(elev=-150, azim=60)
ax.set_axis_off()
plt.show()

# %% [markdown]
# ### Constraint Pareto Frontier illustrations

# %%
from trieste.objectives.multi_objectives import DTLZ2

objs = DTLZ2(4, 3).gen_pareto_optimal_points(20)
# objs = tf.random.normal(shape=(5, 3))
ideal_point = tf.constant([-0.5, -0.5, -0.5])
worst_point = tf.constant([1.5, 1.5, 1.5])
lb, ub = PFES_Pareto(objs).hypercell_bounds(ideal_point, worst_point)

# %%
fig = plt.figure()
ax = fig.gca(projection='3d')
for pos, size in zip(lb.numpy().tolist(), (ub-lb).numpy().tolist()):
    pc = plotCubeAt2([pos],[size], edgecolor="k", alpha=0.2, colors='#1f77b4')
    ax.add_collection3d(pc)    
ax.scatter(PFES_Pareto(objs).fronts[:, 0], PFES_Pareto(objs).fronts[:, 1], PFES_Pareto(objs).fronts[:, 2], s=20, color='r', label='Pareto Front')
ax.scatter(*worst_point, s=20, color='g', label='Worst Point')
plt.title('3D Demo of partition the dominated region')
plt.legend()
plt.show()

# %% [markdown]
# --------------

# %% [markdown]
# ### Speed Test of Divided and Conqure method

# %% [markdown]
# Here, we investigate the speed of divided con conqure w.r.t dimensionality and number of pareto frontiers. The Pareto frontier is generated from DTLZ1 for simplicity, this shoudn't affect too much on the performance of divided and conqure. We test the different implementations from Trieste & GPFlowOpt & Botorch 

# %% [markdown]
# #### Trieste Divided Conqure Ver

# %%
from trieste.objectives.multi_objectives import DTLZ1
import tensorflow as tf
from trieste.utils.pareto import Pareto

# %%
from time import time
time_profile = {}
exp_repeat = 5
for pf_size in np.arange(10, 100, 10):
    print(f'pf_size: {pf_size}')
    time_profile[str(pf_size)] = []
    for repeat in range(exp_repeat):
        pf = tf.cast(DTLZ1(4, 3).gen_pareto_optimal_points(pf_size), dtype=tf.float64)
        best = tf.constant([-100, -100, -100], dtype=tf.float64)
        worst = tf.constant([1e3, 1e3, 1e3], dtype=tf.float64)
        start = time()
        _ = Pareto(pf)
        end = time()
        time_profile[str(pf_size)].append(end - start)

# %% [markdown]
# #### Trieste Flip Trick Partition Ver

# %%
from trieste.objectives.multi_objectives import DTLZ1
import tensorflow as tf
from trieste.utils.pareto import Pareto

from time import time
time_profile = {}
exp_repeat = 5
for pf_size in np.arange(10, 100, 10):
    print(f'pf_size: {pf_size}')
    time_profile[str(pf_size)] = []
    for repeat in range(exp_repeat):
        pf = tf.cast(DTLZ1(4, 3).gen_pareto_optimal_points(pf_size), dtype=tf.float64)
        best = tf.constant([-100, -100, -100], dtype=tf.float64)
        worst = tf.constant([1e3, 1e3, 1e3], dtype=tf.float64)
        start = time()
        _ = Pareto(pf)
        end = time()
        time_profile[str(pf_size)].append(end - start)


# %% [markdown]
# #### GPFlowOpt Ver

# %% [markdown]
# We note GPFlowopt is based on tensorflow 1 and is not campatible with the Trieste env, **the following code is runned in another conda env prepared for GPFlowOpt1.**

# %%
def dtlz1_gen_pareto_optimal_points(num_points):
    rnd = np.random.uniform(0, 1, size=[num_points, 3 - 1])
    strnd = np.sort(rnd, axis=-1)
    strnd = np.concatenate([np.zeros([num_points, 1]), strnd, np.ones([num_points, 1])], axis=-1)
    return 0.5 * (strnd[..., 1:] - strnd[..., :-1])


# %%
from gpflowopt.pareto import Pareto

# %%
from time import time
time_profile = {}
exp_repeat = 5
for pf_size in np.arange(10, 100, 10):
    print(f'pf_size: {pf_size}')
    time_profile[str(pf_size)] = []
    for repeat in range(exp_repeat):
        pf = dtlz1_gen_pareto_optimal_points(pf_size)
        worst = np.array([1e3, 1e3, 1e3])
        start = time()
        _ = Pareto(pf).hypervolume(worst)
        end = time()
        time_profile[str(pf_size)].append(end - start)

# %%
import pickle
with open('../../notebooks/gpflowopt_dc_profile.pkl', 'wb') as fp:
    pickle.dump(time_profile, fp)

# %%
import pickle
file = open('../../notebooks/gpflowopt_dc_profile.pkl', 'rb')
time_profile_gpflowopt = pickle.load(file)

# %% [markdown]
# ------------------------------

# %% [markdown]
# #### Botorch Ver

# %% [markdown]
# We also illustrate the performance through Botorch, to see if this is an issue of Trieste Implementation or not

# %%
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning, FastNondominatedPartitioning
from botorch.test_functions.multi_objective import DTLZ1
import torch

# %%
from time import time
time_profile_botorch = {}
exp_repeat = 2
for pf_size in np.arange(10, 100, 10):
    print(f'pf_size: {pf_size}')
    time_profile_botorch[str(pf_size)] = []
    for repeat in range(exp_repeat):
        pf = DTLZ1(4, 3).gen_pareto_front(pf_size)
        start = time()
        _ = NondominatedPartitioning(-1e3 * torch.ones(3), pf).get_hypercell_bounds()
        end = time()
        time_profile_botorch[str(pf_size)].append(end - start)

# %%
from time import time
time_profile_botorch_wfg = {}
exp_repeat = 5
for pf_size in np.arange(10, 100, 10):
    print(f'pf_size: {pf_size}')
    time_profile_botorch_wfg[str(pf_size)] = []
    for repeat in range(exp_repeat):
        pf = DTLZ1(4, 3).gen_pareto_front(pf_size)
        start = time()
        _ = FastNondominatedPartitioning(-1e3 * torch.ones(3), pf).get_hypercell_bounds()
        end = time()
        time_profile_botorch_wfg[str(pf_size)].append(end - start)

# %%
from matplotlib import pyplot as plt
plt.figure()
plt.errorbar(np.arange(10, 100, 10), [np.mean(val) for val in time_profile_gpflowopt.values()], [np.std(val) for val in time_profile_gpflowopt.values()], label='GPFlowOpt Divide & Conqure')
plt.errorbar(np.arange(10, 100, 10), [np.mean(val) for val in time_profile.values()], [np.std(val) for val in time_profile.values()], label='Trieste Divide & Conqure')

plt.errorbar(np.arange(10, 100, 10), [np.mean(val) for val in time_profile_botorch.values()], 
             [np.std(val) for val in time_profile_botorch.values()], label='Botorch Divide & Conqure')
plt.errorbar(np.arange(10, 100, 10), [np.mean(val) for val in time_profile_botorch_wfg.values()], 
             [np.std(val) for val in time_profile_botorch_wfg.values()], label='Botorch Lacour17\'s method' )
plt.xlabel('PF Size')
plt.ylabel('Time: (Sec)')
plt.title('Profile of partition method time for different pareto size with 3obj')
# plt.yscale('log')
# plt.yticks([1e0, 1e1, 1e2])
plt.legend()
# plt.yticks([1e0, 1e2])
plt.show()

# %% [markdown]
# ## Details investigation of different acquisition functions

# %% [markdown]
# In this detail exp, we investigate the performance of each 

# %% [markdown]
# ### Checking the convergence property of PF samples by RFF

# %% [markdown]
# ##### VLMOP2 

# %% [markdown]
# We investigate whether PF is getting converging w.r.t number of datas

# %% [markdown]
# Sample of Pareto frontier

# %%
num_initial_points = 20

# %%
import math
import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from trieste.experimental.plotting import (
    plot_bo_points,
    plot_function_2d,
    plot_mobo_history,
    plot_mobo_points_in_obj_space,
    plot_gp_2d,
    plot_acq_function_2d,
)

import trieste
from trieste.acquisition.rule import OBJECTIVE
from trieste.data import Dataset
from trieste.models import create_model
from trieste.models.interfaces import ModelStack
from trieste.space import Box
from trieste.objectives.multi_objectives import VLMOP2
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.models.gpflow import GPflowModelConfig
from trieste.acquisition.multi_objective import Pareto

np.random.seed(1793)
tf.random.set_seed(1793)

vlmop2 = VLMOP2().objective()
observer = trieste.objectives.utils.mk_observer(vlmop2, OBJECTIVE)

mins = [-2, -2]
maxs = [2, 2]
search_space = Box(mins, maxs)
num_objective = 2
input_dim = 2

initial_query_points = search_space.sample(num_initial_points)
initial_data = observer(initial_query_points)

300
def build_stacked_independent_objectives_model(data: Dataset, num_output) -> ModelStack:
    gprs = []
    for idx in range(num_output):
        single_obj_data = Dataset(data.query_points, tf.gather(data.observations, [idx], axis=1))
        variance = tf.math.reduce_variance(single_obj_data.observations)
        kernel = gpflow.kernels.RBF(variance, lengthscales=[1.0] * input_dim)
        kernel.lengthscales.prior = tfd.LogNormal(tf.math.log(tf.constant([1] * input_dim, dtype=tf.float64)),
                                                  tf.constant([0.7], dtype=tf.float64))
        # jitter = gpflow.kernels.White(1e-12)
        gpr = gpflow.models.GPR((single_obj_data.query_points, single_obj_data.observations), kernel,
                                noise_variance=1e-5)
        gpflow.utilities.set_trainable(gpr.likelihood, False)
        gprs.append((create_model(GPflowModelConfig(
                        **{
                            "model": gpr,
                            "optimizer": gpflow.optimizers.Scipy(),
                            "optimizer_args": {"minimize_args": {"options": dict(maxiter=100)}},
                        }
                    )), 1))

    return ModelStack(*gprs)


models = {
    OBJECTIVE: build_stacked_independent_objectives_model(initial_data[OBJECTIVE], num_objective)
}
models[OBJECTIVE].optimize(initial_data[OBJECTIVE])

# %%
from trieste.acquisition.multi_objective.mo_utils import sample_pareto_fronts_from_parametric_gp_posterior
from matplotlib import pyplot as plt
pf_samples = sample_pareto_fronts_from_parametric_gp_posterior(models[OBJECTIVE], initial_data, 1, search_space, num_moo_iter=300)
for pf in pf_samples:
    plt.scatter(pf[:, 0], pf[:, 1])
plt.scatter(Pareto(initial_data[OBJECTIVE].observations).front[:, 0], Pareto(initial_data[OBJECTIVE].observations).front[:, 1], 
            marker='X', s=100)

# plot PF as reference


# %% [markdown]
# ##### BraninCurrin

# %% [markdown]
# We investigate whether PF is getting converging w.r.t number of datas

# %% [markdown]
# Sample of Pareto frontier

# %%
num_initial_points = 30

# %%
import math
import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from trieste.experimental.plotting import (
    plot_bo_points,
    plot_function_2d,
    plot_mobo_history,
    plot_mobo_points_in_obj_space,
    plot_gp_2d,
    plot_acq_function_2d,
)

import trieste
from trieste.acquisition.rule import OBJECTIVE
from trieste.data import Dataset
from trieste.models import create_model
from trieste.models.interfaces import ModelStack
from trieste.space import Box
from trieste.objectives.multi_objectives import BraninCurrin
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.models.gpflow import GPflowModelConfig

np.random.seed(1793)
tf.random.set_seed(1793)

bc = BraninCurrin().objective()
observer = trieste.objectives.utils.mk_observer(bc, OBJECTIVE)

mins = [0.0, 0.0]
maxs = [1.0, 1.0]
search_space = Box(mins, maxs)
num_objective = 2
input_dim = 2

initial_query_points = search_space.sample(num_initial_points)
initial_data = observer(initial_query_points)


def build_stacked_independent_objectives_model(data: Dataset, num_output) -> ModelStack:
    gprs = []
    for idx in range(num_output):
        single_obj_data = Dataset(data.query_points, tf.gather(data.observations, [idx], axis=1))
        variance = tf.math.reduce_variance(single_obj_data.observations)
        kernel = gpflow.kernels.RBF(variance, lengthscales=[1.0] * input_dim)
        kernel.lengthscales.prior = tfd.LogNormal(tf.math.log(tf.constant([1] * input_dim, dtype=tf.float64)),
                                                  tf.constant([0.7], dtype=tf.float64))
        # jitter = gpflow.kernels.White(1e-12)
        gpr = gpflow.models.GPR((single_obj_data.query_points, single_obj_data.observations), kernel,
                                noise_variance=1e-5)
        gpflow.utilities.set_trainable(gpr.likelihood, False)
        gprs.append((create_model(GPflowModelConfig(
                        **{
                            "model": gpr,
                            "optimizer": gpflow.optimizers.Scipy(),
                            "optimizer_args": {"minimize_args": {"options": dict(maxiter=100)}},
                        }
                    )), 1))

    return ModelStack(*gprs)


models = {
    OBJECTIVE: build_stacked_independent_objectives_model(initial_data[OBJECTIVE], num_objective)
}
models[OBJECTIVE].optimize(initial_data[OBJECTIVE])

# %%
from trieste.acquisition.multi_objective.mo_utils import sample_pareto_fronts_from_parametric_gp_posterior
from matplotlib import pyplot as plt
pf_samples = sample_pareto_fronts_from_parametric_gp_posterior(models[OBJECTIVE], initial_data, 50, search_space, num_moo_iter=300)
for pf in pf_samples:
    plt.scatter(pf[:, 0], pf[:, 1])
plt.scatter(Pareto(initial_data[OBJECTIVE].observations).front[:, 0], Pareto(initial_data[OBJECTIVE].observations).front[:, 1], 
            marker='X', s=100)

# plot PF as reference


# %%
pf_samples[4]

# %% [markdown]
# ### Checking the acquisition function value at for different PF samples

# %% [markdown]
# #### Unconstraint Case

# %%
num_initial_points = 20

import math
import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from trieste.experimental.plotting import (
    plot_bo_points,
    plot_function_2d,
    plot_mobo_history,
    plot_mobo_points_in_obj_space,
    plot_gp_2d,
    plot_acq_function_2d,
)

import trieste
from trieste.acquisition.rule import OBJECTIVE
from trieste.data import Dataset
from trieste.models import create_model
from trieste.models.interfaces import ModelStack
from trieste.space import Box
from trieste.objectives.multi_objectives import VLMOP2
from trieste.models.gpflow import GaussianProcessRegression

np.random.seed(1793)
tf.random.set_seed(1793)

vlmop2 = VLMOP2().objective()
observer = trieste.objectives.utils.mk_observer(vlmop2, OBJECTIVE)

mins = [-2, -2]
maxs = [2, 2]
search_space = Box(mins, maxs)
num_objective = 2
input_dim = 2

initial_query_points = search_space.sample(num_initial_points)
initial_data = observer(initial_query_points)

300
def build_stacked_independent_objectives_model(data: Dataset, num_output) -> ModelStack:
    gprs = []
    for idx in range(num_output):
        single_obj_data = Dataset(data.query_points, tf.gather(data.observations, [idx], axis=1))
        variance = tf.math.reduce_variance(single_obj_data.observations)
        kernel = gpflow.kernels.RBF(variance, lengthscales=[1.0] * input_dim)
        kernel.lengthscales.prior = tfd.LogNormal(tf.math.log(tf.constant([1] * input_dim, dtype=tf.float64)),
                                                  tf.constant([0.7], dtype=tf.float64))
        # jitter = gpflow.kernels.White(1e-12)
        gpr = gpflow.models.GPR((single_obj_data.query_points, single_obj_data.observations), kernel,
                                noise_variance=1e-5)
        gpflow.utilities.set_trainable(gpr.likelihood, False)
        gprs.append((create_model(GPflowModelConfig(
                        **{
                            "model": gpr,
                            "optimizer": gpflow.optimizers.Scipy(),
                            "optimizer_args": {"minimize_args": {"options": dict(maxiter=100)}},
                        }
                    )), 1))

    return ModelStack(*gprs)


models = {
    OBJECTIVE: build_stacked_independent_objectives_model(initial_data[OBJECTIVE], num_objective)
}
models[OBJECTIVE].optimize(initial_data[OBJECTIVE])


# %% [markdown]
# We investigate different PF samples: what is the acq function contour looks like:

# %%
def sample_and_plot_contours(pf_size):
    print(f'pf size: {pf_size}')
    from trieste.acquisition.multi_objective.mo_utils import sample_pareto_fronts_from_parametric_gp_posterior
    from matplotlib import pyplot as plt
    pf_samples, pf_samples_x = sample_pareto_fronts_from_parametric_gp_posterior(models[OBJECTIVE], initial_data, pf_size, search_space, 
                                                                                 num_moo_iter=1000, return_pf_input=True)
    plt.figure()
    for pf in pf_samples:
        plt.scatter(pf[:, 0], pf[:, 1])
    plt.scatter(Pareto(initial_data[OBJECTIVE].observations).front[:, 0], Pareto(initial_data[OBJECTIVE].observations).front[:, 1], 
                marker='X', s=100)
    plt.savefig(f'PF_samples_with_pf_size_{pf_size}')
    plt.close()
    
    from trieste.acquisition.function import pfes_ibo
    from trieste.acquisition.multi_objective.partition import prepare_default_non_dominated_partition_bounds
    partition_bounds = [
                    prepare_default_non_dominated_partition_bounds(
                        _pf,
                        tf.constant([-1e100] * 2, dtype=_pf.dtype),
                        tf.constant([1e100] * 2, dtype=_pf.dtype),
                    )
                    for _pf in pf_samples]
    acq = pfes_ibo(models[OBJECTIVE], partition_bounds)
    acqf= lambda x: acq(tf.cast(tf.expand_dims(x, -2), dtype=tf.float64))
    plt_inst = view_2D_function_in_contour(
                        acqf,
                        list(
                            zip(search_space.lower.numpy(), search_space.upper.numpy())
                        ),
                        show=False, 
                        colorbar=True,
                        plot_fidelity=100)
    plt_inst.title(f'PFES_IBO with PF size: {pf_size}')
    plt_inst.savefig(f'PFES_IBO_with_PF_size_{pf_size}.png')
    plt_inst.close()


# %%
for pf in [1]:
    sample_and_plot_contours(pf)

# %%

# %%
from trieste.utils.plotting import view_2D_function_in_contour

# %%

# %% [markdown]
# Plot of different PF Samples

# %%
for pf_sample, pf_sample_x, partition_bound in zip(pf_samples, pf_samples_x, partition_bounds):
    plt.figure()
    plt.scatter(pf_sample[:, 0], pf_sample[:, 1])
    plt.xlim([-0.5, 1.5])
    plt.ylim([-0.5, 1.5])
    plt.show()
    acq = pfes_ibo(models[OBJECTIVE], [partition_bound])
    acqf= lambda x: acq(tf.cast(tf.expand_dims(x, -2), dtype=tf.float64))
    plt_inst = view_2D_function_in_contour(
                        acqf,
                        list(
                            zip(search_space.lower.numpy(), search_space.upper.numpy())
                        ),
                        show=False, 
                        colorbar=True,
                        plot_fidelity=100,
                    )
    plt_inst.scatter(
    pf_sample_x[:, 0],
    pf_sample_x[:, 1],
    s=30,
    label="data",
    color="r")

# %% [markdown]
# Investigate the source of numerical instability:

# %% [markdown]
# #### Constraint Case

# %% [markdown]
# Noe we check 2 acq:  
#
# - CPFES
# - CPFES-IBO

# %%
num_initial_points = 50

import math
import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from PyOptimize.utils.visualization import view_2D_function_in_contour

import trieste
from trieste.acquisition.rule import OBJECTIVE
from trieste.data import Dataset
from trieste.models import create_model
from trieste.models.interfaces import ModelStack
from trieste.space import Box
from trieste.objectives.multi_objectives import Constr_Ex
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.models.gpflow import GPflowModelConfig
from trieste.acquisition.multi_objective import Pareto

np.random.seed(1793)
tf.random.set_seed(1793)

constr_ex_obj = Constr_Ex().objective()
constr_ex_con = Constr_Ex().constraint()

OBJECTIVE = 'OBJECTIVE'
CONSTRAINT = 'CONSTRAINT'


def observer(query_points):
    return {
        OBJECTIVE: Dataset(query_points, constr_ex_obj(query_points)),
        CONSTRAINT: Dataset(query_points, constr_ex_con(query_points)),
    }


mins = [0.1, 0]
maxs = [1, 5]
search_space = Box(mins, maxs)
num_objective = 2
input_dim = 2

initial_query_points = search_space.sample(num_initial_points)
initial_data = observer(initial_query_points)

300
def build_stacked_independent_objectives_model(data: Dataset, num_output) -> ModelStack:
    gprs = []
    for idx in range(num_output):
        single_obj_data = Dataset(data.query_points, tf.gather(data.observations, [idx], axis=1))
        variance = tf.math.reduce_variance(single_obj_data.observations)
        kernel = gpflow.kernels.RBF(variance, lengthscales=[1.0] * input_dim)
        kernel.lengthscales.prior = tfd.LogNormal(tf.math.log(tf.constant([1] * input_dim, dtype=tf.float64)),
                                                  tf.constant([0.7], dtype=tf.float64))
        # jitter = gpflow.kernels.White(1e-12)
        gpr = gpflow.models.GPR((single_obj_data.query_points, single_obj_data.observations), kernel,
                                noise_variance=1e-5)
        gpflow.utilities.set_trainable(gpr.likelihood, False)
        gprs.append((create_model(GPflowModelConfig(
                        **{
                            "model": gpr,
                            "optimizer": gpflow.optimizers.Scipy(),
                            "optimizer_args": {"minimize_args": {"options": dict(maxiter=100)}},
                        }
                    )), 1))

    return ModelStack(*gprs)


models = {
    OBJECTIVE: build_stacked_independent_objectives_model(initial_data[OBJECTIVE], num_objective),
    CONSTRAINT: build_stacked_independent_objectives_model(initial_data[CONSTRAINT], num_objective)
}
models[OBJECTIVE].optimize(initial_data[OBJECTIVE])
models[CONSTRAINT].optimize(initial_data[CONSTRAINT])


# %% [markdown]
# ##### CPFE

# %%
def sample_and_plot_contours(pf_size):
    print(f'pf size: {pf_size}')
    from trieste.acquisition.multi_objective.mo_utils import sample_pareto_fronts_from_parametric_gp_posterior
    from matplotlib import pyplot as plt
    pf_samples, pf_samples_x = sample_pareto_fronts_from_parametric_gp_posterior(models[OBJECTIVE],
                                                                                 initial_data, pf_size, search_space,
                                                                                 num_moo_iter=300,
                                                                                 cons_parametric_sampler=models[CONSTRAINT],
                                                                                 return_pf_input=True)
    plt.figure()
    for pf in pf_samples:
        plt.scatter(pf[:, 0], pf[:, 1])
    plt.scatter(Pareto(initial_data[OBJECTIVE].observations).front[:, 0],
                Pareto(initial_data[OBJECTIVE].observations).front[:, 1],
                marker='X', s=100)
    plt.savefig(f'PF_samples_with_pf_size_{pf_size}')
    plt.close()

    from trieste.acquisition.function import cpfes
    from trieste.acquisition.multi_objective.partition import prepare_default_non_dominated_partition_bounds
    partition_bounds = [
        prepare_default_non_dominated_partition_bounds(
            _pf,
            tf.constant([-1e100] * 2, dtype=_pf.dtype),
            tf.constant([1e100] * 2, dtype=_pf.dtype),
        )
        for _pf in pf_samples]
    acq = cpfes(models[OBJECTIVE], models[CONSTRAINT], partition_bounds,
                constraint_threshold=tf.zeros(
                    shape=2, dtype=tf.float64)
                )
    acqf = lambda x: acq(tf.cast(tf.expand_dims(x, -2), dtype=tf.float64))
    plt_inst = view_2D_function_in_contour(
        acqf,
        list(
            zip(search_space.lower.numpy(), search_space.upper.numpy())
        ),
        show=False,
        colorbar=True,
        plot_fidelity=100)
    plt_inst.title(f'cPFES with PF size: {pf_size}')
    plt_inst.show()
    # plt_inst.savefig(f'cPFES_with_PF_size_{pf_size}.png')


# %%
for pf in [10]:
    sample_and_plot_contours(pf)


# %% [markdown]
# ##### CPFES-IBO

# %%
def sample_and_plot_contours(pf_size):
    print(f'pf size: {pf_size}')
    from trieste.acquisition.multi_objective.mo_utils import sample_pareto_fronts_from_parametric_gp_posterior
    from matplotlib import pyplot as plt
    pf_samples, pf_samples_x = sample_pareto_fronts_from_parametric_gp_posterior(models[OBJECTIVE],
                                                                                 initial_data, pf_size, search_space,
                                                                                 num_moo_iter=300,
                                                                                 cons_parametric_sampler=models[CONSTRAINT],
                                                                                 return_pf_input=True)
    plt.figure()
    for pf in pf_samples:
        plt.scatter(pf[:, 0], pf[:, 1])
    plt.scatter(Pareto(initial_data[OBJECTIVE].observations).front[:, 0],
                Pareto(initial_data[OBJECTIVE].observations).front[:, 1],
                marker='X', s=100)
    plt.savefig(f'PF_samples_with_pf_size_{pf_size}')
    plt.close()

    from trieste.acquisition.function import cpfes_ibo
    from trieste.acquisition.multi_objective.partition import prepare_default_non_dominated_partition_bounds
    partition_bounds = [
        prepare_default_non_dominated_partition_bounds(
            _pf,
            tf.constant([-1e100] * 2, dtype=_pf.dtype),
            tf.constant([1e100] * 2, dtype=_pf.dtype),
        )
        for _pf in pf_samples]
    acq = cpfes_ibo(models[OBJECTIVE], models[CONSTRAINT], partition_bounds,
                constraint_threshold=tf.zeros(
                    shape=2, dtype=tf.float64)
                )
    acqf = lambda x: acq(tf.cast(tf.expand_dims(x, -2), dtype=tf.float64))
    plt_inst = view_2D_function_in_contour(
        acqf,
        list(
            zip(search_space.lower.numpy(), search_space.upper.numpy())
        ),
        show=False,
        colorbar=True,
        plot_fidelity=100)
    plt_inst.title(f'cPFES_IBO with PF size: {pf_size}')
    plt_inst.show()
    # plt_inst.savefig(f'cPFES_with_PF_size_{pf_size}.png')


# %%
for pf in [10]:
    sample_and_plot_contours(pf)

# %% [markdown]
# -------

# %% [markdown]
# # Synthetic Funtion Performance Difference

# %% [markdown]
# ###  Evaluation of different acquisition function

# %% [markdown]
# In this section, we provide an accurate estimation of mutual information using brutal force sampling technique. 
#
# Note, in-fact, we are not approximating mutual information: we are approximating part of it (as we use $K=1$)

# %% [markdown]
# Main reference for this include PES. PPESMOC

# %%

# %% [markdown]
# ## Experimental Setting
# ### PFES
# ####  Experimental Details
# | Details Items        | Notes   | Extra Notes  |
# | ------------- |:-------------:| -----:|
# | Pareto Frontier Samples: $\vert𝑃𝐹\vert$   | 10, 30, 50 | NA |
# | Pareto Frontier size (MOO Setting)    |  50  |  NA |
# | Design of Experiments number | 10   |   NA |
# | Performance Measure | Relative Hypervolume Improvement   |   NA |
#
# #### Synthetic Function
#
# | Details Items        | Dimensionality   | Outcome number  |
# | ------------- |:-------------:| -----:|
# | Ackerly/Sphere | 2 | NA |
# | DTLZ3    |    |  NA |
# | ZDT4  |    |   NA |
# | DTLZ4 |   |   NA |

# %%
results_stacked = {}

# %% [markdown]
# ### MESMO Optimizer

# %% [markdown]
# Here we prepare a MESMO optimizer that can be later evaluated by any synthetic function

# %%
import matplotlib.pyplot as plt
import tensorflow as tf
from trieste.experimental.plotting import plot_bo_points, plot_function_2d, plot_mobo_history, plot_mobo_points_in_obj_space, \
    plot_gp_2d, plot_acq_function_2d
from trieste.acquisition.function import MESMO
from trieste.acquisition.rule import OBJECTIVE
from trieste.data import Dataset
from trieste.space import Box
from trieste.acquisition.rule import EfficientGlobalOptimization
from typing import Dict, Generic, TypeVar, cast, overload
from trieste.models import ModelSpec, TrainableProbabilisticModel, create_model
from trieste.utils import Err, Ok, Result, map_values
from trieste.bayesian_optimizer import Record, S, SP, OptimizationResult
from trieste.acquisition.rule import AcquisitionRule
from trieste.utils.plotting import view_2D_function_in_contour
from collections.abc import Mapping
from absl import logging
import copy
import traceback


def mesmo_optimize(starting_data: Dataset, starting_model: ModelSpec, observer, search_space: Box, total_iter: int,
                   pb_name_prefix: str, obj1_lim: [list, None] = None, obj2_lim: [list, None] = None):
    mesmo = MESMO(search_space, num_pf_samples=1, popsize=50, moo_iter=1000)
    rule: EfficientGlobalOptimization[Box] = EfficientGlobalOptimization(builder=mesmo.using(OBJECTIVE))
    num_initial_points = starting_data[OBJECTIVE].query_points.shape[0]

    num_steps = total_iter
    datasets = starting_data
    model_specs = starting_model
    acquisition_rule = rule
    track_state = True
    acquisition_state = None

    if isinstance(datasets, Dataset):
        datasets = {OBJECTIVE: datasets}
        model_specs = {OBJECTIVE: model_specs}

    # reassure the type checker that everything is tagged
    datasets = cast(Dict[str, Dataset], datasets)
    model_specs = cast(Dict[str, ModelSpec], model_specs)

    if num_steps < 0:
        raise ValueError(f"num_steps must be at least 0, got {num_steps}")

    if datasets.keys() != model_specs.keys():
        raise ValueError(
            f"datasets and model_specs should contain the same keys. Got {datasets.keys()} and"
            f" {model_specs.keys()} respectively."
        )

    if not datasets:
        raise ValueError("dicts of datasets and model_specs must be populated.")

    if acquisition_rule is None:
        if datasets.keys() != {OBJECTIVE}:
            raise ValueError(
                f"Default acquisition rule EfficientGlobalOptimization requires tag"
                f" {OBJECTIVE!r}, got keys {datasets.keys()}"
            )

        acquisition_rule = cast(AcquisitionRule[S, SP], EfficientGlobalOptimization())

    models = map_values(create_model, model_specs)
    history = []

    for step in range(num_steps):
        print(f'BO Iter: {step}')
        if track_state:
            history.append(Record(datasets, models, acquisition_state))

        try:
            if track_state:
                models = copy.deepcopy(models)
                acquisition_state = copy.deepcopy(acquisition_state)

            acquisition_function = acquisition_rule._builder.prepare_acquisition_function(datasets, models)
            query_points = acquisition_rule._optimizer(search_space, acquisition_function)

            def acq_f(at):
                return acquisition_function(tf.expand_dims(at, axis=1))

            plt.figure(figsize=(10, 10))
            plt_inst = view_2D_function_in_contour(acq_f,
                                                   list(zip(search_space.lower.numpy(), search_space.upper.numpy())),
                                                   show=False, colorbar=True, plot_fidelity=500, title=f'iter:{step}')
            plt_inst.scatter(
                datasets[OBJECTIVE].query_points[:, 0],
                datasets[OBJECTIVE].query_points[:, 1],
                s=30,
                label="data",
                color='r'
            )
            plt_inst.scatter(
                query_points[:, 0],
                query_points[:, 1],
                s=30,
                label="data just added",
                color='k'
            )
            plt_inst.legend(fontsize=20)
            plt_inst.savefig(f'mesmo_{pb_name_prefix}_iter{step}.png')
            plt_inst.close()

            # plot in obj space
            _, ax = plot_mobo_points_in_obj_space(datasets[OBJECTIVE].observations, num_init=num_initial_points)
            ax.scatter(observer(query_points)['OBJECTIVE'].observations[:, 0],
                       observer(query_points)['OBJECTIVE'].observations[:, 1], s=30, label="data just added", color='k')
            plt.legend()
            if obj1_lim is not None:
                plt.xlim(obj1_lim)
            if obj2_lim is not None:
                plt.ylim(obj2_lim)
            plt.savefig(f'mesmo_{pb_name_prefix}_iter{step}_objspace.png')
            plt.close()

            observer_output = observer(query_points)

            tagged_output = (
                observer_output
                if isinstance(observer_output, Mapping)
                else {OBJECTIVE: observer_output}
            )

            datasets = {tag: datasets[tag] + tagged_output[tag] for tag in tagged_output}

            for tag, model in models.items():
                dataset = datasets[tag]
                model.update(dataset)
                model.optimize(dataset)

        except Exception as error:  # pylint: disable=broad-except
            tf.print(
                f"\nOptimization failed at step {step}, encountered error with traceback:"
                f"\n{traceback.format_exc()}"
                f"\nTerminating optimization and returning the optimization history. You may "
                f"be able to use the history to restart the process from a previous successful "
                f"optimization step.\n",
                output_stream=logging.ERROR,
            )

    tf.print("Optimization completed without errors", output_stream=logging.INFO)

    record = Record(datasets, models, acquisition_state)
    result = OptimizationResult(Ok(record), history)
    return result


# %% [markdown]
# ### PFES Optimizer

# %% [markdown]
# We also prepare a PFES optimizer that ca be later evaluated

# %%
import matplotlib.pyplot as plt
import tensorflow as tf
from trieste.experimental.plotting import plot_bo_points, plot_function_2d, plot_mobo_history, plot_mobo_points_in_obj_space, \
    plot_gp_2d, plot_acq_function_2d
from trieste.acquisition.function.multi_objective import BatchFeasibleParetoFrontierEntropySearch
from trieste.acquisition.rule import OBJECTIVE
from trieste.data import Dataset
from trieste.space import Box
from trieste.acquisition.rule import EfficientGlobalOptimization
from typing import Dict, Generic, TypeVar, cast, overload
from trieste.models import ModelSpec, TrainableProbabilisticModel, create_model
from trieste.utils import Err, Ok, Result, map_values
from trieste.bayesian_optimizer import Record, S, SP, OptimizationResult
from trieste.acquisition.rule import AcquisitionRule
from collections.abc import Mapping
from trieste.utils.plotting import view_2D_function_in_contour
from absl import logging
import copy
import traceback


def pfes_optimize(starting_data: Dataset, starting_model: ModelSpec, observer, search_space: Box, total_iter: int,
                  pb_name_prefix: str, obj1_lim: [list, None] = None, plot_acq = True, obj2_lim: [list, None] = None, ff_method='rff', ff_num = 10000):
    pfes = BatchFeasibleParetoFrontierEntropySearch(search_space, objective_tag='OBJECTIVE', num_pf_mc_samples=5, popsize=50,
                                                    ff_method=ff_method, ff_num=ff_num)
    rule: EfficientGlobalOptimization[Box] = EfficientGlobalOptimization(builder=pfes)
    num_initial_points = starting_data[OBJECTIVE].query_points.shape[0]

    num_steps = total_iter
    datasets = starting_data
    model_specs = starting_model
    acquisition_rule = rule
    track_state = True
    acquisition_state = None

    if isinstance(datasets, Dataset):
        datasets = {OBJECTIVE: datasets}
        model_specs = {OBJECTIVE: model_specs}

    # reassure the type checker that everything is tagged
    datasets = cast(Dict[str, Dataset], datasets)
    model_specs = cast(Dict[str, ModelSpec], model_specs)

    if num_steps < 0:
        raise ValueError(f"num_steps must be at least 0, got {num_steps}")

    if datasets.keys() != model_specs.keys():
        raise ValueError(
            f"datasets and model_specs should contain the same keys. Got {datasets.keys()} and"
            f" {model_specs.keys()} respectively."
        )

    if not datasets:
        raise ValueError("dicts of datasets and model_specs must be populated.")

    if acquisition_rule is None:
        if datasets.keys() != {OBJECTIVE}:
            raise ValueError(
                f"Default acquisition rule EfficientGlobalOptimization requires tag"
                f" {OBJECTIVE!r}, got keys {datasets.keys()}"
            )

        acquisition_rule = cast(AcquisitionRule[S, SP], EfficientGlobalOptimization())

    models = map_values(create_model, model_specs)
    history = []

    for step in range(num_steps):
        print(f'BO Iter: {step}')
        if track_state:
            history.append(Record(datasets, models, acquisition_state))

        try:
            if track_state:
                models = copy.deepcopy(models)
                acquisition_state = copy.deepcopy(acquisition_state)

            acquisition_function = acquisition_rule._builder.prepare_acquisition_function(models, datasets)
            query_points = acquisition_rule._optimizer(search_space, acquisition_function)
            if plot_acq:
                def acq_f(at):
                    return acquisition_function(tf.expand_dims(at, axis=1))
    
                plt.figure(figsize=(10, 10))
                plt_inst = view_2D_function_in_contour(acq_f,
                                                       list(zip(search_space.lower.numpy(), search_space.upper.numpy())),
                                                       show=False, colorbar=True, plot_fidelity=500, title=f'iter:{step}')
                plt_inst.scatter(
                    datasets[OBJECTIVE].query_points[:, 0],
                    datasets[OBJECTIVE].query_points[:, 1],
                    s=30,
                    label="data",
                    color='r'
                )
                plt_inst.scatter(
                    query_points[:, 0],
                    query_points[:, 1],
                    s=30,
                    label="data just added",
                    color='k'
                )
                plt_inst.legend(fontsize=20)
                plt_inst.savefig(f'pfes_{pb_name_prefix}_iter{step}.png')
                plt_inst.close()
            # plot in obj space
            _, ax = plot_mobo_points_in_obj_space(datasets[OBJECTIVE].observations, num_init=num_initial_points)
            ax.scatter(observer(query_points)['OBJECTIVE'].observations[:, 0],
                       observer(query_points)['OBJECTIVE'].observations[:, 1], s=30, label="data just added", color='k')
            plt.legend()
            if obj1_lim is not None:
                plt.xlim(obj1_lim)
            if obj2_lim is not None:
                plt.ylim(obj2_lim)
            plt.savefig(f'pfes_{pb_name_prefix}_iter{step}_objspace.png')
            plt.close()

            observer_output = observer(query_points)

            tagged_output = (
                observer_output
                if isinstance(observer_output, Mapping)
                else {OBJECTIVE: observer_output}
            )

            datasets = {tag: datasets[tag] + tagged_output[tag] for tag in tagged_output}

            for tag, model in models.items():
                dataset = datasets[tag]
                model.update(dataset)
                model.optimize(dataset)

        except Exception as error:  # pylint: disable=broad-except
            tf.print(
                f"\nOptimization failed at step {step}, encountered error with traceback:"
                f"\n{traceback.format_exc()}"
                f"\nTerminating optimization and returning the optimization history. You may "
                f"be able to use the history to restart the process from a previous successful "
                f"optimization step.\n",
                output_stream=logging.ERROR,
            )

    tf.print("Optimization completed without errors", output_stream=logging.INFO)

    record = Record(datasets, models, acquisition_state)
    result = OptimizationResult(Ok(record), history)
    return result

# %% [markdown]
# ## VLMOP2-2IN-2OUT

# %%
results_stacked['VLMOP2'] = {}

# %% [markdown]
# We first initialize the synthetic function, which can be used by both MESMO and PFES

# %%
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import gpflow
from trieste.experimental.plotting import plot_gp_2d

from trieste.acquisition.rule import OBJECTIVE
from trieste.data import Dataset
from trieste.models.interfaces import ModelStack
from trieste.space import Box
from trieste.objectives.multi_objectives import VLMOP2
from trieste.objectives.utils import mk_observer
from trieste.models.gpflow import GPflowModelConfig

np.random.seed(1793)
tf.random.set_seed(1793)

vlmop2 = VLMOP2().objective()
observer = mk_observer(vlmop2, OBJECTIVE)

mins = [-2, -2]
maxs = [2, 2]
search_space = Box(mins, maxs)
num_objective = 2

num_initial_points = 10
initial_query_points = search_space.sample(num_initial_points)
initial_data = observer(initial_query_points)


# %%
def build_stacked_independent_objectives_model(data: Dataset, num_output) -> ModelStack:
        gprs =[]
        for idx in range(num_output):
            single_obj_data = Dataset(data.query_points, tf.gather(data.observations, [idx], axis=1))
            variance = tf.math.reduce_variance(single_obj_data.observations)
            kernel = gpflow.kernels.RBF(variance)
            gpr = gpflow.models.GPR((single_obj_data.query_points, single_obj_data.observations), kernel, noise_variance=1e-5)
            gpflow.utilities.set_trainable(gpr.likelihood, False)
            gprs.append((create_model(GPflowModelConfig(**{
            "model": gpr,
            "optimizer": gpflow.optimizers.Scipy(),
            "optimizer_args": {
            "minimize_args": {"options": dict(maxiter=100)}}})), 1))

        return ModelStack(*gprs)


# %%
models = {OBJECTIVE: build_stacked_independent_objectives_model(initial_data[OBJECTIVE], num_objective)}
models[OBJECTIVE].optimize(initial_data[OBJECTIVE])

# %%
plot_gp_2d(models[OBJECTIVE]._models[0]._models, [-2, -2], [2, 2])
plot_gp_2d(models[OBJECTIVE]._models[1]._models, [-2, -2], [2, 2])

# %% [markdown]
# ### Run experiments

# %% [markdown]
# We save the plot in local disk instead of showing in this notebook, hence we temprally disable plot showing: 

# %%
plt.ioff()

# %% [markdown]
# We run the experiments

# %% [markdown]
# **MESMO:**

# %%
results_stacked['VLMOP2']['MESMO'] = mesmo_optimize(starting_data = initial_data, starting_model = models, 
               observer = observer, search_space = search_space , total_iter = 60, 
                  pb_name_prefix =  'VLMOP2', obj1_lim = [0, 1.05], obj2_lim = [0, 1.05])

# %% [markdown]
# **PFES**

# %%
results_stacked['VLMOP2']['PFES'] = pfes_optimize(starting_data = initial_data, starting_model = models, 
               observer = observer, search_space = search_space , total_iter = 60, 
                  pb_name_prefix =  'VLMOP2', obj1_lim = [0, 1.05], obj2_lim = [0, 1.05], ff_method='qff', ff_num = 50)

# %% [markdown]
# **PFES-IBO**

# %%
results_stacked['VLMOP2']['PFES-IBO'] = pfes_ibo_optimize(starting_data = initial_data, starting_model = models, 
               observer = observer, search_space = search_space , total_iter = 100, 
                  pb_name_prefix =  'VLMOP2', obj1_lim = [0, 1.05], obj2_lim = [0, 1.05], num_pf_samples = 1, plot_contour=True)

# %% [markdown]
# ### Result Comparison

# %% [markdown]
# We can also visualize how a performance metric evolved with respect to the number of BO iterations.
# First, we need to define a performance metric. Many metrics have been considered for multi-objective optimization. Here, we use the log hypervolume difference, defined as the difference between the hypervolume of the actual Pareto front and the hypervolume of the approximate Pareto front based on the bo-obtained data.

# %% [markdown]
#
# $$
# log_{10}\ \text{HV}_{\text{diff}} = log_{10}(\text{HV}_{\text{actual}} - \text{HV}_{\text{bo-obtained}})
# $$
#

# %% [markdown]
# First we need to calculate the $\text{HV}_{\text{actual}}$ based on the actual Pareto front. For some multi-objective synthetic functions like VLMOP2, the actual Pareto front has a clear definition, thus we could use `gen_pareto_optimal_points` to near uniformly sample on the actual Pareto front. And use these generated Pareto optimal points to (approximately) calculate the hypervolume of the actual Pareto frontier:

# %%
from trieste.utils.pareto import Pareto, get_reference_point

actual_pf = VLMOP2().gen_pareto_optimal_points(100)  # gen 100 pf points
ref_point = get_reference_point(tf.cast(actual_pf, dtype=tf.float64))
idea_hv = Pareto(tf.cast(actual_pf, dtype=tf.float64)).hypervolume_indicator(ref_point)


# %% [markdown]
# Then we define the metric function:


# %%
def log_hv(observations):
    obs_hv = Pareto(observations).hypervolume_indicator(ref_point)
    return math.log10(idea_hv - obs_hv)


# %% [markdown]
# Finally, we can plot the convergence of our performance metric over the course of the optimization.
# The blue vertical line in the figure denotes the time after which BO starts.

# %%
plt.figure()

obs_mesmo = results_stacked['VLMOP2']['MESMO'].try_get_final_datasets()['OBJECTIVE'].observations
obs_pfes = results_stacked['VLMOP2']['PFES'].try_get_final_datasets()['OBJECTIVE'].observations
size, obj_num = obs_mesmo.shape

plt.plot([log_hv(obs_mesmo[:pts, :]) for pts in range(size)], color="tab:orange", label='MESMO')
plt.plot([log_hv(obs_pfes[:pts, :]) for pts in range(size)], color="tab:blue", label='PFES')

plt.axvline(x=num_initial_points - 0.5, color="tab:blue")

plt.xlabel("Iterations")
plt.ylabel("log HV difference")
plt.legend()
plt.title('Performance Comparison of MESMO & PFES')
plt.show()

# %% [markdown]
# ## BraninCurrin-2IN-2OUT

# %% [markdown]
# PyMOO Solver

# %%
from trieste

# %%
Pareto(res).hypervolume_indicator(tf.constant([18.0, 6.0], dtype=tf.float64))

# %%
from trieste.acquisition.multi_objective.mo_utils import moo_optimize_pymoo
from trieste.objectives.multi_objectives import BraninCurrin
import tensorflow as tf

obj = BraninCurrin().objective()

res = moo_optimize_pymoo(obj, 2, 2, bounds = (tf.constant([0.0, 0.0]), tf.constant([1.0, 1.0])),
                         popsize=100, num_generation=2000)

from matplotlib import pyplot as plt
plt.scatter(res[:, 0], res[:, 1], label='Pareto Frontier')
# plt.scatter(uc_res[:, 0], uc_res[:, 1], label='unconstraint Pareto Frontier')
plt.legend()
plt.show()

# %%
np.savetxt('Constr_Ex_PF.txt', res)

# %%
results_stacked['BraninCurrin'] = {}

# %%
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from trieste.experimental.plotting import plot_bo_points, plot_function_2d, plot_mobo_points_in_obj_space

import trieste
from trieste.acquisition.rule import OBJECTIVE
from trieste.data import Dataset
from trieste.models.model_interfaces import ModelStack
from trieste.space import Box
from trieste.objectives.multi_objectives import BraninCurrin
from trieste.utils.pareto import Pareto

np.random.seed(1817)
tf.random.set_seed(1817)

bc_obj = BraninCurrin().objective()
observer = trieste.utils.objectives.mk_observer(bc_obj, OBJECTIVE)

mins = [0, 0]
maxs = [1, 1]
search_space = Box(mins, maxs)
num_objective = 2

num_initial_points = 10
initial_query_points = search_space.sample(num_initial_points)
initial_data = observer(initial_query_points)

# %%
_, ax = plot_function_2d(
    bc_obj,
    mins,
    maxs,
    grid_density=100,
    contour=True,
    title=["Branin", "Currin"],
    figsize=(8, 6),
    colorbar=True,
    xlabel="$X_1$",
    ylabel="$X_2$",
)
plot_bo_points(initial_data['OBJECTIVE'].query_points, ax=ax[0, 0], num_init=num_initial_points)
plot_bo_points(initial_data['OBJECTIVE'].query_points, ax=ax[0, 1], num_init=num_initial_points)
plt.show()


# %%
def build_stacked_independent_objectives_model(data: Dataset, num_output) -> ModelStack:
        gprs =[]
        for idx in range(num_output):
            single_obj_data = Dataset(data.query_points, tf.gather(data.observations, [idx], axis=1))
            variance = tf.math.reduce_variance(single_obj_data.observations)
            kernel = gpflow.kernels.RBF(variance, lengthscales=[1.0] * 2) 
            # jitter = gpflow.kernels.White(1e-12)
            gpr = gpflow.models.GPR((single_obj_data.query_points, single_obj_data.observations), kernel, noise_variance=1.1e-6)
            gpflow.utilities.set_trainable(gpr.likelihood, False)
            gprs.append((create_model({
            "model": gpr,
            "optimizer": gpflow.optimizers.Scipy(),
            "optimizer_args": {
            "minimize_args": {"options": dict(maxiter=300)}}}), 1))

        return ModelStack(*gprs)
    
models = {OBJECTIVE: build_stacked_independent_objectives_model(initial_data[OBJECTIVE], num_objective)}
models[OBJECTIVE].optimize(initial_data[OBJECTIVE])

# %%
plt.ion()

# %%
plot_gp_2d(models[OBJECTIVE]._models[0]._models, [0, 0], [1, 1])
plot_gp_2d(models[OBJECTIVE]._models[1]._models, [0, 0], [1, 1])

# %% [markdown]
# ### Run experiments

# %% [markdown]
# We save the plot in local disk instead of showing in this notebook, hence we temprally disable plot showing: 

# %%
plt.ioff()

# %% [markdown]
# We run the experiments

# %% [markdown]
# **MESMO:**

# %%
results_stacked['BraninCurrin']['MESMO'] = mesmo_optimize(starting_data = initial_data, starting_model = models, 
               observer = observer, search_space = search_space , total_iter = 50, 
                  pb_name_prefix =  'BraninCurrin', obj1_lim = [-1, 220], obj2_lim = [0, 11])

# %% [markdown]
# **PFES**

# %%
results_stacked['BraninCurrin']['PFES'] = pfes_optimize(starting_data = initial_data, starting_model = models, 
               observer = observer, search_space = search_space , total_iter = 50, 
                  pb_name_prefix =  'BraninCurrin', obj1_lim = [-1, 220], obj2_lim = [0, 11])

# %% [markdown]
# **PFES-IBO**

# %%
results_stacked['BraninCurrin']['PFES-IBO'] = pfes_ibo_optimize(starting_data = initial_data, starting_model = models, 
               observer = observer, search_space = search_space , total_iter = 50, 
                  pb_name_prefix =  'BraninCurrin', obj1_lim = [-1, 220], obj2_lim = [0, 11], plot_contour=True)

# %% [markdown]
# ### Result Evaluation

# %%
obs_mesmo = results_stacked['BraninCurrin']['MESMO'].try_get_final_datasets()['OBJECTIVE'].observations
obs_pfes = results_stacked['BraninCurrin']['PFES'].try_get_final_datasets()['OBJECTIVE'].observations

# %%
plot_mobo_points_in_obj_space(obs_mesmo)

# %%
plot_mobo_points_in_obj_space(obs_pfes)

# %%
tf.reduce_min(Pareto(obs_pfes).fronts, axis=0)

# %%
tf.reduce_min(Pareto(obs_mesmo).fronts, axis=0)

# %%
Pareto(Pareto(obs_pfes).fronts).hypervolume_indicator(tf.convert_to_tensor([50, 10], dtype=tf.float64))

# %%
Pareto(Pareto(obs_mesmo).fronts).hypervolume_indicator(tf.convert_to_tensor([50, 10], dtype=tf.float64))

# %%
Pareto(Pareto(obs_pfes).fronts).hypervolume_indicator(tf.convert_to_tensor([1e5] * 2, dtype=tf.float64))

# %%
Pareto(Pareto(obs_mesmo).fronts).hypervolume_indicator(tf.convert_to_tensor([1e5] * 2, dtype=tf.float64))

# %% [markdown]
# We first generate the pareto frontier from NSGA2 as the reference pareto front

# %%
plt.cla() 
plt.ion()

# %%
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_crossover, get_mutation, get_sampling
from pymoo.model.problem import Problem
from pymoo.optimize import minimize

class MyProblem(Problem):
    def __init__(self, n_var, n_obj, n_constr: int = 0):
        super().__init__(
            n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=mins, xu=maxs
        )

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = bc_obj(x)

problem = MyProblem(n_var=2, n_obj=2)
algorithm = NSGA2(
    pop_size=50,
    sampling=get_sampling("real_random"),
    crossover=get_crossover("real_sbx", prob=0.9, eta=15),
    mutation=get_mutation("real_pm", eta=20),
    eliminate_duplicates=True,
)

res = minimize(problem, algorithm, ("n_gen", 1000), save_history=False, verbose=False)

# %%
plt.figure()
plt.scatter(res.F[:, 0], res.F[:, 1])
plt.xlabel('Branin')
plt.xlim([0, 200])
plt.ylim([0, 15])
plt.ylabel('Currin')
plt.title('Idea PF')
plt.show()
# np.savetxt('Pareto_Front_BC.txt', res.F)


# %%
ideal_hv = Pareto(res.F).hypervolume_indicator(tf.convert_to_tensor([1e5] * 2, dtype=tf.float64))


# %%
def log_hv(observations):
    obs_hv = Pareto(observations).hypervolume_indicator(tf.convert_to_tensor([1e5] * 2, dtype=tf.float64))
    return math.log(ideal_hv - obs_hv)

plt.figure()

size, obj_num = obs_mesmo.shape

plt.plot([log_hv(obs_mesmo[:pts, :]) for pts in range(size)], color="tab:orange", label='MESMO')
plt.plot([log_hv(obs_pfes[:pts, :]) for pts in range(size)], color="tab:blue", label='PFES')

plt.axvline(x=num_initial_points - 0.5, color="tab:blue")

plt.xlabel("Iterations")
plt.ylabel("log HV difference")
plt.legend()
plt.title('Performance Comparison of MESMO & PFES')
plt.show()

# %% [markdown]
# --------------------

# %% [markdown]
# ## DTLZ1-4IN-3OUT

# %%
results_stacked['DTLZ1'] = {}

# %% [markdown]
# 3 Obj 4 Input dim

# %%
dtlz1_input_dim = 4

# %%
import numpy as np
import tensorflow as tf
from trieste.experimental.plotting import plot_mobo_points_in_obj_space

import trieste
from trieste.acquisition.rule import OBJECTIVE
from trieste.data import Dataset
from trieste.models.model_interfaces import ModelStack
from trieste.space import Box
from trieste.objectives.multi_objectives import DTLZ1

np.random.seed(1793)
tf.random.set_seed(1793)

dtlz1_obj = DTLZ1(input_dim = dtlz1_input_dim, num_objective=3).objective()
observer = trieste.utils.objectives.mk_observer(dtlz1_obj, OBJECTIVE)

mins = [0] * dtlz1_input_dim
maxs = [1] * dtlz1_input_dim
search_space = Box(mins, maxs)
num_objective = 3

# %%
num_initial_points = 10
initial_query_points = search_space.sample(num_initial_points)
initial_data = observer(initial_query_points)


def build_stacked_independent_objectives_model(data: Dataset, num_output) -> ModelStack:
        gprs =[]
        for idx in range(num_output):
            single_obj_data = Dataset(data.query_points, tf.gather(data.observations, [idx], axis=1))
            variance = tf.math.reduce_variance(single_obj_data.observations)
            kernel = gpflow.kernels.RBF(variance, lengthscales=[1.0] * dtlz1_input_dim) 
            kernel.lengthscales.prior = tfd.LogNormal(tf.math.log(tf.constant([1] * dtlz1_input_dim, dtype=tf.float64)), 
                                                      tf.constant([0.7], dtype=tf.float64))
            # jitter = gpflow.kernels.White(1e-12)
            gpr = gpflow.models.GPR((single_obj_data.query_points, single_obj_data.observations), kernel, noise_variance=1.1e-4)
            gpflow.utilities.set_trainable(gpr.likelihood, False)
            gprs.append((create_model({
            "model": gpr,
            "optimizer": gpflow.optimizers.Scipy(),
            "optimizer_args": {
            "minimize_args": {"options": dict(maxiter=300)}}}), 1))

        return ModelStack(*gprs)
    
models = {OBJECTIVE: build_stacked_independent_objectives_model(initial_data[OBJECTIVE], num_objective)}
models[OBJECTIVE].optimize(initial_data[OBJECTIVE])

# %% [markdown]
# ### Run experiments

# %% [markdown]
# We run the experiments

# %% [markdown]
# **MESMO:**

# %%
results_stacked['DTLZ1']['MESMO'] = mesmo_optimize(starting_data = initial_data, starting_model = models, 
               observer = observer, search_space = search_space , total_iter = 50, 
                  pb_name_prefix =  'DTLZ1')

# %% [markdown]
# **PFES**

# %%
results_stacked['DTLZ1']['PFES'] = pfes_optimize(starting_data = initial_data, starting_model = models, 
               observer = observer, search_space = search_space , total_iter = 100, 
                  pb_name_prefix =  'DTLZ1')

# %% [markdown]
# ### Result Evaluation

# %%
# obs_mesmo = results_stacked['BraninCurrin']['MESMO'].try_get_final_datasets()['OBJECTIVE'].observations
obs_pfes = results_stacked['DTLZ1']['PFES'].try_get_final_datasets()['OBJECTIVE'].observations

# %%
plot_mobo_points_in_obj_space(obs_mesmo)

# %%
plot_mobo_points_in_obj_space(obs_pfes)

# %% [markdown]
# ## DTLZ2-3IN-2OUT

# %%
results_stacked['DTLZ2'] = {}

# %%
from trieste.acquisition.multi_objective.mo_utils import moo_optimize_pymoo
from trieste.objectives.multi_objectives import DTLZ2

obj = DTLZ2(3, 2).objective()

res = moo_optimize_pymoo(obj, 3, 2, DTLZ2(3, 2).bounds,
                         popsize=10,  num_generation=500)




# %%
res = DTLZ2(3, 2).gen_pareto_optimal_points(100)

# %%
Pareto(tf.cast(res, dtype=tf.float64)).hypervolume_indicator(tf.constant([1.2] * 2, dtype=tf.float64))

# %%
np.savetxt('DTLZ2_PF.txt', res)

# %%
# %matplotlib notebook

# %%
from matplotlib import pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(res[:, 0], res[:, 1], res[:, 2])
plt.legend()
plt.show()

# %%
from trieste.acquisition.multi_objective import Pareto
import tensorflow as tf
Pareto(tf.cast(res, dtype=tf.float64)).hypervolume_indicator(tf.constant([1.5] * 2, dtype=tf.float64))

# %%
Pareto(tf.cast(DTLZ2(6, 3).gen_pareto_optimal_points(200), dtype=tf.float64)).hypervolume_indicator(tf.constant([1.5] * 3, dtype=tf.float64))

# %%

# %% [markdown]
# 2 Obj 3 Input dim

# %%
dtlz2_input_dim = 3

# %%
import numpy as np
import tensorflow as tf
from trieste.experimental.plotting import plot_mobo_points_in_obj_space

import trieste
from trieste.acquisition.rule import OBJECTIVE
from trieste.data import Dataset
from trieste.models.interfaces import ModelStack
from trieste.space import Box
from trieste.objectives.multi_objectives import DTLZ2
from trieste.acquisition.multi_objective.pareto import Pareto
from trieste.objectives.utils import mk_observer
import gpflow
from trieste.models.gpflow import GPflowModelConfig

np.random.seed(1793)
tf.random.set_seed(1793)

dtlz2_obj = DTLZ2(input_dim = dtlz2_input_dim, num_objective=2).objective()
observer = mk_observer(dtlz2_obj, OBJECTIVE)

mins = [0] * dtlz2_input_dim
maxs = [1] * dtlz2_input_dim
search_space = Box(mins, maxs)
num_objective = 2

# %%
num_initial_points = 10
initial_query_points = search_space.sample(num_initial_points)
initial_data = observer(initial_query_points)


def build_stacked_independent_objectives_model(data: Dataset, num_output) -> ModelStack:
        gprs =[]
        for idx in range(num_output):
            single_obj_data = Dataset(data.query_points, tf.gather(data.observations, [idx], axis=1))
            variance = tf.math.reduce_variance(single_obj_data.observations)
            kernel = gpflow.kernels.RBF(variance)
            gpr = gpflow.models.GPR((single_obj_data.query_points, single_obj_data.observations), kernel, noise_variance=1e-5)
            gpflow.utilities.set_trainable(gpr.likelihood, False)
            gprs.append((create_model(GPflowModelConfig(**{
            "model": gpr,
            "optimizer": gpflow.optimizers.Scipy(),
            "optimizer_args": {
            "minimize_args": {"options": dict(maxiter=100)}}})), 1))

        return ModelStack(*gprs)
    
models = {OBJECTIVE: build_stacked_independent_objectives_model(initial_data[OBJECTIVE], num_objective)}
models[OBJECTIVE].optimize(initial_data[OBJECTIVE])

# %% [markdown]
# ### Run experiments

# %% [markdown]
# We run the experiments

# %% [markdown]
# **MESMO:**

# %%
results_stacked['DTLZ2']['MESMO'] = mesmo_optimize(starting_data = initial_data, starting_model = models, 
               observer = observer, search_space = search_space , total_iter = 50, 
                  pb_name_prefix =  'DTLZ2')

# %% [markdown]
# **PFES**

# %% [raw]
# results_stacked['DTLZ2']['PFES'] = pfes_optimize(starting_data = initial_data, starting_model = models, 
#                observer = observer, search_space = search_space , total_iter = 100, 
#                   pb_name_prefix =  'DTLZ2', obj1_lim = [0, 1.5], obj2_lim = [0, 1.5], ff_method='rff', ff_num = 2000, plot_acq=False)

# %% [markdown]
# ### Result Evaluation

# %%
obs_pfes = results_stacked['DTLZ2']['PFES'].try_get_final_datasets()['OBJECTIVE'].observations

# %%
plot_mobo_points_in_obj_space(obs_mesmo)

# %%
plot_mobo_points_in_obj_space(Pareto(obs_pfes).front)

# %% [markdown]
# ## DTLZ3-6IN-4OUT [TODO]

# %%
from trieste.objectives.multi_objectives import DTLZ3

# %%
from trieste.acquisition.multi_objective.mo_utils import moo_optimize_pymoo
from trieste.objectives.multi_objectives import DTLZ3

obj = DTLZ3(3, 2).objective()

res = moo_optimize_pymoo(obj, 3, 2, DTLZ3(3, 2).bounds,
                         popsize=50,  num_generation=1000)

# %%
from matplotlib import pyplot as plt
plt.scatter(res[:, 0], res[:, 1])
plt.title('3-In-2-Obj DTLZ3 NSGA2 results (50 pop 1000 Iter)')

# %%
from trieste.acquisition.multi_objective.mo_utils import moo_optimize_pymoo
from trieste.objectives.multi_objectives import DTLZ3

obj = DTLZ3(6, 4).objective()

res = moo_optimize_pymoo(obj, 6, 4, DTLZ3(6, 4).bounds,
                         popsize=500,  num_generation=5000)

# %% [markdown]
# Calculate the hypervolume indicator on this PF: (8099999999.301131)

# %%
import tensorflow as tf
from trieste.acquisition.multi_objective.pareto import Pareto
Pareto(res).hypervolume_indicator(tf.constant([100] * 4, dtype=tf.float64))

# %%
import numpy as np
np.savetxt('DTLZ3_PF.txt', res)

# %%
tf.reduce_max(res, 0)

# %%
# %matplotlib notebook

# %%
from matplotlib import pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(res[:, 0], res[:, 1], res[:, 2])

# %%
from trieste.acquisition.multi_objective import Pareto

# %%
import tensorflow as tf
Pareto(res).hypervolume_indicator(tf.constant([5] * 3, dtype=tf.float64))

# %%
results_stacked['DTLZ3'] = {}

# %% [markdown]
# 3 Obj 4 Input dim

# %%
dtlz3_input_dim = 6

# %%
import numpy as np
import tensorflow as tf
from trieste.experimental.plotting import plot_mobo_points_in_obj_space

import trieste
from trieste.acquisition.rule import OBJECTIVE
from trieste.data import Dataset
from trieste.models.model_interfaces import ModelStack
from trieste.space import Box
from trieste.objectives.multi_objectives import DTLZ3
from trieste.utils.pareto import Pareto

np.random.seed(1793)
tf.random.set_seed(1793)

dtlz3_obj = DTLZ3(input_dim = dtlz3_input_dim, num_objective=4).objective()
observer = trieste.utils.objectives.mk_observer(dtlz3_obj, OBJECTIVE)

mins = [0] * dtlz3_input_dim
maxs = [1] * dtlz3_input_dim
search_space = Box(mins, maxs)
num_objective = 3

# %%
num_initial_points = 10
initial_query_points = search_space.sample(num_initial_points)
initial_data = observer(initial_query_points)


def build_stacked_independent_objectives_model(data: Dataset, num_output) -> ModelStack:
        gprs =[]
        for idx in range(num_output):
            single_obj_data = Dataset(data.query_points, tf.gather(data.observations, [idx], axis=1))
            variance = tf.math.reduce_variance(single_obj_data.observations)
            kernel = gpflow.kernels.RBF(variance, lengthscales=[1.0] * dtlz3_input_dim) 
            kernel.lengthscales.prior = tfd.LogNormal(tf.math.log(tf.constant([1] * dtlz3_input_dim, dtype=tf.float64)), 
                                                      tf.constant([0.7], dtype=tf.float64))
            # jitter = gpflow.kernels.White(1e-12)
            gpr = gpflow.models.GPR((single_obj_data.query_points, single_obj_data.observations), kernel, noise_variance=1.1e-4)
            gpflow.utilities.set_trainable(gpr.likelihood, False)
            gprs.append((create_model({
            "model": gpr,
            "optimizer": gpflow.optimizers.Scipy(),
            "optimizer_args": {
            "minimize_args": {"options": dict(maxiter=300)}}}), 1))

        return ModelStack(*gprs)
    
models = {OBJECTIVE: build_stacked_independent_objectives_model(initial_data[OBJECTIVE], num_objective)}
models[OBJECTIVE].optimize(initial_data[OBJECTIVE])

# %% [markdown]
# ### Run experiments

# %% [markdown]
# We run the experiments

# %% [markdown]
# **MESMO:**

# %%
results_stacked['DTLZ3']['MESMO'] = mesmo_optimize(starting_data = initial_data, starting_model = models, 
               observer = observer, search_space = search_space , total_iter = 50, 
                  pb_name_prefix =  'DTLZ3')

# %% [markdown]
# **PFES**

# %%
results_stacked['DTLZ3']['PFES'] = pfes_optimize(starting_data = initial_data, starting_model = models, 
               observer = observer, search_space = search_space , total_iter = 200, 
                  pb_name_prefix =  'DTLZ3')

# %% [markdown]
# ### Result Evaluation

# %%
# obs_mesmo = results_stacked['BraninCurrin']['MESMO'].try_get_final_datasets()['OBJECTIVE'].observations
obs_pfes = results_stacked['DTLZ3']['PFES'].try_get_final_datasets()['OBJECTIVE'].observations

# %%
plot_mobo_points_in_obj_space(obs_mesmo)

# %%
plot_mobo_points_in_obj_space(Pareto(obs_pfes).fronts)

# %% [markdown]
# ## DTLZ4-6IN-4OUT

# %%
import tensorflow as tf
[tf.constant(DTLZ4(6, 4).bounds[0]), tf.constant(DTLZ4(6, 4).bounds[1])]

# %%
from trieste.acquisition.multi_objective.mo_utils import moo_optimize_pymoo
from trieste.objectives.multi_objectives import DTLZ4

obj = DTLZ4(4, 3).objective()

res = moo_optimize_pymoo(obj, 4, 3, [tf.constant(DTLZ4(4, 3).bounds[0]), tf.constant(DTLZ4(4, 3).bounds[1])],
                         popsize=100,  num_generation=5000)

# %%
# %matplotlib notebook

# %%
from matplotlib import pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(res[:, 0], res[:, 1], res[:, 2])

# %%
import numpy as np
np.savetxt('DTLZ4_PF.txt', res)

# %% [markdown]
# Calculate the hypervolume indicator on this PF: (8099999999.301131)

# %%
import tensorflow as tf
from trieste.acquisition.multi_objective.pareto import Pareto
Pareto(res).hypervolume_indicator(tf.constant([1.2] * 3, dtype=tf.float64))

# %% [markdown]
# ## DTLZ5-6IN-4OUT

# %%
from trieste.acquisition.multi_objective.mo_utils import moo_optimize_pymoo
from trieste.objectives.multi_objectives import DTLZ5

obj = DTLZ5(6, 3).objective()

res = moo_optimize_pymoo(obj, 6, 3, DTLZ5(6, 3).bounds,
                         popsize=50,  num_generation=300)

# %%
# %matplotlib notebook

# %%
from matplotlib import pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(res[:, 0], res[:, 1], res[:, 2])

# %%
import numpy as np
np.savetxt('DTLZ4_PF.txt', res)

# %% [markdown]
# -------------------

# %% [markdown]
# ## DTLZ7-6IN-3OUT

# %%
from trieste.acquisition.multi_objective.mo_utils import moo_optimize_pymoo
from trieste.objectives.multi_objectives import DTLZ7

obj = DTLZ7(6, 3).objective()

res = moo_optimize_pymoo(obj, 6, 3, DTLZ7(6, 3).bounds,
                         popsize=100,  num_generation=1000)

# %%
# %matplotlib notebook

# %%
from matplotlib import pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(res[:, 0], res[:, 1], res[:, 2])

# %%
import numpy as np
np.savetxt('DTLZ4_PF.txt', res)

# %% [markdown]
# ## DTLZ-5IN-4OUT [TODO]

# %%
import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from trieste.experimental.plotting import plot_bo_points, plot_function_2d

import trieste
from trieste.acquisition.function import MESMO
from trieste.acquisition.rule import OBJECTIVE
from trieste.data import Dataset
from trieste.models.model_interfaces import ModelStack
from trieste.space import Box
from trieste.objectives.multi_objectives import DTLZ1
from trieste.utils.pareto import Pareto
from trieste.acquisition.rule import EfficientGlobalOptimization

np.random.seed(1793)
tf.random.set_seed(1793)

dtlz1_obj = DTLZ1(5, 4).objective()
observer = trieste.utils.objectives.mk_observer(dtlz1_obj, OBJECTIVE)

input_dim = 5
mins = [0] * input_dim
maxs = [1] * input_dim
search_space = Box(mins, maxs)
num_objective = 4

num_initial_points = 6
initial_query_points = search_space.sample(num_initial_points)
initial_data = observer(initial_query_points)


def build_stacked_independent_objectives_model(data: Dataset, num_output) -> ModelStack:
        gprs =[]
        for idx in range(num_output):
            single_obj_data = Dataset(data.query_points, tf.gather(data.observations, [idx], axis=1))
            variance = tf.math.reduce_variance(single_obj_data.observations)
            kernel = gpflow.kernels.RBF(variance, lengthscales=[1.0] * input_dim) 
            jitter = gpflow.kernels.White(1e-2)
            gpr = gpflow.models.GPR((single_obj_data.query_points, single_obj_data.observations), kernel, noise_variance=1e-5)
            gpflow.utilities.set_trainable(gpr.likelihood, False)
            gprs.append((create_model({
            "model": gpr,
            "optimizer": gpflow.optimizers.Scipy(),
            "optimizer_args": {
            "minimize_args": {"options": dict(maxiter=100)}}}), 1))

        return ModelStack(*gprs)
    
models = {OBJECTIVE: build_stacked_independent_objectives_model(initial_data[OBJECTIVE], num_objective)}

mesmo = MESMO(search_space, num_pf_samples=1, moo_iter = 1500)
rule: EfficientGlobalOptimization[Box] = EfficientGlobalOptimization(builder=mesmo.using(OBJECTIVE))
num_steps = 94
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
DTLZ_result = bo.optimize(num_steps, initial_data, models, acquisition_rule=rule)

# %%
dtlz1_obj(tf.constant([[0.2] * 5], dtype=tf.float64))

# %%
from trieste.bayesian_optimizer import OptimizationResult
result = OptimizationResult(Ok(record), history)

# %%
datasets = result.try_get_final_datasets()
data_query_points = datasets[OBJECTIVE].query_points
data_observations = datasets[OBJECTIVE].observations

_, ax = plot_function_2d(
    vlmop2,
    mins,
    maxs,
    grid_density=100,
    contour=True,
    figsize=(12, 6),
    title=["Obj 1", "Obj 2"],
    xlabel="$X_1$",
    ylabel="$X_2$",
    colorbar=True,
)
plot_bo_points(data_query_points, ax=ax[0, 0], num_init=num_initial_points)
plot_bo_points(data_query_points, ax=ax[0, 1], num_init=num_initial_points)
plt.show()

# %% [markdown]
# ----------------------------------

# %% [markdown]
# ## Ackerly/Sphere-2IN-2OUT [TODO]

# %% [markdown]
# TODO

# %% [markdown]
# --------------------------

# %% [markdown]
# ## Constr-EX 2IN-2OUT-2CON

# %%
pfx = np.loadtxt(r'C:\Users\Administrator\Desktop\trieste_pfes\docs\exp\exp_res\Constr_Ex\Constr_Ex_0_pfx.txt')

from trieste.objectives.multi_objectives import constr_ex

from matplotlib import pyplot as plt
plt.scatter(constr_ex(pfx)[:, 0], constr_ex(pfx)[:, 1])

# %%
results_stacked['ConstrEX'] = {}

# %% [markdown]
# A 2d input toy function from [wiki](https://en.wikipedia.org/wiki/File:Constr-Ex_problem.pdf) 

# %% [markdown]
# As a quick check, we could visualize its constraint Pareto frontier by performing a MOO using genetic algorithm

# %%
from trieste.acquisition.multi_objective.mo_utils import moo_optimize_pygmo
from trieste.objectives.multi_objectives import constr_ex, constr_ex_cons_func
# constraint version
res, x = moo_optimize_pygmo(constr_ex, 2, 2, bounds = ([0.0, 0], [1, 1]), # bounds = ([0.1, 0], [1, 5])
                   popsize=50, cons = constr_ex_cons_func, cons_num=2,
                         return_pf_x=True, num_generation=100)

# None constraint version
uc_res, uc_x = moo_optimize_pygmo(constr_ex, 2, 2, bounds = ([0.0, 0], [1, 1]),
                   popsize=50, return_pf_x=True, num_generation=100)
from matplotlib import pyplot as plt
plt.scatter(res[:, 0], res[:, 1], label='constraint Pareto Frontier')
plt.scatter(uc_res[:, 0], uc_res[:, 1], label='unconstraint Pareto Frontier')
plt.legend()
plt.show()

# %%
from trieste.acquisition.multi_objective.mo_utils import moo_optimize_pymoo
from trieste.objectives.multi_objectives import constr_ex, constr_ex_cons_func
# constraint version
res, x = moo_optimize_pymoo(constr_ex, 2, 2, bounds = ([0.0, 0], [1, 1]),
                   popsize=100, cons = constr_ex_cons_func, cons_num=2,
                         return_pf_x=True, num_generation=500)

# None constraint version
uc_res, uc_x = moo_optimize_pymoo(constr_ex, 2, 2, bounds = ([0.0, 0], [1, 1]),
                   popsize=50, return_pf_x=True, num_generation=50)
from matplotlib import pyplot as plt
plt.scatter(res[:, 0], res[:, 1], label='constraint Pareto Frontier')
plt.scatter(uc_res[:, 0], uc_res[:, 1], label='unconstraint Pareto Frontier')
plt.legend()
plt.show()

# %%
from trieste.acquisition.multi_objective.pareto import Pareto
import tensorflow as tf
Pareto(res).hypervolume_indicator(tf.constant([1.1, 10.0], dtype=tf.float64))

# %%
np.savetxt('Constr_Ex_PF.txt', res)

# %% [markdown]
# --------------

# %%
constrex_ref_pts = tf.constant([1.5, 11], dtype=tf.float64)

# %%
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from trieste.experimental.plotting import plot_gp_2d

import trieste
from trieste.acquisition.rule import OBJECTIVE
from trieste.data import Dataset
from trieste.models.model_interfaces import ModelStack
from trieste.space import Box
from trieste.objectives.multi_objectives import constr_ex, constr_ex_cons_func
from trieste.acquisition.rule import EfficientGlobalOptimization

np.random.seed(1793)
tf.random.set_seed(1793)

obj = constr_ex

CONSTRAINT = 'CONSTRAINT'

def observer(query_points):
    return {
        OBJECTIVE: Dataset(query_points, obj(query_points)),
        CONSTRAINT: Dataset(query_points, constr_ex_cons_func(query_points)),
    }

mins = [0.1, 0]
maxs = [1, 5]
search_space = Box(mins, maxs)
num_objective = 2
num_constraint = 2

# %%
num_initial_points = 2
initial_query_points = search_space.sample(num_initial_points)
initial_data = observer(initial_query_points)


def build_stacked_independent_objectives_model(data: Dataset, num_output) -> ModelStack:
    gprs = []
    for idx in range(num_output):
        single_obj_data = Dataset(data.query_points, tf.gather(data.observations, [idx], axis=1))
        variance = tf.math.reduce_variance(single_obj_data.observations)
        kernel = gpflow.kernels.RBF(variance, lengthscales=[1.0] * 2)
        kernel.lengthscales.prior = tfd.LogNormal(tf.math.log(tf.constant([1] * 2, dtype=tf.float64)),
                                                  tf.constant([0.7], dtype=tf.float64))
        # jitter = gpflow.kernels.White(1e-12)
        gpr = gpflow.models.GPR((single_obj_data.query_points, single_obj_data.observations), kernel,
                                noise_variance=1e-5)
        gpflow.utilities.set_trainable(gpr.likelihood, False)
        gprs.append((create_model({
            "model": gpr,
            "optimizer": gpflow.optimizers.Scipy(),
            "optimizer_args": {
                "minimize_args": {"options": dict(maxiter=100)}}}), 1))

    return ModelStack(*gprs)


# %%
models = {OBJECTIVE: build_stacked_independent_objectives_model(initial_data[OBJECTIVE], num_objective),
          CONSTRAINT: build_stacked_independent_objectives_model(initial_data[CONSTRAINT], num_constraint)}
models[OBJECTIVE].optimize(initial_data[OBJECTIVE])
models[CONSTRAINT].optimize(initial_data[CONSTRAINT])

# %%
plot_gp_2d(models[OBJECTIVE]._models[0]._models, mins, maxs)
plot_gp_2d(models[OBJECTIVE]._models[1]._models, mins, maxs)

# %% [markdown]
# ### Run experiments

# %% [markdown]
# We save the plot in local disk instead of showing in this notebook, hence we temprally disable plot showing: 

# %%
plt.ioff()

# %% [markdown]
# We run the experiments

# %% [markdown]
# **CPFES**

# %%
from trieste.acquisition.function import BatchFeasibleParetoFrontierEntropySearchInformationLowerBound

acq_cpfes_ibo = BatchFeasibleParetoFrontierEntropySearchInformationLowerBound(search_space, 'OBJECTIVE', 'CONSTRAINT', pf_mc_sample_num=5,
                                                                              popsize=50, moo_iterations=300)
rule_cpfes_ibo = EfficientGlobalOptimization(builder=acq_cpfes_ibo)  # type: ignore

num_steps = 100
bo = trieste.bayesian_optimizer.CustomBayesianOptimizer(observer, search_space, pb_name_prefix='Constr_Ex', acq_name='cpfes_ibo')
results_stacked['ConstrEX']['CPFES'] = bo.optimize(num_steps, initial_data, models, acquisition_rule=rule_cpfes_ibo,
                     kwargs_for_inferred_pareto={'obj1_lim': [0, 1.2], 'obj2_lim': [0, 20]},
                     kwargs_for_obj_space_plot={'obj1_lim': [0, 1.2], 'obj2_lim': [0, 20]}, 
                                             inspect_input_contour=True)

# %% [markdown]
# ## TNK Function 2IN-2OUT-2CON

# %%
results_stacked['TNK'] = {}

# %%
from trieste.acquisition.multi_objective.mo_utils import moo_optimize_pymoo
from trieste.objectives.multi_objectives import TNK
from math import pi

obj = TNK().objective()
con = TNK().constraint()
res, x = moo_optimize_pymoo(obj, 2, 2, bounds = ([0.0, 0], [1, 1]),
                   popsize=100, cons = con, cons_num=2,
                         return_pf_x=True, num_generation=500)

uc_res, uc_x = moo_optimize_pymoo(obj, 2, 2, bounds = ([0.0, 0], [1, 1]),
                   popsize=100, return_pf_x=True, num_generation=100)

from matplotlib import pyplot as plt
plt.scatter(res[:, 0], res[:, 1], label='constraint Pareto Frontier')
plt.scatter(uc_res[:, 0], uc_res[:, 1], label='unconstraint Pareto Frontier')
plt.legend()
plt.show()

# %%
from trieste.acquisition.multi_objective.pareto import Pareto
import tensorflow as tf
Pareto(res).hypervolume_indicator(tf.constant([1.1, 1.1], dtype=tf.float64))

# %%
np.savetxt('TNK_PF.txt', res)

# %% [markdown]
# -----------------

# %%
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from trieste.experimental.plotting import plot_mobo_points_in_obj_space, plot_gp_2d

import trieste
from math import pi
from trieste.acquisition.rule import OBJECTIVE
from trieste.data import Dataset
from trieste.models.model_interfaces import ModelStack
from trieste.space import Box
from trieste.objectives.multi_objectives import TNK
from trieste.acquisition.rule import EfficientGlobalOptimization

np.random.seed(1793)
tf.random.set_seed(1793)

obj = TNK().objective()
con = TNK().constraint()

CONSTRAINT = 'CONSTRAINT'

def observer(query_points):
    return {
        OBJECTIVE: Dataset(query_points, obj(query_points)),
        CONSTRAINT: Dataset(query_points, con(query_points)),
    }

mins = [0, 0]
maxs = [pi, pi]
search_space = Box(mins, maxs)
num_objective = 2
num_constraint = 2

# %%
num_initial_points = 10
initial_query_points = search_space.sample(num_initial_points)
initial_data = observer(initial_query_points)


def build_stacked_independent_objectives_model(data: Dataset, num_output) -> ModelStack:
    gprs = []
    for idx in range(num_output):
        single_obj_data = Dataset(data.query_points, tf.gather(data.observations, [idx], axis=1))
        variance = tf.math.reduce_variance(single_obj_data.observations)
        kernel = gpflow.kernels.RBF(variance, lengthscales=[1.0] * 2)
        kernel.lengthscales.prior = tfd.LogNormal(tf.math.log(tf.constant([0.8] * 2, dtype=tf.float64)),
                                                  tf.constant([0.7], dtype=tf.float64))
        # jitter = gpflow.kernels.White(1e-12)
        gpr = gpflow.models.GPR((single_obj_data.query_points, single_obj_data.observations), kernel,
                                noise_variance=1e-5)
        gpflow.utilities.set_trainable(gpr.likelihood, False)
        gprs.append((create_model({
            "model": gpr,
            "optimizer": gpflow.optimizers.Scipy(),
            "optimizer_args": {
                "minimize_args": {"options": dict(maxiter=100)}}}), 1))

    return ModelStack(*gprs)


# %%
models = {OBJECTIVE: build_stacked_independent_objectives_model(initial_data[OBJECTIVE], num_objective),
          CONSTRAINT: build_stacked_independent_objectives_model(initial_data[CONSTRAINT], num_constraint)}
models[OBJECTIVE].optimize(initial_data[OBJECTIVE])
models[CONSTRAINT].optimize(initial_data[CONSTRAINT])

# %%
plot_gp_2d(models[OBJECTIVE]._models[0]._models, mins, maxs)
plot_gp_2d(models[OBJECTIVE]._models[1]._models, mins, maxs)

# %% [markdown]
# ### Run experiments

# %% [markdown]
# We save the plot in local disk instead of showing in this notebook, hence we temprally disable plot showing: 

# %%
plt.ioff()

# %% [markdown]
# We run the experiments

# %% [markdown]
# **CPFES**

# %%
from trieste.acquisition.function import BatchFeasibleParetoFrontierEntropySearchInformationLowerBound

acq_cpfes_ibo = BatchFeasibleParetoFrontierEntropySearchInformationLowerBound(search_space, 'OBJECTIVE', 'CONSTRAINT', pf_mc_sample_num=1,
                                                                              popsize=50, moo_iterations=300)
rule_cpfes_ibo = EfficientGlobalOptimization(builder=acq_cpfes_ibo)  # type: ignore

num_steps = 100
bo = trieste.bayesian_optimizer.CustomBayesianOptimizer(observer, search_space, pb_name_prefix='TNK', acq_name='cpfes_ibo')
results_stacked['TNK']['CPFES'] = bo.optimize(num_steps, initial_data, models, acquisition_rule=rule_cpfes_ibo,
                     kwargs_for_inferred_pareto={'obj1_lim': [0, 1.2], 'obj2_lim': [0, 1.2]},
                     kwargs_for_obj_space_plot={'obj1_lim': [0, 1.2], 'obj2_lim': [0, 1.2]}, 
                                             inspect_input_contour=False)

# %% [markdown]
# ### Result Evaluation

# %%
obs_cpfes_ibo = results_stacked['TNK']['CPFES'].try_get_final_datasets()['OBJECTIVE'].observations

# %%
fig, ax = plot_mobo_points_in_obj_space(obs_cpfes_ibo)
ax.set_xlim([0, 1.2])
ax.set_ylim([0, 1.2])

# %%
tf.reduce_min(Pareto(obs_pfes).front, axis=0)

# %%
tf.reduce_min(Pareto(obs_mesmo).front, axis=0)

# %%
from trieste.acquisition.multi_objective.mo_utils import inference_pareto_fronts_from_gp_mean

res, resx = inference_pareto_fronts_from_gp_mean(results_stacked['TNK']['CPFES'].try_get_final_models()[OBJECTIVE], 
                                                search_space, popsize=50, cons_models = results_stacked['TNK']['CPFES'].try_get_final_models()[CONSTRAINT])

# %%
fig, ax = plot_mobo_points_in_obj_space(res)
ax.set_xlim([0, 1.2])
ax.set_ylim([0, 1.2])

# %% [markdown]
# ## Osy Function 6IN-2OUT-6CON

# %%
results_stacked['Osy'] = {}

# %% [markdown]
# PyMOO Solver

# %%
from trieste.acquisition.multi_objective.mo_utils import moo_optimize_pymoo
from trieste.objectives.multi_objectives import Osy

obj = Osy().objective()
con = Osy().constraint()

res = moo_optimize_pymoo(obj, 6, 2, bounds = ([0.0, 0.0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]),
                         popsize=200, cons = con, cons_num=6, num_generation=1000)

# uc_res = moo_optimize_pymoo(obj, 6, 2, bounds = ([0.0, 0.0, 1, 0, 1, 0], [10, 10, 5, 6, 5, 10]),
#                          popsize=200, num_generation=300)

from matplotlib import pyplot as plt
plt.scatter(res[:, 0], res[:, 1], label='constraint Pareto Frontier')
# plt.scatter(uc_res[:, 0], uc_res[:, 1], label='unconstraint Pareto Frontier')
plt.legend()
plt.show()

# %%
np.savetxt('Osy_PF.txt', res)

# %%
from trieste.acquisition.multi_objective.pareto import Pareto
import tensorflow as tf
Pareto(res).hypervolume_indicator(tf.constant([0.0, 80], dtype=tf.float64))

# %% [markdown]
# PyGMO Solver

# %%
from trieste.acquisition.multi_objective.mo_utils import moo_optimize_pygmo
from trieste.objectives.multi_objectives import Osy

obj = Osy().objective()
con = Osy().constraint()
res, x = moo_optimize_pygmo(obj, 6, 2, bounds = ([0.0, 0.0, 1, 0, 1, 0], [10, 10, 5, 6, 5, 10]),
                   popsize=200, cons = con, cons_num=6,
                         return_pf_x=True, num_generation=100)

from matplotlib import pyplot as plt
plt.scatter(res[:, 0], res[:, 1])
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from trieste.experimental.plotting import plot_mobo_points_in_obj_space

import trieste
from trieste.acquisition.rule import OBJECTIVE
from trieste.data import Dataset
from trieste.models.model_interfaces import ModelStack
from trieste.space import Box
from trieste.objectives.multi_objectives import Osy
from trieste.acquisition.rule import EfficientGlobalOptimization

np.random.seed(1793)
tf.random.set_seed(1793)

obj = Osy().objective()
con = Osy().constraint()


CONSTRAINT = 'CONSTRAINT'

def observer(query_points):
    return {
        OBJECTIVE: Dataset(query_points, obj(query_points)),
        CONSTRAINT: Dataset(query_points, con(query_points)),
    }

mins = [0.0, 0.0, 1, 0, 1, 0]
maxs = [10, 10, 5, 6, 5, 10]
search_space = Box(mins, maxs)
num_objective = 2
num_constraint = 6
input_dim = 6

# %%
num_initial_points = 10
initial_query_points = search_space.sample(num_initial_points)
initial_data = observer(initial_query_points)


def build_stacked_independent_objectives_model(data: Dataset, num_output) -> ModelStack:
    gprs = []
    for idx in range(num_output):
        single_obj_data = Dataset(data.query_points, tf.gather(data.observations, [idx], axis=1))
        variance = tf.math.reduce_variance(single_obj_data.observations)
        kernel = gpflow.kernels.RBF(variance, lengthscales=[1.0] * input_dim)
        kernel.lengthscales.prior = tfd.LogNormal(tf.math.log(tf.constant([1] * input_dim, dtype=tf.float64)),
                                                  tf.constant([0.7], dtype=tf.float64))
        # jitter = gpflow.kernels.White(1e-12)
        gpr = gpflow.models.GPR((single_obj_data.query_points, single_obj_data.observations), kernel,
                                noise_variance=1e-5)
        gpflow.utilities.set_trainable(gpr.likelihood, False)
        gprs.append((create_model({
            "model": gpr,
            "optimizer": gpflow.optimizers.Scipy(),
            "optimizer_args": {
                "minimize_args": {"options": dict(maxiter=100)}}}), 1))

    return ModelStack(*gprs)


# %%
models = {OBJECTIVE: build_stacked_independent_objectives_model(initial_data[OBJECTIVE], num_objective),
          CONSTRAINT: build_stacked_independent_objectives_model(initial_data[CONSTRAINT], num_constraint)}
models[OBJECTIVE].optimize(initial_data[OBJECTIVE])
models[CONSTRAINT].optimize(initial_data[CONSTRAINT])

# %% [markdown]
# ### Run Experiments

# %%
plt.ioff()

# %% [markdown]
# We run the experiments

# %% [markdown]
# **CPFES**

# %%
from trieste.acquisition.function import BatchFeasibleParetoFrontierEntropySearchInformationLowerBound

acq_cpfes_ibo = BatchFeasibleParetoFrontierEntropySearchInformationLowerBound(search_space, 'OBJECTIVE', 'CONSTRAINT', pf_mc_sample_num=5,
                                                                              popsize=50, moo_iterations=300)
rule_cpfes_ibo = EfficientGlobalOptimization(builder=acq_cpfes_ibo)  # type: ignore

num_steps = 100
bo = trieste.bayesian_optimizer.CustomBayesianOptimizer(observer, search_space, pb_name_prefix='Osy', acq_name='cpfes_ibo')
results_stacked['Osy']['CPFES'] = bo.optimize(num_steps, initial_data, models, acquisition_rule=rule_cpfes_ibo,
                     kwargs_for_inferred_pareto={'obj1_lim': [-400, 0], 'obj2_lim': [0, 120]},
                     kwargs_for_obj_space_plot={'obj1_lim': [-400, 0], 'obj2_lim': [0, 120]}, 
                                             inspect_input_contour=False)

# %% [markdown]
# ### Result Evaluation

# %%
obs_cpfes_ibo = results_stacked['Osy']['CPFES'].try_get_final_datasets()['OBJECTIVE'].observations
queries_cpfes_ibo = results_stacked['Osy']['CPFES'].try_get_final_datasets()['OBJECTIVE'].query_points

# %%
cons_model = results_stacked['Osy']['CPFES'].try_get_final_models()[CONSTRAINT]
mask_fail = tf.reduce_any(cons_model.predict(queries_cpfes_ibo)[0]<0, -1)

# %%
fig, ax = plot_mobo_points_in_obj_space(obs_cpfes_ibo, mask_fail = mask_fail)
ax.set_xlim([-300, 0])
ax.set_ylim([0, 80])

# %%
from trieste.acquisition.multi_objective.mo_utils import inference_pareto_fronts_from_gp_mean

res, resx = inference_pareto_fronts_from_gp_mean(results_stacked['Osy']['CPFES'].try_get_final_models()[OBJECTIVE], 
                                                search_space, popsize=50, cons_models = results_stacked['Osy']['CPFES'].try_get_final_models()[CONSTRAINT])

# %%
fig, ax = plot_mobo_points_in_obj_space(res)
ax.set_xlim([-300, 0])
ax.set_ylim([0, 80])

# %% [markdown]
# ## BNH Function 2IN-2OUT-2CON

# %%
results_stacked['BNH'] = {}

# %% [markdown]
# PyMOO Solver

# %%
from trieste.acquisition.multi_objective.mo_utils import moo_optimize_pymoo
from trieste.objectives.multi_objectives import BNH

obj = BNH().objective()
con = BNH().constraint()

res = moo_optimize_pymoo(obj, 2, 2, bounds = ([0, 0], [5, 3]),
                         popsize=100, cons = con, cons_num=2, num_generation=500)

uc_res = moo_optimize_pymoo(obj, 2, 2, bounds = ([0, 0], [5, 3]),
                         popsize=100, num_generation=500)

from matplotlib import pyplot as plt
_, axs = plt.subplots(1, 2)
axs[0].scatter(res[:, 0], res[:, 1], label='constraint Pareto Frontier')
axs[1].scatter(uc_res[:, 0], uc_res[:, 1], label='unconstraint Pareto Frontier')
plt.legend()
plt.show()

# %% [markdown]
# ## SRN Function 2IN-2OUT-2CON

# %%
results_stacked['SRN'] = {}

# %% [markdown]
# PyMOO Solver

# %% [markdown]
# The correctness of SRN can be seen as [here](https://books.google.be/books?id=p-AqDwAAQBAJ&pg=PA101&lpg=PA101&dq=SRN+function&source=bl&ots=k6jGg2V__F&sig=ACfU3U39NQSqR6knlwszY6EA9zt_s57FmA&hl=en&sa=X&ved=2ahUKEwj-3PCrrsnyAhWF2aQKHak7B_QQ6AF6BAgxEAM#v=onepage&q=SRN%20function&f=false)

# %%
from trieste.acquisition.multi_objective.mo_utils import moo_optimize_pymoo
from trieste.objectives.multi_objectives import SRN

obj = SRN().objective()
con = SRN().constraint()

res = moo_optimize_pymoo(obj, 2, 2, bounds = ([0, 0], [1, 1]),
                         popsize=200, cons = con, cons_num=2, num_generation=1000)

uc_res = moo_optimize_pymoo(obj, 2, 2, bounds = ([0, 0], [1, 1]),
                         popsize=50, num_generation=200)

from matplotlib import pyplot as plt
_, axs = plt.subplots(1, 2)
axs[0].scatter(res[:, 0], res[:, 1], label='constraint Pareto Frontier')
axs[1].scatter(uc_res[:, 0], uc_res[:, 1], label='unconstraint Pareto Frontier')
plt.legend()
plt.show()

# %%
res.max(0)

# %%
from trieste.acquisition.multi_objective.pareto import Pareto
import tensorflow as tf
Pareto(res).hypervolume_indicator(tf.constant([220, 20], dtype=tf.float64))

# %%
np.savetxt('SRN_PF.txt', res)

# %% [markdown]
# ## TwoBarTruss 3IN-2OUT-1CON

# %%
results_stacked['TwoBarTruss'] = {}

# %% [markdown]
# PyMOO Solver

# %% [markdown]
# The correctness of TwoBarTruss can be seen from Fig. 3 in the paper

# %%
from trieste.acquisition.multi_objective.mo_utils import moo_optimize_pymoo
from trieste.objectives.multi_objectives import TwoBarTruss

obj = TwoBarTruss().objective()
con = TwoBarTruss().constraint()

res, res_x = moo_optimize_pymoo(obj, 3, 2, bounds = ([0, 0, 0], [1, 1, 1]),
                         popsize=100, cons = con, cons_num=1, num_generation=2000, return_pf_x=True)

uc_res = moo_optimize_pymoo(obj, 3, 2, bounds = ([0, 0, 0], [1, 1, 1]),
                         popsize=50, num_generation=300)

from matplotlib import pyplot as plt
_, axs = plt.subplots(1, 2)
axs[0].scatter(res[:, 0], res[:, 1], label='constraint Pareto Frontier')
axs[1].scatter(uc_res[:, 0], uc_res[:, 1], label='unconstraint Pareto Frontier')
plt.legend()
plt.show()

# %%
res.min(0)

# %%
test_res = np.loadtxt(r'C:\Users\Administrator\Desktop\trieste_pfes\docs\exp\cfg\reference_optimal_solutions\TwoBarTruss\TwoBarTruss_PF.txt')
plt.scatter(test_res[:, 0], test_res[:, 1])

# %%
from trieste.utils.metrics import AverageHausdoff
AverageHausdoff(res_x, TwoBarTruss(), np.loadtxt(r'C:\Users\Administrator\Desktop\trieste_pfes\docs\exp\cfg\reference_optimal_solutions\TwoBarTruss\TwoBarTruss_PF.txt'))

# %%
np.savetxt('TwoBarTruss_PF.txt', res)

# %%
from trieste.acquisition.multi_objective.pareto import Pareto
import tensorflow as tf
Pareto(res).hypervolume_indicator(tf.constant([0.065, 105000], dtype=tf.float64))

# %% [markdown]
# ## WeldedBeamDesign 4IN-2OUT-4CON

# %%
results_stacked['WeldedBeamDesign'] = {}

# %% [markdown]
# PyMOO Solver

# %% [markdown]
# The correctness of TwoBarTruss can be seen from Fig. 4 in the paper

# %%
from trieste.acquisition.multi_objective.mo_utils import moo_optimize_pymoo
from trieste.objectives.multi_objectives import WeldedBeamDesign

obj = WeldedBeamDesign().objective()
con = WeldedBeamDesign().constraint()

res = moo_optimize_pymoo(obj, 4, 2, bounds = ([0, 0, 0, 0], [1, 1, 1, 1]),
                         popsize=200, cons = con, cons_num=4, num_generation=3000)

uc_res = moo_optimize_pymoo(obj, 4, 2, bounds = ([0, 0, 0, 0], [1, 1, 1, 1]),
                         popsize=100, num_generation=500)

from matplotlib import pyplot as plt
_, axs = plt.subplots(1, 2)
axs[0].scatter(res[:, 0], res[:, 1], label='constraint Pareto Frontier')
axs[1].scatter(uc_res[:, 0], uc_res[:, 1], label='unconstraint Pareto Frontier')
plt.legend()
plt.show()

# %%
res.max(0)

# %%
np.savetxt('WeldedBeamDesign_PF.txt', res)

# %%
pf = np.loadtxt(r'C:\Users\Administrator\Desktop\trieste_pfes\docs\exp\cfg\reference_optimal_solutions\WeldedBeamDesign\WeldedBeamDesign_PF.txt')

# %%
from trieste.acquisition.multi_objective.pareto import Pareto
import tensorflow as tf
Pareto(pf).hypervolume_indicator(tf.constant([40, 0.02], dtype=tf.float64))

# %% [markdown]
# # Result Visualization

# %%
import numpy as np

def loadres(dir, pb_name, aux_info, metric = 'LogHvDiff', max_query = 100, q=1):
    regrets = []
    i = 0
    while i <= max_query:
        # print(i)
        try:
            regret = np.loadtxt(rf'{dir}\{pb_name}_{i}_{aux_info}_{metric}_q{q}_.txt')
            regrets.append(regret)
            i +=1
        except:
            i +=1 
    return regrets


# %% [markdown]
# ## VLMOP2

# %%
from matplotlib import pyplot as plt
import numpy as np


ehvi = loadres(r'/docs/exp/exp_res/VLMOP2/EHVI', 'VLMOP2', 'out_of_sample')
random = loadres(r'/docs/exp/exp_res/VLMOP2/Random', 'VLMOP2', 'out_of_sample')
pfes_res_mc_1 = loadres(r'/docs/exp/exp_res/VLMOP2/BCPFES/MC1', 'VLMOP2', 'out_of_sample')
pfes_ibo_res_mc_1 = loadres(r'/docs/exp/exp_res/VLMOP2/BCPFES_IBO/MC1', 'VLMOP2', 'out_of_sample')
mesmo_res_mc_1 = loadres(r'/docs/exp/exp_res/VLMOP2/MESMO/MC1', 'VLMOP2', 'out_of_sample')
pfes_res_mc_10 = loadres(r'/docs/exp/exp_res/VLMOP2/BCPFES/MC10', 'VLMOP2', 'out_of_sample')
pfes_ibo_res_mc_10 = loadres(r'/docs/exp/exp_res/VLMOP2/BCPFES_IBO/MC10', 'VLMOP2', 'out_of_sample')

plt.plot(np.arange(len(pfes_res_mc_1[0])), np.mean(pfes_res_mc_1, 0), label='PFES 1')
plt.fill_between(np.arange(len(pfes_res_mc_1[0])), np.mean(pfes_res_mc_1, 0) - 1.96 * np.std(pfes_res_mc_1, 0), np.mean(pfes_res_mc_1, 0) + 1.96 * np.std(pfes_res_mc_1, 0), label='PFES 1')
plt.plot(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_1, 0), label='PFES-IBO 1')
plt.plot(np.arange(len(mesmo_res_mc_1[0])), np.mean(mesmo_res_mc_1, 0), label='MESMO 1')
# plt.fill_between(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_1, 0) - 1.96 * np.std(pfes_ibo_res_mc_1, 0), np.mean(pfes_ibo_res_mc_1, 0) + 1.96 * np.std(pfes_ibo_res_mc_1, 0))
plt.plot(np.arange(len(pfes_res_mc_1[0])), np.mean(pfes_res_mc_10, 0), label='PFES 10')
plt.plot(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_10, 0), label='PFES-IBO 10')
plt.plot(np.arange(len(ehvi[0])), np.mean(ehvi, 0), label='EHVI')
plt.plot(np.arange(len(random[0])), np.mean(random, 0), label='Random')
plt.legend()
plt.title('VLMOP2 inference out of sample regret')
plt.xlabel('Iterations')
plt.ylabel('log HV difference')

# %%
from matplotlib import pyplot as plt
import numpy as np

ehvi = loadres(r'/docs/exp/exp_res/VLMOP2/EHVI', 'VLMOP2', 'in_sample')
random = loadres(r'/docs/exp/exp_res/VLMOP2/Random', 'VLMOP2', 'in_sample')
pfes_res_mc_1 = loadres(r'/docs/exp/exp_res/VLMOP2/BCPFES/MC1', 'VLMOP2', 'in_sample')
pfes_ibo_res_mc_1 = loadres(r'/docs/exp/exp_res/VLMOP2/BCPFES_IBO/MC1', 'VLMOP2', 'in_sample')
mesmo_res_mc_1 = loadres(r'/docs/exp/exp_res/VLMOP2/MESMO/MC1', 'VLMOP2', 'in_sample')
pfes_res_mc_10 = loadres(r'/docs/exp/exp_res/VLMOP2/BCPFES/MC10', 'VLMOP2', 'in_sample')
pfes_ibo_res_mc_10 = loadres(r'/docs/exp/exp_res/VLMOP2/BCPFES_IBO/MC10', 'VLMOP2', 'in_sample')

plt.plot(np.arange(len(pfes_res_mc_1[0])), np.mean(pfes_res_mc_1, 0), label='PFES 1')
plt.fill_between(np.arange(len(pfes_res_mc_1[0])), np.mean(pfes_res_mc_1, 0) - 1.96 * np.std(pfes_res_mc_1, 0), np.mean(pfes_res_mc_1, 0) + 1.96 * np.std(pfes_res_mc_1, 0), label='PFES 1')
plt.plot(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_1, 0), label='PFES-IBO 1')
plt.plot(np.arange(len(mesmo_res_mc_1[0])), np.mean(mesmo_res_mc_1, 0), label='MESMO 1')
# plt.fill_between(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_1, 0) - 1.96 * np.std(pfes_ibo_res_mc_1, 0), np.mean(pfes_ibo_res_mc_1, 0) + 1.96 * np.std(pfes_ibo_res_mc_1, 0))
plt.plot(np.arange(len(pfes_res_mc_1[0])), np.mean(pfes_res_mc_10, 0), label='PFES 10')
plt.plot(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_10, 0), label='PFES-IBO 10')
plt.fill_between(np.arange(len(ehvi[0])), np.mean(ehvi, 0)- 1.96 * np.std(ehvi, 0), np.mean(ehvi, 0) + 1.96 * np.std(ehvi, 0), label='EHVI')
plt.plot(np.arange(len(ehvi[0])), np.mean(ehvi, 0), label='EHVI')
plt.plot(np.arange(len(random[0])), np.mean(random, 0), label='Random')
plt.legend()
plt.title('VLMOP2 inference in sample regret')
plt.xlabel('Iterations')
plt.ylabel('log HV difference')

# %% [markdown]
# ## BraninCurrin

# %%
from matplotlib import pyplot as plt

ehvi = loadres(r'/docs/exp/exp_res/BraninCurrin/EHVI', 'BraninCurrin', 'out_of_sample')
random = loadres(r'/docs/exp/exp_res/BraninCurrin/Random', 'BraninCurrin', 'out_of_sample')
mesmo_res_mc_1 = loadres(r'/docs/exp/exp_res/BraninCurrin/MESMO/MC1', 'BraninCurrin', 'out_of_sample')
pfes_res_mc_1 = loadres(r'/docs/exp/exp_res/BraninCurrin/BCPFES/MC1', 'BraninCurrin', 'out_of_sample')
pfes_ibo_res_mc_1 = loadres(r'/docs/exp/exp_res/BraninCurrin/BCPFES_IBO/MC1', 'BraninCurrin', 'out_of_sample')
pfes_res_mc_10 = loadres(r'/docs/exp/exp_res/BraninCurrin/BCPFES/MC10', 'BraninCurrin', 'out_of_sample')
pfes_ibo_res_mc_10 = loadres(r'/docs/exp/exp_res/BraninCurrin/BCPFES_IBO/MC10', 'BraninCurrin', 'out_of_sample')

plt.plot(np.arange(len(pfes_res_mc_1[0])), np.mean(pfes_res_mc_1, 0), label='PFES MC 1')
plt.plot(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_1, 0), label='PFES-IBO MC 1')
# plt.fill_between(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_1, 0) - 1.96 * np.std(pfes_ibo_res_mc_1, 0), np.mean(pfes_ibo_res_mc_1, 0) + 1.96 * np.std(pfes_ibo_res_mc_1, 0))
plt.plot(np.arange(len(pfes_res_mc_1[0])), np.mean(pfes_res_mc_10, 0), label='PFES MC 10')
plt.plot(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_10, 0), label='PFES-IBO MC 10')
plt.plot(np.arange(len(ehvi[0])), np.mean(ehvi, 0), label='EHVI')
plt.plot(np.arange(len(random[0])), np.mean(random, 0), label='Random')
plt.plot(np.arange(len(mesmo_res_mc_1[0])), np.mean(mesmo_res_mc_1, 0), label='MESMO 1')

plt.legend()
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('log HV difference')
plt.title('BraninCurrin out of sample regret')

# %%
from matplotlib import pyplot as plt

ehvi = loadres(r'/docs/exp/exp_res/BraninCurrin/EHVI', 'BraninCurrin', 'in_sample')
random = loadres(r'/docs/exp/exp_res/BraninCurrin/Random', 'BraninCurrin', 'in_sample')
mesmo_res_mc_1 = loadres(r'/docs/exp/exp_res/BraninCurrin/MESMO/MC1', 'BraninCurrin', 'in_sample')
pfes_res_mc_1 = loadres(r'/docs/exp/exp_res/BraninCurrin/BCPFES/MC1', 'BraninCurrin', 'in_sample')
pfes_ibo_res_mc_1 = loadres(r'/docs/exp/exp_res/BraninCurrin/BCPFES_IBO/MC1', 'BraninCurrin', 'in_sample')
pfes_res_mc_10 = loadres(r'/docs/exp/exp_res/BraninCurrin/BCPFES/MC10', 'BraninCurrin', 'in_sample')
pfes_ibo_res_mc_10 = loadres(r'/docs/exp/exp_res/BraninCurrin/BCPFES_IBO/MC10', 'BraninCurrin', 'in_sample')

plt.plot(np.arange(len(pfes_res_mc_1[0])), np.mean(pfes_res_mc_1, 0), label='PFES MC 1')
plt.plot(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_1, 0), label='PFES-IBO MC 1')
# plt.fill_between(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_1, 0) - 1.96 * np.std(pfes_ibo_res_mc_1, 0), np.mean(pfes_ibo_res_mc_1, 0) + 1.96 * np.std(pfes_ibo_res_mc_1, 0))
plt.plot(np.arange(len(pfes_res_mc_1[0])), np.mean(pfes_res_mc_10, 0), label='PFES MC 10')
plt.plot(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_10, 0), label='PFES-IBO MC 10')
# plt.plot(np.arange(len(ehvi[0])), np.mean(ehvi, 0), label='EHVI')
plt.plot(np.arange(len(random[0])), np.mean(random, 0), label='Random')
plt.plot(np.arange(len(mesmo_res_mc_1[0])), np.mean(mesmo_res_mc_1, 0), label='MESMO 1')

plt.legend()
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('log HV difference')
plt.title('BraninCurrin in sample PF regret')

# %% [markdown]
# ## DTLZ2

# %%
from matplotlib import pyplot as plt

ehvi = loadres(r'/docs/exp/exp_res/DTLZ2/EHVI', 'DTLZ2', 'in_sample', metric='RelHv')
# random = loadres(r'C:\Users\Administrator\Desktop\trieste_pfes\docs\exp\exp_res\DTLZ3\Random', 'DTLZ3', 'in_sample')
mesmo_res_mc_1 = loadres(r'/docs/exp/exp_res/DTLZ2/MESMO/MC1', 'DTLZ2', 'in_sample', metric='RelHv')
mesmo_res_mc_10 = loadres(r'/docs/exp/exp_res/DTLZ2/MESMO/MC10', 'DTLZ2', 'in_sample', metric='RelHv')
pfes_res_mc_1 = loadres(r'/docs/exp/exp_res/DTLZ2/BCPFES/MC1', 'DTLZ2', 'in_sample', metric='RelHv')
pfes_ibo_res_mc_1 = loadres(r'/docs/exp/exp_res/DTLZ2/BCPFES_IBO/MC1', 'DTLZ2', 'in_sample', metric='RelHv')
pfes_res_mc_10 = loadres(r'/docs/exp/exp_res/DTLZ2/BCPFES/MC10', 'DTLZ2', 'in_sample', 'RelHv')
pfes_ibo_res_mc_10 = loadres(r'/docs/exp/exp_res/DTLZ2/BCPFES_IBO/MC10', 'DTLZ2', 'in_sample', metric='RelHv')

plt.plot(np.arange(len(pfes_res_mc_1[0])), np.mean(pfes_res_mc_1, 0), label='PFES MC 1')
plt.plot(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_1, 0), label='PFES-IBO MC 1')
# plt.fill_between(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_1, 0) - np.std(pfes_ibo_res_mc_1, 0), np.mean(pfes_ibo_res_mc_1, 0) +  np.std(pfes_ibo_res_mc_1, 0))
plt.plot(np.arange(len(pfes_res_mc_1[0])), np.mean(pfes_res_mc_10, 0), label='PFES MC 10')
plt.plot(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_10, 0), label='PFES-IBO MC 10')
plt.plot(np.arange(len(ehvi[0])), np.mean(ehvi, 0), label='EHVI')
# plt.plot(np.arange(len(random[0])), np.mean(random, 0), label='Random')
plt.plot(np.arange(len(mesmo_res_mc_1[0])), np.mean(mesmo_res_mc_1, 0), label='MESMO 1')
plt.plot(np.arange(len(mesmo_res_mc_1[0])), np.mean(mesmo_res_mc_10, 0), label='MESMO 10')

plt.legend()
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('RelHv')
plt.title('DTLZ2 in sample regret')
# plt.show()
plt.savefig('DTLZ2_In_sample_regret1.png')

# %%
from matplotlib import pyplot as plt

ehvi = loadres(r'/docs/exp/exp_res/DTLZ2/EHVI', 'DTLZ2', 'in_sample', metric='AVD')
# random = loadres(r'C:\Users\Administrator\Desktop\trieste_pfes\docs\exp\exp_res\DTLZ3\Random', 'DTLZ3', 'in_sample')
mesmo_res_mc_1 = loadres(r'/docs/exp/exp_res/DTLZ2/MESMO/MC1', 'DTLZ2', 'in_sample', metric='AVD')
mesmo_res_mc_10 = loadres(r'/docs/exp/exp_res/DTLZ2/MESMO/MC10', 'DTLZ2', 'in_sample', metric='AVD')
pfes_res_mc_1 = loadres(r'/docs/exp/exp_res/DTLZ2/BCPFES/MC1', 'DTLZ2', 'in_sample', metric='AVD')
pfes_ibo_res_mc_1 = loadres(r'/docs/exp/exp_res/DTLZ2/BCPFES_IBO/MC1', 'DTLZ2', 'in_sample', metric='AVD')
pfes_res_mc_10 = loadres(r'/docs/exp/exp_res/DTLZ2/BCPFES/MC10', 'DTLZ2', 'in_sample', 'AVD')
pfes_ibo_res_mc_10 = loadres(r'/docs/exp/exp_res/DTLZ2/BCPFES_IBO/MC10', 'DTLZ2', 'in_sample', metric='AVD')

plt.plot(np.arange(len(pfes_res_mc_1[0])), np.mean(pfes_res_mc_1, 0), label='PFES MC 1')
plt.plot(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_1, 0), label='PFES-IBO MC 1')
# plt.fill_between(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_1, 0) - 1.96 * np.std(pfes_ibo_res_mc_1, 0), np.mean(pfes_ibo_res_mc_1, 0) + 1.96 * np.std(pfes_ibo_res_mc_1, 0))
plt.plot(np.arange(len(pfes_res_mc_1[0])), np.mean(pfes_res_mc_10, 0), label='PFES MC 10')
plt.plot(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_10, 0), label='PFES-IBO MC 10')
plt.plot(np.arange(len(ehvi[0])), np.mean(ehvi, 0), label='EHVI')
# plt.plot(np.arange(len(random[0])), np.mean(random, 0), label='Random')
plt.plot(np.arange(len(mesmo_res_mc_1[0])), np.mean(mesmo_res_mc_1, 0), label='MESMO 1')
plt.plot(np.arange(len(mesmo_res_mc_1[0])), np.mean(mesmo_res_mc_10, 0), label='MESMO 10')

plt.legend()
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Average Hausdauff Distance')
plt.title('DTLZ2 in sample regret AVD')
# plt.show()
plt.savefig('DTLZ2_In_sample_regret2.png')

# %% [markdown]
# ## Constr-EX

# %%
from matplotlib import pyplot as plt


random = loadres(r'/docs/exp/exp_res/Constr_Ex/Random', 'Constr_Ex', 'in_sample')
pfes_res_mc_1 = loadres(r'/docs/exp/exp_res/Constr_Ex/BCPFES/MC1', 'Constr_Ex', 'in_sample')
pfes_ibo_res_mc_1 = loadres(r'/docs/exp/exp_res/Constr_Ex/BCPFES_IBO/MC1', 'Constr_Ex', 'in_sample')
cehvi = loadres(r'/docs/exp/exp_res/Constr_Ex/CEHVI', 'Constr_Ex', 'in_sample')
pfes_res_mc_10 = loadres(r'/docs/exp/exp_res/Constr_Ex/BCPFES/MC10', 'Constr_Ex', 'in_sample')
pfes_ibo_res_mc_10 = loadres(r'/docs/exp/exp_res/Constr_Ex/BCPFES_IBO/MC10', 'Constr_Ex', 'in_sample')


plt.plot(np.arange(len(pfes_res_mc_1[0])), np.mean(pfes_res_mc_1, 0), label='CPFES MC 1')
plt.plot(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_1, 0), label='CPFES-IBO MC 1')
# plt.fill_between(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_1, 0) - 1.96 * np.std(pfes_ibo_res_mc_1, 0), np.mean(pfes_ibo_res_mc_1, 0) + 1.96 * np.std(pfes_ibo_res_mc_1, 0))
plt.plot(np.arange(len(pfes_res_mc_1[0])), np.mean(pfes_res_mc_10, 0), label='PFES MC 10')
plt.plot(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_10, 0), label='PFES-IBO MC 10')
plt.plot(np.arange(len(cehvi[0])), np.mean(cehvi, 0), label='CEHVI')
plt.plot(np.arange(len(random[0])), np.mean(random, 0), label='Random')
# plt.plot(np.arange(len(mesmo_res_mc_1[0])), np.mean(mesmo_res_mc_1, 0), label='MESMO 1')

plt.legend()
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('log HV difference')
plt.title('Constr-Ex: 2-d, 2-con, 2-obj in-sample performance')
# plt.savefig('Constr_Ex_res.png')

# %%
from matplotlib import pyplot as plt


random = loadres(r'/docs/exp/exp_res/Constr_Ex/Random', 'Constr_Ex', 'out_of_sample')
pfes_res_mc_1 = loadres(r'/docs/exp/exp_res/Constr_Ex/BCPFES/MC1', 'Constr_Ex', 'out_of_sample')
pfes_ibo_res_mc_1 = loadres(r'/docs/exp/exp_res/Constr_Ex/BCPFES_IBO/MC1', 'Constr_Ex', 'out_of_sample')
cehvi = loadres(r'/docs/exp/exp_res/Constr_Ex/CEHVI', 'Constr_Ex', 'out_of_sample')
pfes_res_mc_10 = loadres(r'/docs/exp/exp_res/Constr_Ex/BCPFES/MC10', 'Constr_Ex', 'out_of_sample')
pfes_ibo_res_mc_10 = loadres(r'/docs/exp/exp_res/Constr_Ex/BCPFES_IBO/MC10', 'Constr_Ex', 'out_of_sample')


plt.plot(np.arange(len(pfes_res_mc_1[0])), np.mean(pfes_res_mc_1, 0), label='CPFES MC 1')
plt.plot(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_1, 0), label='CPFES-IBO MC 1')
# plt.fill_between(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_1, 0) - 1.96 * np.std(pfes_ibo_res_mc_1, 0), np.mean(pfes_ibo_res_mc_1, 0) + 1.96 * np.std(pfes_ibo_res_mc_1, 0))
plt.plot(np.arange(len(pfes_res_mc_1[0])), np.mean(pfes_res_mc_10, 0), label='PFES MC 10')
plt.plot(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_10, 0), label='PFES-IBO MC 10')
plt.plot(np.arange(len(cehvi[0])), np.mean(cehvi, 0), label='CEHVI')
plt.plot(np.arange(len(random[0])), np.mean(random, 0), label='Random')
# plt.plot(np.arange(len(mesmo_res_mc_1[0])), np.mean(mesmo_res_mc_1, 0), label='MESMO 1')

plt.legend()
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('log HV difference')
plt.title('Constr-Ex: 2-d, 2-con, 2-obj in-sample performance')
# plt.savefig('Constr_Ex_res.png')

# %% [markdown]
# ## TNK

# %%
from matplotlib import pyplot as plt


random = loadres(r'/docs/exp/exp_res/TNK/Random', 'TNK', 'in_sample')
pfes_res_mc_1 = loadres(r'/docs/exp/exp_res/TNK/BCPFES/MC1', 'TNK', 'in_sample')
pfes_ibo_res_mc_1 = loadres(r'/docs/exp/exp_res/TNK/BCPFES_IBO/MC1', 'TNK', 'in_sample')
cehvi = loadres(r'/docs/exp/exp_res/TNK/CEHVI', 'TNK', 'in_sample')

plt.plot(np.arange(len(pfes_res_mc_1[0])), np.mean(pfes_res_mc_1, 0), label='CPFES MC 1')
plt.plot(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_1, 0), label='CPFES-IBO MC 1')
plt.plot(np.arange(len(cehvi[0])), np.mean(cehvi, 0), label='CEHVI')
# plt.fill_between(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_1, 0) - 1.96 * np.std(pfes_ibo_res_mc_1, 0), np.mean(pfes_ibo_res_mc_1, 0) + 1.96 * np.std(pfes_ibo_res_mc_1, 0))
# plt.plot(np.arange(len(pfes_res_mc_1[0])), np.mean(pfes_res_mc_10, 0), label='PFES MC 10')
# plt.plot(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_10, 0), label='PFES-IBO MC 10')
# plt.plot(np.arange(len(ehvi[0])), np.mean(ehvi, 0), label='EHVI')
plt.plot(np.arange(len(random[0])), np.mean(random, 0), label='Random')
# plt.plot(np.arange(len(mesmo_res_mc_1[0])), np.mean(mesmo_res_mc_1, 0), label='MESMO 1')

plt.legend()
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('log HV difference')
plt.title('TNK: 2-d, 2-con, 2-obj in-sample performance')
plt.savefig('TNK_res.png')

# %% [markdown]
# ## Osy

# %%
from matplotlib import pyplot as plt


random = loadres(r'/docs/exp/exp_res/Osy/Random', 'Osy', 'in_sample')
pfes_res_mc_1 = loadres(r'/docs/exp/exp_res/Osy/BCPFES/MC1', 'Osy', 'in_sample')
pfes_ibo_res_mc_1 = loadres(r'/docs/exp/exp_res/Osy/BCPFES_IBO/MC1', 'Osy', 'in_sample')
cehvi = loadres(r'/docs/exp/exp_res/Osy/CEHVI', 'Osy', 'in_sample')

plt.plot(np.arange(len(pfes_res_mc_1[0])), np.mean(pfes_res_mc_1, 0), label='CPFES MC 1')
plt.plot(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_1, 0), label='CPFES-IBO MC 1')
# plt.fill_between(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_1, 0) - 1.96 * np.std(pfes_ibo_res_mc_1, 0), np.mean(pfes_ibo_res_mc_1, 0) + 1.96 * np.std(pfes_ibo_res_mc_1, 0))
# plt.plot(np.arange(len(pfes_res_mc_1[0])), np.mean(pfes_res_mc_10, 0), label='PFES MC 10')
# plt.plot(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_10, 0), label='PFES-IBO MC 10')
plt.plot(np.arange(len(cehvi[0])), np.mean(cehvi, 0), label='CEHVI')
plt.plot(np.arange(len(random[0])), np.mean(random, 0), label='Random')
# plt.plot(np.arange(len(mesmo_res_mc_1[0])), np.mean(mesmo_res_mc_1, 0), label='MESMO 1')

plt.legend()
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('AVD')
plt.yscale('log', base=100)
plt.title('Osy: 6-d, 6-con, 2-obj in-sample performance')
plt.savefig('Osy_res.png')

# %%
from matplotlib import pyplot as plt


random = loadres(r'/docs/exp/exp_res/Osy/Random', 'Osy', 'in_sample', metric='AVD')
pfes_res_mc_1 = loadres(r'/docs/exp/exp_res/Osy/BCPFES/MC1', 'Osy', 'in_sample', metric='AVD')
pfes_ibo_res_mc_1 = loadres(r'/docs/exp/exp_res/Osy/BCPFES_IBO/MC1', 'Osy', 'in_sample', metric='AVD')
cehvi = loadres(r'/docs/exp/exp_res/Osy/CEHVI', 'Osy', 'in_sample', metric='AVD')

plt.plot(np.arange(len(pfes_res_mc_1[0])), np.mean(pfes_res_mc_1, 0), label='CPFES MC 1')
plt.plot(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_1, 0), label='CPFES-IBO MC 1')
# plt.fill_between(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_1, 0) - 1.96 * np.std(pfes_ibo_res_mc_1, 0), np.mean(pfes_ibo_res_mc_1, 0) + 1.96 * np.std(pfes_ibo_res_mc_1, 0))
# plt.plot(np.arange(len(pfes_res_mc_1[0])), np.mean(pfes_res_mc_10, 0), label='PFES MC 10')
# plt.plot(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_10, 0), label='PFES-IBO MC 10')
plt.plot(np.arange(len(cehvi[0])), np.mean(cehvi, 0), label='CEHVI')
plt.plot(np.arange(len(random[0])), np.mean(random, 0), label='Random')
# plt.plot(np.arange(len(mesmo_res_mc_1[0])), np.mean(mesmo_res_mc_1, 0), label='MESMO 1')

plt.legend()
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('AVD')
plt.yscale('log', base=100)
plt.title('Osy: 6-d, 6-con, 2-obj in-sample performance')
plt.savefig('Osy_res_AVD.png')

# %% [markdown]
# ## SRN

# %%
from matplotlib import pyplot as plt


random = loadres(r'/docs/exp/exp_res/SRN/Random', 'SRN', 'in_sample', 'RelHv')
pfes_res_mc_1 = loadres(r'/docs/exp/exp_res/SRN/BCPFES/MC1', 'SRN', 'in_sample', 'RelHv')
pfes_ibo_res_mc_1 = loadres(r'/docs/exp/exp_res/SRN/BCPFES_IBO/MC1', 'SRN', 'in_sample', 'RelHv')
cehvi = loadres(r'/docs/exp/exp_res/SRN/CEHVI', 'SRN', 'in_sample', 'RelHv')


plt.plot(np.arange(len(pfes_res_mc_1[0])), np.mean(pfes_res_mc_1, 0), label='CPFES MC 1')
plt.plot(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_1, 0), label='CPFES-IBO MC 1')
# plt.fill_between(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_1, 0) - 1.96 * np.std(pfes_ibo_res_mc_1, 0), np.mean(pfes_ibo_res_mc_1, 0) + 1.96 * np.std(pfes_ibo_res_mc_1, 0))
# plt.plot(np.arange(len(pfes_res_mc_1[0])), np.mean(pfes_res_mc_10, 0), label='PFES MC 10')
# plt.plot(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_10, 0), label='PFES-IBO MC 10')
plt.plot(np.arange(len(cehvi[0])), np.mean(cehvi, 0), label='CEHVI')
plt.plot(np.arange(len(random[0])), np.mean(random, 0), label='Random')
# plt.plot(np.arange(len(mesmo_res_mc_1[0])), np.mean(mesmo_res_mc_1, 0), label='MESMO 1')

plt.legend()
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('RelHv')
plt.title('SRN: 2-d, 2-con, 2-obj in-sample performance')
plt.savefig('SRN_RelHv_res.png')

# %%
from matplotlib import pyplot as plt


random = loadres(r'/docs/exp/exp_res/SRN/Random', 'SRN', 'in_sample')
pfes_res_mc_1 = loadres(r'/docs/exp/exp_res/SRN/BCPFES/MC1', 'SRN', 'in_sample')
pfes_ibo_res_mc_1 = loadres(r'/docs/exp/exp_res/SRN/BCPFES_IBO/MC1', 'SRN', 'in_sample')
cehvi = loadres(r'/docs/exp/exp_res/SRN/CEHVI', 'SRN', 'in_sample')


plt.plot(np.arange(len(pfes_res_mc_1[0])), np.mean(pfes_res_mc_1, 0), label='CPFES MC 1')
plt.plot(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_1, 0), label='CPFES-IBO MC 1')
# plt.fill_between(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_1, 0) - 1.96 * np.std(pfes_ibo_res_mc_1, 0), np.mean(pfes_ibo_res_mc_1, 0) + 1.96 * np.std(pfes_ibo_res_mc_1, 0))
# plt.plot(np.arange(len(pfes_res_mc_1[0])), np.mean(pfes_res_mc_10, 0), label='PFES MC 10')
# plt.plot(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_10, 0), label='PFES-IBO MC 10')
plt.plot(np.arange(len(cehvi[0])), np.mean(cehvi, 0), label='CEHVI')
plt.plot(np.arange(len(random[0])), np.mean(random, 0), label='Random')
# plt.plot(np.arange(len(mesmo_res_mc_1[0])), np.mean(mesmo_res_mc_1, 0), label='MESMO 1')

plt.legend()
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('log HV difference')
plt.title('SRN: 2-d, 2-con, 2-obj in-sample performance')
plt.savefig('SRN_res.png')

# %%
from matplotlib import pyplot as plt


random = loadres(r'/docs/exp/exp_res/SRN/Random', 'SRN', 'in_sample', metric='AVD')
pfes_res_mc_1 = loadres(r'/docs/exp/exp_res/SRN/BCPFES/MC1', 'SRN', 'in_sample', metric='AVD')
pfes_ibo_res_mc_1 = loadres(r'/docs/exp/exp_res/SRN/BCPFES_IBO/MC1', 'SRN', 'in_sample', metric='AVD')
cehvi = loadres(r'/docs/exp/exp_res/SRN/CEHVI', 'SRN', 'in_sample', metric='AVD')


plt.plot(np.arange(len(pfes_res_mc_1[0])), np.mean(pfes_res_mc_1, 0), label='CPFES MC 1')
plt.plot(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_1, 0), label='CPFES-IBO MC 1')
# plt.fill_between(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_1, 0) - 1.96 * np.std(pfes_ibo_res_mc_1, 0), np.mean(pfes_ibo_res_mc_1, 0) + 1.96 * np.std(pfes_ibo_res_mc_1, 0))
# plt.plot(np.arange(len(pfes_res_mc_1[0])), np.mean(pfes_res_mc_10, 0), label='PFES MC 10')
# plt.plot(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_10, 0), label='PFES-IBO MC 10')
plt.plot(np.arange(len(cehvi[0])), np.mean(cehvi, 0), label='CEHVI')
plt.plot(np.arange(len(random[0])), np.mean(random, 0), label='Random')
# plt.plot(np.arange(len(mesmo_res_mc_1[0])), np.mean(mesmo_res_mc_1, 0), label='MESMO 1')

plt.legend()
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Average Hausdauff')
plt.title('SRN: 2-d, 2-con, 2-obj in-sample performance')
plt.savefig('SRN_AVD_res.png')

# %% [markdown]
# ## TwoBarTruss

# %%
from matplotlib import pyplot as plt


random = loadres(r'/docs/exp/exp_res/TwoBarTruss/Random', 'TwoBarTruss', 'in_sample')
pfes_res_mc_1 = loadres(r'/docs/exp/exp_res/TwoBarTruss/BCPFES/MC1', 'TwoBarTruss', 'in_sample')
pfes_ibo_res_mc_1 = loadres(r'/docs/exp/exp_res/TwoBarTruss/BCPFES_IBO/MC1', 'TwoBarTruss', 'in_sample')
cehvi = loadres(r'/docs/exp/exp_res/TwoBarTruss/CEHVI', 'TwoBarTruss', 'in_sample')

plt.plot(np.arange(len(pfes_res_mc_1[0])), np.mean(pfes_res_mc_1, 0), label='CPFES MC 1')
plt.plot(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_1, 0), label='CPFES-IBO MC 1')
# plt.fill_between(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_1, 0) - 1.96 * np.std(pfes_ibo_res_mc_1, 0), np.mean(pfes_ibo_res_mc_1, 0) + 1.96 * np.std(pfes_ibo_res_mc_1, 0))
# plt.plot(np.arange(len(pfes_res_mc_1[0])), np.mean(pfes_res_mc_10, 0), label='PFES MC 10')
# plt.plot(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_10, 0), label='PFES-IBO MC 10')
plt.plot(np.arange(len(cehvi[0])), np.mean(cehvi, 0), label='CEHVI')
plt.plot(np.arange(len(random[0])), np.mean(random, 0), label='Random')
# plt.plot(np.arange(len(mesmo_res_mc_1[0])), np.mean(mesmo_res_mc_1, 0), label='MESMO 1')

plt.legend()
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('LogHvDiff')
plt.yscale('log', base=100)
plt.title('TwoBarTruss: 2-d, 2-con, 2-obj in-sample performance')
# plt.savefig('TwoBarTruss_res.png')

# %%
from matplotlib import pyplot as plt


random = loadres(r'/docs/exp/exp_res/TwoBarTruss/Random', 'TwoBarTruss', 'in_sample', 'RelHv')
pfes_res_mc_1 = loadres(r'/docs/exp/exp_res/TwoBarTruss/BCPFES/MC1', 'TwoBarTruss', 'in_sample', 'RelHv')
pfes_ibo_res_mc_1 = loadres(r'/docs/exp/exp_res/TwoBarTruss/BCPFES_IBO/MC1', 'TwoBarTruss', 'in_sample', 'RelHv')
cehvi = loadres(r'/docs/exp/exp_res/TwoBarTruss/CEHVI', 'TwoBarTruss', 'in_sample', 'RelHv')

plt.plot(np.arange(len(pfes_res_mc_1[0])), np.mean(pfes_res_mc_1, 0), label='CPFES MC 1')
plt.plot(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_1, 0), label='CPFES-IBO MC 1')
# plt.fill_between(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_1, 0) - 1.96 * np.std(pfes_ibo_res_mc_1, 0), np.mean(pfes_ibo_res_mc_1, 0) + 1.96 * np.std(pfes_ibo_res_mc_1, 0))
# plt.plot(np.arange(len(pfes_res_mc_1[0])), np.mean(pfes_res_mc_10, 0), label='PFES MC 10')
# plt.plot(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_10, 0), label='PFES-IBO MC 10')
plt.plot(np.arange(len(cehvi[0])), np.mean(cehvi, 0), label='CEHVI')
plt.plot(np.arange(len(random[0])), np.mean(random, 0), label='Random')
# plt.plot(np.arange(len(mesmo_res_mc_1[0])), np.mean(mesmo_res_mc_1, 0), label='MESMO 1')

plt.legend()
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('RelHv')

# %%
from matplotlib import pyplot as plt


random = loadres(r'/docs/exp/exp_res/TwoBarTruss/Random', 'TwoBarTruss', 'in_sample', 'AVD')
pfes_res_mc_1 = loadres(r'/docs/exp/exp_res/TwoBarTruss/BCPFES/MC1', 'TwoBarTruss', 'in_sample', 'AVD')
pfes_ibo_res_mc_1 = loadres(r'/docs/exp/exp_res/TwoBarTruss/BCPFES_IBO/MC1', 'TwoBarTruss', 'in_sample', 'AVD')
cehvi = loadres(r'/docs/exp/exp_res/TwoBarTruss/CEHVI', 'TwoBarTruss', 'in_sample', 'AVD')

plt.plot(np.arange(len(pfes_res_mc_1[0])), np.mean(pfes_res_mc_1, 0), label='CPFES MC 1')
plt.plot(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_1, 0), label='CPFES-IBO MC 1')
# plt.fill_between(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_1, 0) - 1.96 * np.std(pfes_ibo_res_mc_1, 0), np.mean(pfes_ibo_res_mc_1, 0) + 1.96 * np.std(pfes_ibo_res_mc_1, 0))
# plt.plot(np.arange(len(pfes_res_mc_1[0])), np.mean(pfes_res_mc_10, 0), label='PFES MC 10')
# plt.plot(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_10, 0), label='PFES-IBO MC 10')
plt.plot(np.arange(len(cehvi[0])), np.mean(cehvi, 0), label='CEHVI')
plt.plot(np.arange(len(random[0])), np.mean(random, 0), label='Random')
# plt.plot(np.arange(len(mesmo_res_mc_1[0])), np.mean(mesmo_res_mc_1, 0), label='MESMO 1')

plt.legend()
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('AVD')

# %%
from matplotlib import pyplot as plt


random = loadres(r'/docs/exp/exp_res/TwoBarTruss/Random', 'TwoBarTruss', 'in_sample', 'EPS')
pfes_res_mc_1 = loadres(r'/docs/exp/exp_res/TwoBarTruss/BCPFES/MC1', 'TwoBarTruss', 'in_sample', 'EPS')
pfes_ibo_res_mc_1 = loadres(r'/docs/exp/exp_res/TwoBarTruss/BCPFES_IBO/MC1', 'TwoBarTruss', 'in_sample', 'EPS')
cehvi = loadres(r'/docs/exp/exp_res/TwoBarTruss/CEHVI', 'TwoBarTruss', 'in_sample', 'EPS')

plt.plot(np.arange(len(pfes_res_mc_1[0])), np.mean(pfes_res_mc_1, 0), label='CPFES MC 1')
plt.plot(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_1, 0), label='CPFES-IBO MC 1')
# plt.fill_between(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_1, 0) - 1.96 * np.std(pfes_ibo_res_mc_1, 0), np.mean(pfes_ibo_res_mc_1, 0) + 1.96 * np.std(pfes_ibo_res_mc_1, 0))
# plt.plot(np.arange(len(pfes_res_mc_1[0])), np.mean(pfes_res_mc_10, 0), label='PFES MC 10')
# plt.plot(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_10, 0), label='PFES-IBO MC 10')
plt.plot(np.arange(len(cehvi[0])), np.mean(cehvi, 0), label='CEHVI')
plt.plot(np.arange(len(random[0])), np.mean(random, 0), label='Random')
# plt.plot(np.arange(len(mesmo_res_mc_1[0])), np.mean(mesmo_res_mc_1, 0), label='MESMO 1')

plt.legend()
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('EPS')

# %% [markdown]
# ### WeldedBeam Design

# %%

# %% [markdown]
# ## DTLZ3

# %%
from matplotlib import pyplot as plt

ehvi = loadres(r'/docs/exp/exp_res/DTLZ3/EHVI', 'DTLZ3', 'in_sample', metric='RelHv')
# random = loadres(r'C:\Users\Administrator\Desktop\trieste_pfes\docs\exp\exp_res\DTLZ3\Random', 'DTLZ3', 'in_sample')
mesmo_res_mc_1 = loadres(r'/docs/exp/exp_res/DTLZ3/MESMO/MC1', 'DTLZ3', 'in_sample', metric='RelHv')
mesmo_res_mc_10 = loadres(r'/docs/exp/exp_res/DTLZ3/MESMO/MC10', 'DTLZ3', 'in_sample', metric='RelHv')
pfes_res_mc_1 = loadres(r'/docs/exp/exp_res/DTLZ3/BCPFES/MC1', 'DTLZ3', 'in_sample', metric='RelHv')
pfes_ibo_res_mc_1 = loadres(r'/docs/exp/exp_res/DTLZ3/BCPFES_IBO/MC1', 'DTLZ3', 'in_sample', metric='RelHv')
pfes_res_mc_10 = loadres(r'/docs/exp/exp_res/DTLZ3/BCPFES/MC10', 'DTLZ3', 'in_sample', 'RelHv')
pfes_ibo_res_mc_10 = loadres(r'/docs/exp/exp_res/DTLZ3/BCPFES_IBO/MC10', 'DTLZ3', 'in_sample', metric='RelHv')

plt.plot(np.arange(len(pfes_res_mc_1[0])), np.mean(pfes_res_mc_1, 0), label='PFES MC 1')
plt.plot(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_1, 0), label='PFES-IBO MC 1')
# plt.fill_between(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_1, 0) - np.std(pfes_ibo_res_mc_1, 0), np.mean(pfes_ibo_res_mc_1, 0) +  np.std(pfes_ibo_res_mc_1, 0))
plt.plot(np.arange(len(pfes_res_mc_1[0])), np.mean(pfes_res_mc_10, 0), label='PFES MC 10')
plt.plot(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_10, 0), label='PFES-IBO MC 10')
plt.plot(np.arange(len(ehvi[0])), np.mean(ehvi, 0), label='EHVI')
# plt.plot(np.arange(len(random[0])), np.mean(random, 0), label='Random')
plt.plot(np.arange(len(mesmo_res_mc_1[0])), np.mean(mesmo_res_mc_1, 0), label='MESMO 1')
plt.plot(np.arange(len(mesmo_res_mc_1[0])), np.mean(mesmo_res_mc_10, 0), label='MESMO 10')

plt.legend()
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('log HV difference')
plt.title('DTLZ3 in sample regret')
plt.savefig('DTLZ3_In_sample_regret1.png')

# %%
from matplotlib import pyplot as plt

ehvi = loadres(r'/docs/exp/exp_res/DTLZ3/EHVI', 'DTLZ3', 'in_sample', metric='AVD')
# random = loadres(r'C:\Users\Administrator\Desktop\trieste_pfes\docs\exp\exp_res\DTLZ3\Random', 'DTLZ3', 'in_sample')
mesmo_res_mc_1 = loadres(r'/docs/exp/exp_res/DTLZ3/MESMO/MC1', 'DTLZ3', 'in_sample', metric='AVD')
mesmo_res_mc_10 = loadres(r'/docs/exp/exp_res/DTLZ3/MESMO/MC10', 'DTLZ3', 'in_sample', metric='AVD')
pfes_res_mc_1 = loadres(r'/docs/exp/exp_res/DTLZ3/BCPFES/MC1', 'DTLZ3', 'in_sample', metric='AVD')
pfes_ibo_res_mc_1 = loadres(r'/docs/exp/exp_res/DTLZ3/BCPFES_IBO/MC1', 'DTLZ3', 'in_sample', metric='AVD')
pfes_res_mc_10 = loadres(r'/docs/exp/exp_res/DTLZ3/BCPFES/MC10', 'DTLZ3', 'in_sample', 'AVD')
pfes_ibo_res_mc_10 = loadres(r'/docs/exp/exp_res/DTLZ3/BCPFES_IBO/MC10', 'DTLZ3', 'in_sample', metric='AVD')

plt.plot(np.arange(len(pfes_res_mc_1[0])), np.mean(pfes_res_mc_1, 0), label='PFES MC 1')
plt.plot(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_1, 0), label='PFES-IBO MC 1')
# plt.fill_between(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_1, 0) - 1.96 * np.std(pfes_ibo_res_mc_1, 0), np.mean(pfes_ibo_res_mc_1, 0) + 1.96 * np.std(pfes_ibo_res_mc_1, 0))
plt.plot(np.arange(len(pfes_res_mc_1[0])), np.mean(pfes_res_mc_10, 0), label='PFES MC 10')
plt.plot(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_10, 0), label='PFES-IBO MC 10')
plt.plot(np.arange(len(ehvi[0])), np.mean(ehvi, 0), label='EHVI')
# plt.plot(np.arange(len(random[0])), np.mean(random, 0), label='Random')
plt.plot(np.arange(len(mesmo_res_mc_1[0])), np.mean(mesmo_res_mc_1, 0), label='MESMO 1')
plt.plot(np.arange(len(mesmo_res_mc_1[0])), np.mean(mesmo_res_mc_10, 0), label='MESMO 10')

plt.legend()
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Average Hausdauff Distance')
plt.title('DTLZ3 in sample regret AVD')
plt.savefig('DTLZ3_In_sample_regret2.png')

# %% [markdown]
# ## DTLZ4

# %%
from matplotlib import pyplot as plt

ehvi = loadres(r'/docs/exp/exp_res/DTLZ4/EHVI', 'DTLZ4', 'in_sample', metric='logHvDiff')
# random = loadres(r'C:\Users\Administrator\Desktop\trieste_pfes\docs\exp\exp_res\DTLZ3\Random', 'DTLZ3', 'in_sample')
mesmo_res_mc_1 = loadres(r'/docs/exp/exp_res/DTLZ4/MESMO/MC1', 'DTLZ4', 'in_sample', metric='logHvDiff')
mesmo_res_mc_10 = loadres(r'/docs/exp/exp_res/DTLZ4/MESMO/MC10', 'DTLZ4', 'in_sample', metric='logHvDiff')
pfes_res_mc_1 = loadres(r'/docs/exp/exp_res/DTLZ4/BCPFES/MC1', 'DTLZ4', 'in_sample', metric='logHvDiff')
pfes_ibo_res_mc_1 = loadres(r'/docs/exp/exp_res/DTLZ4/BCPFES_IBO/MC1', 'DTLZ4', 'in_sample', metric='logHvDiff')
pfes_res_mc_10 = loadres(r'/docs/exp/exp_res/DTLZ4/BCPFES/MC10', 'DTLZ4', 'in_sample')
pfes_ibo_res_mc_10 = loadres(r'/docs/exp/exp_res/DTLZ4/BCPFES_IBO/MC10', 'DTLZ4', 'in_sample')

plt.plot(np.arange(len(pfes_res_mc_1[0])), np.mean(pfes_res_mc_1, 0), label='PFES MC 1')
plt.plot(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_1, 0), label='PFES-IBO MC 1')
# plt.fill_between(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_1, 0) - 1.96 * np.std(pfes_ibo_res_mc_1, 0), np.mean(pfes_ibo_res_mc_1, 0) + 1.96 * np.std(pfes_ibo_res_mc_1, 0))
plt.plot(np.arange(len(pfes_res_mc_1[0])), np.mean(pfes_res_mc_10, 0), label='PFES MC 10')
plt.plot(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_10, 0), label='PFES-IBO MC 10')
plt.plot(np.arange(len(ehvi[0])), np.mean(ehvi, 0), label='EHVI')
# plt.plot(np.arange(len(random[0])), np.mean(random, 0), label='Random')
plt.plot(np.arange(len(mesmo_res_mc_1[0])), np.mean(mesmo_res_mc_1, 0), label='MESMO 1')
plt.plot(np.arange(len(mesmo_res_mc_1[0])), np.mean(mesmo_res_mc_10, 0), label='MESMO 10')

plt.legend()
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('(log) logHvDiff')
plt.yscale('log')
plt.title('DTLZ4 in sample regret')
plt.savefig('DTLZ4_In_sample_regret1.png')

# %%
from matplotlib import pyplot as plt

ehvi = loadres(r'/docs/exp/exp_res/DTLZ4/EHVI', 'DTLZ4', 'in_sample', metric='AVD')
# random = loadres(r'C:\Users\Administrator\Desktop\trieste_pfes\docs\exp\exp_res\DTLZ3\Random', 'DTLZ3', 'in_sample')
mesmo_res_mc_1 = loadres(r'/docs/exp/exp_res/DTLZ4/MESMO/MC1', 'DTLZ4', 'in_sample', metric='AVD')
mesmo_res_mc_10 = loadres(r'/docs/exp/exp_res/DTLZ4/MESMO/MC10', 'DTLZ4', 'in_sample', metric='AVD')
pfes_res_mc_1 = loadres(r'/docs/exp/exp_res/DTLZ4/BCPFES/MC1', 'DTLZ4', 'in_sample', metric='AVD')
pfes_ibo_res_mc_1 = loadres(r'/docs/exp/exp_res/DTLZ4/BCPFES_IBO/MC1', 'DTLZ4', 'in_sample', metric='AVD')
pfes_res_mc_10 = loadres(r'/docs/exp/exp_res/DTLZ4/BCPFES/MC10', 'DTLZ4', 'in_sample', metric='AVD')
pfes_ibo_res_mc_10 = loadres(r'/docs/exp/exp_res/DTLZ4/BCPFES_IBO/MC10', 'DTLZ4', 'in_sample', metric='AVD')

plt.plot(np.arange(len(pfes_res_mc_1[0])), np.mean(pfes_res_mc_1, 0), label='PFES MC 1')
plt.plot(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_1, 0), label='PFES-IBO MC 1')
# plt.fill_between(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_1, 0) - 1.96 * np.std(pfes_ibo_res_mc_1, 0), np.mean(pfes_ibo_res_mc_1, 0) + 1.96 * np.std(pfes_ibo_res_mc_1, 0))
plt.plot(np.arange(len(pfes_res_mc_1[0])), np.mean(pfes_res_mc_10, 0), label='PFES MC 10')
plt.plot(np.arange(len(pfes_ibo_res_mc_1[0])), np.mean(pfes_ibo_res_mc_10, 0), label='PFES-IBO MC 10')
plt.plot(np.arange(len(ehvi[0])), np.mean(ehvi, 0), label='EHVI')
# plt.plot(np.arange(len(random[0])), np.mean(random, 0), label='Random')
plt.plot(np.arange(len(mesmo_res_mc_1[0])), np.mean(mesmo_res_mc_1, 0), label='MESMO 1')
plt.plot(np.arange(len(mesmo_res_mc_1[0])), np.mean(mesmo_res_mc_10, 0), label='MESMO 10')

plt.legend()
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Average Hausdauff Distance')
plt.title('DTLZ4 in sample regret')
plt.savefig('DTLZ4_In_sample_regret2.png')

# %% [markdown]
# # Potential Direction: Random Fourier Features Issue

# %%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gpflow


# %%
from trieste.space import Box


def h(x):
    return - 20 * (2 - 0.8 * tf.exp(-(((x - 0.35) / 0.25) ** 2)) - tf.exp(-(((x - 0.8) / 0.05) ** 2)) - 1.5)

ds = Box([0], [1])
X = ds.sample(30)
obj = h
Y = obj(X)

## generate test points for prediction
xx = np.linspace(0, 1, 100).reshape(100, 1)  # test points must be of shape (N, D)


## Add predict g by Kernel MC sampling (NOT INCLUDING THE DIAGONAL PART)
from trieste.data import Dataset
from trieste.models import create_model

single_obj_data = Dataset(X, Y)
variance = tf.math.reduce_variance(single_obj_data.observations)
kernel = gpflow.kernels.RBF(variance, lengthscales=[1.0] * 1)

# jitter = gpflow.kernels.White(1e-12)
gpr = gpflow.models.GPR((single_obj_data.query_points, single_obj_data.observations), kernel,
                        noise_variance=1e-5)
gpflow.utilities.set_trainable(gpr.likelihood, False)

m2 = create_model({
    "model": gpr,
    "optimizer": gpflow.optimizers.Scipy(),
    "optimizer_args": {
        "minimize_args": {"options": dict(maxiter=100)}}})

m2.optimize(single_obj_data)

# %%
from trieste.utils.parametric_gp_posterior import gen_approx_posterior_through_rff_wsa
from trieste.data import Dataset

# sampler = RandomFourierFeatureThompsonSampler(10000, m2, Dataset(X, Y))

# %% [markdown]
# Below is a clear demonstration of variance starvation

# %%
fs = gen_approx_posterior_through_rff_wsa(m2._model, 50)

fig = plt.figure()
for f in fs:
    plt.plot(xx, f(xx))


## plot
# plot f
fmean, fvar = m2.predict(xx)
# plt.figure(figsize=(10, 5))
plt.plot(X, Y, "kx", mew=2)
plt.plot(xx, fmean, "C0", lw=2)
plt.fill_between(
    xx[:, 0],
    fmean[:, 0] - 1.96 * np.sqrt(fvar[:, 0]),
    fmean[:, 0] + 1.96 * np.sqrt(fvar[:, 0]),
    color="C0",
    alpha=0.2,
    label="$f$ GP Posterior",
)


plt.xlim(0, 1)


plt.legend(loc='upper right', fontsize=15)

plt.xlim([0, 1])

plt.tight_layout()
plt.show()
