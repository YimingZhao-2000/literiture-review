## 3-D Self-Attention vs. Light 3-D Conv Decoder  
（复杂度 & 几何性质一览）

| **性质** | **3-D Self-Attention 解码器** | **轻量 3-D Conv 解码器** |
| --- | --- | --- |
| 核心公式 | $\displaystyle \operatorname{softmax}\!\bigl(QK^{\top}/\sqrt d\bigr)V$ | $g'=\mathcal U_{2}\!\bigl(\sigma(r_1+r_2)\bigr)$ |
| FLOPs <br>（设 $T=\frac{HW}{p^{2}}D$） | $T^{2}d=\mathcal{O}\!\Bigl(\tfrac{H^{2}W^{2}D^{2}}{p^{4}}\Bigr)$ <br>*二次* | $\mathcal{O}(HWD)$ <br>*一次* |
| 输出集合 | $\Bigl\{\sum_j a_j v_j \,\big|\, a_j\!\ge0,\sum a_j=1\Bigr\}\;=\;\operatorname{conv}\{v_j\}$ | $\bigl\{\,W* g \bigr\}=x_0+\operatorname{im}(W)$ |
| Lipschitz 上界 | 依 soft-max 温度与 $T$，难显式给出 | $\lVert W\rVert_1$（卷积核 L1） |
| 显存 | $\mathcal{O}(T^{2})$ | $\mathcal{O}(T)$ |

---

### 1 必备定义

* **凸包 (convex hull)**  

  $\displaystyle
  \operatorname{conv}(S)=
  \Bigl\{\;\sum_{i=1}^{n}\lambda_i x_i
  \ \bigm|\ 
  x_i\!\in S,\ \lambda_i\!\ge0,\ \sum_{i=1}^{n}\lambda_i=1\Bigr\}$

* **仿射子空间 (affine subspace)**  

  $x_0+V=\{\,x_0+v \mid v\in V\,\}$，其中 $V$ 是线性子空间。

---

### 2 输出集合证明（概要）

#### 2.1 Self-Attention ⇒ 凸包  

$h=Va,\;a=\operatorname{softmax}(z)\Rightarrow a_j\!\ge0,\ \sum_ja_j=1$  
$\Longrightarrow\ h$ 为 $\{v_j\}$ 的凸组合  
$\Longrightarrow\ h\in\operatorname{conv}\{v_j\}$.

#### 2.2 卷积 ⇒ 仿射子空间  

$L(g)=W* g$ 为线性映射，像集 $\operatorname{im}(L)$ 是线性子空间；  
若 seg-head 含偏置 $b$，输出为 $b+\operatorname{im}(L)$，即仿射子空间。

---

### 3 残差加法 ($r_1+r_2$) 的梯度优势

设 $y=g+\phi(g)$（$\phi$ 代表两层 Conv3D-IN-σ-Conv3D-IN）。

*Jacobian*： $\nabla_g y = \mathbf I+\nabla_g\phi$  
⇒ 始终含单位矩阵 ⇒ **梯度下界 1**，避免消失；  
串行两层无该恒等项。

---

### 4 损失函数要点

* **单级损失**  

  $\displaystyle
  L^{(\ell)} = L_{\mathrm{CE}}^{(\ell)} + L_{\mathrm{Dice}}^{(\ell)}$

  - $L_{\mathrm{CE}}$ 最小化 $\mathrm{KL}(Y\Vert\hat Y)$。  
  - $L_{\mathrm{Dice}} = 1-\dfrac{2\langle Y,\hat Y\rangle}{\|Y\|_2^2+\|\hat Y\|_2^2}$  
    ——在 $\|Y\|_2=\|\hat Y\|_2$ 时即 **余弦相似度**，最大化相关能量。  
  - **线性可加**：$\nabla(\lambda_1L_1+\lambda_2L_2)=\lambda_1\nabla L_1+\lambda_2\nabla L_2$.

* **深监督总损失**  

  $\displaystyle
  L_{\text{tot}}=\sum_{\ell=1}^{4}\alpha_\ell L^{(\ell)},
  \quad(\alpha_4,\alpha_3,\alpha_2,\alpha_1)=\tfrac1{15}(8,4,2,1)$

  反向传播时各级误差 **并行累加**，粗分辨率给全局方向，细分辨率修正边界。

---
