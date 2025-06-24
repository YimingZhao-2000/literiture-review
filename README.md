## 3-D Self-Attention vs. Light 3-D Conv Decoder  

| **性质** | **3-D Self-Attention 解码器** | **轻量 3-D Conv 解码器** |
| --- | --- | --- |
| 核心公式 | $\displaystyle \text{softmax}\!\bigl(QK^{\!\top}/\sqrt d\bigr)V$ | $g'\;=\;\mathcal U_2\!\bigl(\sigma(r_1+r_2)\bigr)$ |
| FLOPs<br>(设 $T=\tfrac{HW}{p^{2}}D$) | $T^{2}d\;=\;\mathcal O\!\bigl(H^{2}W^{2}D^{2}/p^{4}\bigr)$ <br>*二次* | $\mathcal O(HWD)$ <br>*一次* |
| 输出集合 | $\bigl\{\sum_j a_j v_j\,\big|\,a_j\!\ge0,\sum a_j=1\bigr\}\;=\;\text{conv}\,\{v_j\}$ | $\{\,W* g\}\;=\;x_0+\text{im}(W)$ |
| Lipschitz 上界 | 依 soft-max 温度与 $T$，难显式给出 | $\lVert W\rVert_{1}$（卷积核 L1） |
| 显存 | $\mathcal O(T^{2})$ | $\mathcal O(T)$ |

---

### 1 必备定义

* **凸包**  

  $\displaystyle
  \text{conv}(S)=
  \Bigl\{\sum_{i=1}^{n}\lambda_i x_i\;\Big|\;
        x_i\in S,\;\lambda_i\!\ge0,\;\sum_{i=1}^{n}\lambda_i=1\Bigr\}$

* **仿射子空间**  

  $x_0+V=\{\,x_0+v \mid v\in V\,\}$，其中 $V$ 是线性子空间。

---

### 2 输出集合证明（概要）

#### 2.1 Self-Attention ⇒ 凸包  

$h=Va,\quad a_j\!\ge0,\;\sum_ja_j=1  
\;\Longrightarrow\;h$ 为 $\{v_j\}$ 的凸组合  
$\;\Longrightarrow\;h\in\text{conv}\,\{v_j\}$.

#### 2.2 卷积 ⇒ 仿射子空间  

$L(g)=W* g$ 为线性映射，像集 $\text{im}(L)$ 是线性子空间；  
若 seg-head 含偏置 $b$，输出为 $b+\text{im}(L)$，即仿射子空间。

---

### 3 残差加法 ($r\_1+r\_2$) 的梯度优势

设 $y=g+\phi(g)$（$\phi$ 为两层 Conv3D-IN-σ-Conv3D-IN）。

*Jacobian*： $\nabla\_g y = I+\nabla\_g\phi$  
⇒ 始终含单位矩阵 $I$ ⇒ **梯度下界 1**，避免消失；  
纯串行两层无该恒等项。

---

### 4 损失函数要点

* **单级损失**  

  $\displaystyle
  L^{(\ell)} = L_{\mathrm{CE}}^{(\ell)} + L_{\mathrm{Dice}}^{(\ell)}$

  * $L_{\mathrm{CE}}$ 最小化 $\text{KL}(Y\;\|\;\hat Y)$.  
  * $L_{\mathrm{Dice}} = 1-\dfrac{2\langle Y,\hat Y\rangle}{\lVert Y\rVert_2^{2}+\lVert\hat Y\rVert_2^{2}}$  
    – 若 $\lVert Y\rVert_2=\lVert\hat Y\rVert_2$，即 **余弦相似度**，最大化相关能量。  
  * **线性可加**：  
    $\nabla(\lambda\_1L\_1+\lambda\_2L\_2)=\lambda\_1\nabla L\_1+\lambda\_2\nabla L\_2$.

* **深监督总损失**  

  $\displaystyle
  L_{\text{tot}}=\sum_{\ell=1}^{4}\alpha_\ell L^{(\ell)},
  \quad(\alpha\_4,\alpha\_3,\alpha\_2,\alpha\_1)=\tfrac1{15}(8,4,2,1)$

  反向传播时各级误差 **并行累加**：  
  粗分辨率给全局方向，细分辨率精修边界。

---
