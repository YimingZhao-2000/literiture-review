### SAM-3D — 核心流程与严格代数分析

$$
I\;\xrightarrow{\text{slicing}}\;
\{I_i\}_{i=1}^{D}\;\xrightarrow{\text{SAM ViT-B}}\;
F\;\xrightarrow{\text{light 3-D decoder}}\;
\{\hat Y^{(\ell)}\}_{\ell=1}^{4}\;\xrightarrow{\large\sum\alpha_\ell L^{(\ell)}}\;
\mathcal L_{\text{total}}
$$

---

## 1 Encoder（冻结 SAM ViT-B）

1. **切片** $I\in\mathbb R^{H\times W\times D}\longrightarrow I_i\in\mathbb R^{3\times H\times W}$.
2. **Patch Embedding** 步幅$p=16$：$\displaystyle f_i=\operatorname{Enc}(I_i)\in\mathbb R^{H/16\times W/16\times256}$.
3. **堆栈** $F=\bigl[f_i\bigr]_{i=1}^{D}\in\mathbb R^{H/16\times W/16\times D\times256}$.

---

## 2 解码器：四级 3-D 残差-上采样

### 2.1 单块公式

给定 $g^{(\ell)}\in\mathbb R^{C_\ell\times H_\ell\times W_\ell\times D}$：

$$
\begin{aligned}
r_1 &= \operatorname{IN}\!\bigl(W_2*\sigma(\operatorname{IN}(W_1*g^{(\ell)}))\bigr),\\
r_2 &= \operatorname{IN}(W_0*g^{(\ell)}),\\
g^{(\ell+1)} &= \mathcal U_{2\times}\!\bigl(\sigma(r_1+r_2)\bigr),
\end{aligned}
$$

其中 $*$ 为 3^3 卷积，$\mathcal U_{2\times}$ 为三维线性插值 ×2。

### 2.2 FLOPs 推导（线性标度）

设第 $\ell$ 级空间尺寸 $H_\ell=W_\ell=H/2^{4-\ell}$，卷积核体积 $k=3^3$，则

$$
\mathrm{FLOPs}_\ell
 = H_\ell W_\ell D\;k\;C_\ell C_\ell
 = \frac{HWD\;k\,C_\ell^2}{2^{2(4-\ell)}}.
$$

令 $C_0=256,C_1=128,C_2=64,C_3=32,C_4=16$，求和得

$$
\textstyle\sum_{\ell=0}^{3}\mathrm{FLOPs}_\ell
 = \mathcal O\bigl(HWD\;k\,256^2 / 16^2\bigr)
 = \boxed{\mathcal O}(HWD).
$$

---

## 3 与 3-D Self-Attention 解码器的严格比较


$$\begin{array}{|l|c|c|}
\hline
\text{性质}\rule{0pt}{1.2em} &
\text{3-D Self-Attn 解码器} &
\text{轻量 3-D Conv 解码器} \\ \hline
\text{核心公式}\rule{0pt}{1.2em} &
\displaystyle \operatorname{softmax}\!\!\Bigl(\tfrac{QK^{\!\top}}{\sqrt d}\Bigr)V &
g'=\mathcal U_{2}\bigl(\sigma(r_1+r_2)\bigr) \\ \hline
\text{FLOPs}\ \bigl(T=\tfrac{HW}{p^{2}}D\bigr)\rule{0pt}{1.2em} &
T^{2}d=
\mathcal O\!\Bigl(\tfrac{H^{2}W^{2}D^{2}}{p^{4}}\Bigr) &
\mathcal O(HWD) \\ \hline
\text{输出集合}\rule{0pt}{1.2em} &
\bigl\{\sum_j a_j v_j\mid a_j\!\ge0,\sum a_j=1\bigr\}
=\operatorname{conv}\{v_j\} &
\{\,W*g\}
=x_0+\operatorname{im}(W) \\ \hline
\text{Lipschitz 上界}\rule{0pt}{1.2em} &
\text{依温度与}\,T\,\text{难显式给出} &
\lVert W\rVert_{1} \\ \hline
\text{显存}\rule{0pt}{1.2em} &
\mathcal O(T^{2}) &
\mathcal O(T) \\ \hline
\end{array}
$$




> **证明 FLOPs**
> 自注意力：矩阵 $Q,K\in\mathbb R^{T\times d}$ 乘积耗 $T^2d$.
> 卷积：§2.2 已线性证明。

---

## 4 两条卷积路径相加 ⊕：梯度与 Lipschitz

设输出 $y=g+\phi(g)$（$\phi$ 为卷积-IN-σ-卷积-IN）。

* **梯度** $\nabla_g y = I + \nabla_g\phi$. 恒含单位阵 ⇒ 反传梯度下界 1。
* **串行两层**：$y'=\phi(g)$，$\nabla_g y'=\nabla_g\phi$，梯度可能 <1 或 >1。
* **拼接** (concat) 会倍增通道并需额外 1×1×1 融合，参数与显存翻倍。

**结论**：⊕ 保持维度，梯度稳定，无新增参数。

---

## 5 损失

### 5.1 单级 $L^{(\ell)}$

$$
\boxed{
\begin{aligned}
\!L^{(\ell)} &=L_{\mathrm{CE}}^{(\ell)}+L_{\mathrm{Dice}}^{(\ell)} \\
L_{\mathrm{CE}}^{(\ell)} &=-\sum_{k,n}Y_{k,n}^{(\ell)}\ln\hat Y_{k,n}^{(\ell)} \\
L_{\mathrm{Dice}}^{(\ell)} &=1-\frac{2\sum_{k,n}Y_{k,n}^{(\ell)}\hat Y_{k,n}^{(\ell)}}
                               {\sum_{k,n}Y_{k,n}^{(\ell)2}+\sum_{k,n}\hat Y_{k,n}^{(\ell)2}}
\end{aligned}}
$$

* $L_{\mathrm{CE}}=H(Y)+\mathrm{KL}(Y\|\hat Y)$（常数 $H(Y)$ 忽略）
* 若 $\|Y\|_2=\|\hat Y\|_2$，Dice = cosine，相当于最大化相关能量。
* **线性可加性**：梯度线性
  $\nabla(\lambda_1L_1+\lambda_2L_2)=\lambda_1\nabla L_1+\lambda_2\nabla L_2$.

### 5.2 深监督组合

$$
L_{\mathrm{tot}}=\sum_{\ell=1}^{4}\alpha_\ell L^{(\ell)},
\qquad
(\alpha_4,\alpha_3,\alpha_2,\alpha_1)=\frac{1}{15}(8,4,2,1).
$$

**链式法则**

$$
\nabla_\theta L_{\mathrm{tot}}
 =\sum_{\ell}\alpha_\ell
   \frac{\partial L^{(\ell)}}{\partial\hat Y^{(\ell)}}
   \nabla_\theta f_\theta^{(\ell)}.
$$

每级误差 **并联** 发送 ⇒ 低层参数收到直接梯度；数值稳定因 $\sum\alpha_\ell=1$。

---

### 6 总结性矩阵

| 环节                | 线性/凸性质             | 复杂度                      | 正则/梯度要点         |
| ----------------- | ------------------ | ------------------------ | --------------- |
| Encoder           | 线性变换 + MSA (冻结)    | —                        | —               |
| Decoder ConvBlock | 仿射子空间映射            | $\mathcal O(HWD)$       | ⊕ 含 $I$ ⇒ 梯度≥1  |
| Attention Decoder | 输出凸包               | $\mathcal O(H^2W^2D^2)$ | softmax 温度需额外归一 |
| Loss (CE+Dice)    | Convex combination | —                        | 线性可加, 多尺度并联     |
