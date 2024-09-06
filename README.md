# 【学习笔记】【深度生成模型】能量函数模型EBM、Contrastive Divergence、朗之万动力学

## 能量函数模型EBM

最灵活的概率建模方法。之所以说它灵活，是因为它没有引入假设，直接以$$x$$为输入去求分布$$p_{\theta}(x)$$，对负责建模的神经网络模型没有约束条件。从这个角度来说，EBM的motivation和GAN是有异曲同工之妙的，都是为了卸下模型表达能力上的枷锁（假设和要求），但它和GAN又不一样，GAN直接生成sample，EBM则仍然是输出分布。

VAE、自回归、流模型这些生成模型都是基于一定的规则对数据建模概率分布，EBM则更为灵活，EBM的建模方法是定义一个能量函数$$f_{\theta}(x)$$，然后通过变换的方式将能量函数转换为合法的概率密度。合法的概率密度需要满足两条性质：非负且积分为1。这两个性质EBM通过两个方法对$$f_{\theta}(x)$$进行转换：

1. 通过激活函数$$g_{\theta}(x)$$将$$f_{\theta}(x)$$转换为非负，例如$$g_{\theta}(x)=f_{\theta}(x)^2$$；
2. 定义$$p_{\theta}(x) = \frac{g_{\theta}(x)}{\int g_{\theta}(x)dx}$$，则$$p_{\theta}(x)$$满足积分为1的性质，其中$$Z_{}(\theta) = \int g_{\theta}(x)dx$$是一个常数项，用于归一化；

一般情况下，对$$g_{\theta}(x)$$的积分是很难算的高维积分，这是EBM的一个问题。

EBM采用指数函数作为$$g_{\theta}(x)=\exp(f_{\theta}(x))$$，我们把EBM建模的概率密度函数写出来：

$$\begin{align}
p_{\theta}(x) &= \frac{\exp(f_{\theta}(x))}{Z_{}(\theta)} \tag{1} \\
&= \frac{\exp(f_{\theta}(x))}{\int \exp(f_{\theta}(x))dx} \tag{2} \\
\end{align}$$

指数函数有一些好处，包括计算对数似然时取log后形式会更加简单，以及高维空间的概率密度往往数值很小，取指数后可以避免溢出等等。

EBM的问题：

1. 采样困难。不像此前梳理的其它生成模型在概率建模时就已经定义了采样的过程，EBM的建模中没有引入先验分布$$p(z)$$，要采样$$x$$需要直接从神经网络的输出$$p_{\theta}(x)$$中采，但$$p_{\theta}(x)$$虽然是一个合法的概率密度函数，我们却不知道怎么从中采样$$x$$；
2. 似然难以计算。EBM的似然实际上就是模型的输出$$p_{\theta}(x)$$，它的归一化常数项$$Z_{\theta}(x) = \int \exp(f_{\theta}(x))dx$$在$$x$$维度很高的情况下是intractable的；

## 应用

虽然EBM计算似然是困难的，但是计算同一个EBM的两个输出似然之比却十分方便，因为对于同一个EBM来说，每一个输出$$p_{\theta}(x)$$对应的归一化常数项都是一致的，因此有：

$$\begin{align}
\frac{p_{\theta}(x1)}{p_{\theta}(x2)} = \frac{\exp(f_{\theta}(x1))}{\exp(f_{\theta}(x2))} = f_{\theta}(x1) - f_{\theta}(x2) \tag{3} \\
\end{align}$$

可以看到两个似然比就等于能量函数之差，同样的性质使得EBM非常适合用于PoE（product of experts)。PoE可以理解为将多个模型的输出进行融合最终输出的一种专家系统，类似于MoE，但区别在于PoE所做的是将每个模型的输出概率相乘，最终再归一化为合法的概率密度函数，类似于一个AND操作：

![image-20240903105421851](C:\Users\l00850616\AppData\Roaming\Typora\typora-user-images\image-20240903105421851.png)

当使用多个EBMs进行PoE时，实际上就变成了对多个EBMs的能量函数的求和操作后，再统一进行归一化。举个例子，我们希望生成多样化的人脸图像，人脸有这些标签：年轻/年长、男性/女性、笑/不笑、卷发/非卷发，假设我们希望最终的生成模型能够生成这四种标签描述的人脸图像，那么我们通常需要训练这四种标签的联合分布。但另一种更简单的做法是我们对每种标签训练一个EBM，然后应用PoE进行融合：我们可以直接将每个EBM的神经网络的输出（也就是能量函数）求和，然后再进行归一化，这样就能在不需要训联合分布的生成模型的条件下，直接融合多个已经训练好的EBM的知识得到一个融合模型，然后直接在上面采样$$x$$。

![image-20240903110911538](C:\Users\l00850616\AppData\Roaming\Typora\typora-user-images\image-20240903110911538.png)

事实上很多工作都是以EBM为基础的，往前可以追溯到2006年深度学习领域里程碑式的工作深度信念网络，现在Yann LeCun提出的world model也是一种EBM，只不过是一个非概率形式的（没做归一化）。

## 参数学习：Contrastive Divergence

EBM的优化目标同样是最大化对数似然：

$$\begin{align}
\max_{\theta} \mathcal{L}(x;\theta) &= \log \frac{\exp(f_{\theta}(x))}{Z_{\theta}(x)} \tag{4} \\
&= f_{\theta}(x) - \log Z_{}(\theta) \tag{5} \\
\end{align}$$

前面提到，$$Z_{\theta}(x)$$是一个高维积分，不好算，因此式（4）是很难直接计算的，也就不好直接优化。和VAE通过将似然转换成下界ELBO进行优化的思路不同，EBM不对目标函数做任何近似。既然我们最终更新参数$$\theta$$是通过梯度下降，那么有没有可能把式（5）对参数$$\theta$$的梯度写出来呢？我们试着推导一下：

$$\begin{align}
\nabla \mathcal{L}(x;\theta) &= \nabla f_{\theta}(x) - \nabla\log Z_{}(\theta) \tag{6} \\
&= \nabla f_{\theta}(x) - \frac{\nabla Z_{}(\theta)}{Z_{}(\theta)}\tag{7} \\
&= \nabla  f_{\theta}(x) - \frac{\nabla \int \exp(f_{\theta}(x)) dx}{Z_{}(\theta)}\tag{8} \\
&= \nabla  f_{\theta}(x) - \frac{\int \nabla  \exp(f_{\theta}(x)) dx}{Z_{}(\theta)}\tag{9} \\
&= \nabla  f_{\theta}(x) - \frac{\int \exp(f_{\theta}(x)) \nabla  f_{\theta}(x) dx}{Z_{}(\theta)}\tag{10} \\
&= \nabla  f_{\theta}(x) - \int \frac{\exp(f_{\theta}(x))}{Z_{}(\theta)} \nabla  f_{\theta}(x) dx\tag{11} \\
&= \nabla  f_{\theta}(x) - \int p_{\theta}(x) \nabla  f_{\theta}(x) dx\tag{12} \\
&= \nabla  f_{\theta}(x) - \mathbb{E}_{x \sim p_{\theta}(x)} \nabla  f_{\theta}(x) \tag{13} \\
&\approx  \nabla  f_{\theta}(x) - \nabla  f_{\theta}(x_{sample}) \tag{14} \\
\end{align}$$

其中式（13）到式（14）的变换采用了蒙特卡洛估计，用一次采样的结果代替了期望。式（14）告诉我们一个结论：EBM的对数似然的梯度等于从真实数据集中采样的$$x$$对应能量函数的梯度减去从模型分布中采样的$$x$$对应能量函数的梯度。

我们再从直觉上理解一下这个结论，实际上EBM的概率模型式（2）和常见的softmax是比较像的，类似于连续变量版本的softmax，可以理解为样本$$x$$出现的机会占高维空间中全部样本出现的机会的比例，那么直觉上来说我们当然希望训练数据里的正样本$$x$$出现的机会要大于训练数据外的负样本$$x$$。在模型训练的早期阶段，模型并不能生成很像训练数据的$$x$$，因此对模型来说从早期模型的分布中采样的$$x$$就是相对于训练数据的负样本，我们希望这些负样本出现的可能性尽可能小，因此我们用正样本的梯度减去负样本的梯度，算出来梯度下降的正确方向；在模型收敛阶段，模型已经可以生成很像训练数据集的$$x$$了，此时来自训练数据和来自模型采样的$$x$$的梯度会非常接近，模型也就会慢慢停止更新。

式（14）相当于是借助CD的手段，将优化问题转化为了采样问题。我们现在知道只要我们能从模型分布$$p_{\theta}(x)$$中采样$$x$$，我们就能优化EBM。从$$p_{\theta}(x)$$的表达式可以看出来，这是一个相当复杂的分布，对于高维的$$x$$我们很难直接写出表达式来，所以直接从$$p_{\theta}(x)$$中采样是不现实的。那么应该怎么采？

## 采样

我们先来思考一个问题：什么是采样？

假设我们有一个一维的单峰的高斯分布$$\mathcal{N}(0, 1)$$，我们从中采样$$x$$，我们期望采到的样本是来自概率密度较高的区间$$[-1, 1]$$的样本，因为这个区间外的$$x$$对应的概率密度太低了，并不容易被采到。因此我们可以说一个合适的采样样本$$x$$对应的概率密度要大，例如极大值点。所以采样实际上可以理解为一个优化样本的概率密度的过程：

$$\begin{align}
x^* = \mathop{\arg \max}\limits_{x}\log p_{\theta}(x)  \tag{15} \\
\end{align}$$

是不是有点像MLE。式（15）和MLE的区别在于MLE是固定$$x$$更新$$\theta$$，而采样是在分布确定的情况下（也就是$$\theta$$固定的情况下）更新$$x$$。

回到EBM的采样。既然$$p_{\theta}(x)$$太复杂了，我们没法直接采样，我们能不能先随机初始化一个$$x$$，然后把它往$$p_{\theta}(x)$$的某个极大值点附近去更新？实际上就是梯度上升：

$$\begin{align}
x_{t+1} = x_t + \epsilon \nabla_x \log p_{\theta}(x_t)  \tag{16} \\
\end{align}$$

在$$t$$足够大时，$$x_t$$能够更新至一个使得$$p_{\theta}(x_t)$$也很大的位置。值得注意的是虽热我们希望$$p_{\theta}(x_t)$$足够大，采样却并不是一个优化问题，也就是我们并不需要$$p_{\theta}(x_t)$$最大。以高斯分布为例，我们采样时假如直接取$$\mu$$作为采样结果，那么是概率密度最大的样本，但我们通常不那么做，我们往往还要给它加个方差来避免每次都采到同一个样本。所以我们会给式（16）加上一个噪声，使之变成随机梯度上升：

$$\begin{align}
x_{t+1} = x_t + \epsilon \nabla_x \log p_{\theta}(x_t) + \sqrt{2 \epsilon} z_t, z_t \sim \mathcal{N}(0, 1) \tag{17} \\
\end{align}$$

式（17）就是郎之万动力学（Langevin dynamics）的过程，实际上就是随机梯度上升，是一种采样方法。下面这张图展示了一个例子：将均匀分布初始化的一维随机变量通过郎之万动力学更新至双峰GMM的样本的过程。

![image-20240904111206799](C:\Users\l00850616\AppData\Roaming\Typora\typora-user-images\image-20240904111206799.png)

朗之万动力学中用到了Stein's Score Function也就是diffusion中大家常说的score function：

$$\begin{align}
s_{\theta}(x) := \nabla_x \log p_{\theta}(x) \tag{18} \\
\end{align}$$

score function是对数似然对样本$$x$$的梯度。将EBM的概率模型式（1）带入式（18），得到EBM的score function：

$$\begin{align}
s_{\theta}(x) := \nabla_x f_{\theta}(x)  - \nabla_x \log Z(\theta) = \nabla_x f_{\theta}(x) \tag{19} \\
\end{align}$$

由于$$Z(\theta)$$关于$$x$$是一个常量，与$$x$$无关，因此这一项对$$x$$求梯度为0，score function就是能量函数关于$$x$$的梯度。score function表示的是当前采样样本$$x$$在梯度场中最快增大概率密度$$p_{\theta}(x)$$的更新方向。有了score function，我们可以随机初始化一个$$x_0$$，然后通过梯度上升将$$x_0$$迭代更新1000次，然后认为$$x_{1000} \sim p_{\theta}(x)$$。

应用朗之万动力学，我们在EBM中可以这样采样：

1. 从一个简单的先验分布（高斯）采样一个噪声$$x_0$$，$$x_0$$和样本$$x$$同维；
2. 计算EBM模型的$$s_{\theta}(x_0)= \nabla_x \log p_{\theta}(x_0)$$；
3. 根据$$s_{\theta}(x_0)$$更新$$x_0$$为$$x_1：x_1 = x_0 + \epsilon s_{\theta}(x_0) + \sqrt{2 \epsilon} z_0, z_0 \sim \mathcal{N}(0, 1)$$；
4. 重复2、3步骤$$t$$次，最终输出$$x_t$$认为是服从目标分布$$p_{\theta}(x)$$的样本；



## 总结

diffusion可以用EBM进行解释，并且在此基础上做POE（参考ECCV2022 Compositional Visual Generation with Composable Diffusion Models https://arxiv.org/pdf/2206.01714）
