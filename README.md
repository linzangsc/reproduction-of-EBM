# 【学习笔记】深度生成模型（七）：能量函数模型EBM、Contrastive Divergence、朗之万动力学

## 写在前面

本文是对人大高瓴人工智能学院 李崇轩教授主讲的公开课第六节能量函数模型部分内容的梳理（课程链接[了解Sora背后的原理，你需要学习深度生成模型这门课！ 人大高瓴人工智能学院李崇轩讲授深度生成模型之原理与应用（第1讲）_哔哩哔哩_bilibili](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1yq421A7ig/%3Fspm_id_from%3D333.337.search-card.all.click%26vd_source%3D196c5d43f645df8f93e712dc5e152b18)）。如有不准确的地方还请大家指出。

作为一种基于极大似然估计(MLE)的生成模型，能量函数模型(Energy based model, EBM)对似然的建模可以说是最灵活的。VAE、自回归、流模型都是基于一定的规则和假设对数据建模概率分布，EBM则直接以$$x$$为输入去求分布$$p_{\theta}(x)$$，同时对负责建模的神经网络模型没有约束条件。从这个角度来说，EBM的motivation和GAN是有异曲同工之妙的，都是为了卸下模型表达能力上的枷锁（假设和要求），但它和GAN又不一样，GAN直接生成sample，EBM则仍然是输出分布。

## 能量函数模型EBM

和VAE、GAN不同的是，EBM通常只有一个神经网络模型。一方面，我们有一个表达能力很强的神经网络模型，它接收样本$$x$$作为输入，可以用来拟合非常复杂的非线性函数$$f_{\theta}(x)$$；另一方面，为了能够通过MLE驱动模型学习和更新，我们需要拟合似然$$p_{\theta}(x)$$。EBM计算似然的方法很直接，且没有对概率建模做任何假设，也没有对模型做任何约束：将$$f_{\theta}(x)$$转换成合法的概率密度。合法的概率密度需要满足两条性质：非负且积分为1。因此EBM对神经网络的输出$$f_{\theta}(x)$$进行如下转换：

1. 通过激活函数$$g_{\theta}(x)$$将$$f_{\theta}(x)$$转换为非负，例如$$g_{\theta}(x)=f_{\theta}(x)^2$$；
2. 定义$$p_{\theta}(x) = \frac{g_{\theta}(x)}{\int g_{\theta}(x)dx}$$，则$$p_{\theta}(x)$$满足积分为1的性质，其中$$Z_{}(\theta) = \int g_{\theta}(x)dx$$是一个常数项，用于归一化；

通过上述两步，就能够将神经网络的输出$$f_{\theta}(x)$$转换为合法的概率密度函数$$p_{\theta}(x)$$。一般地，EBM采用指数函数作为$$g_{\theta}(x)=\exp(f_{\theta}(x))$$，我们把EBM建模的概率密度函数写出来：

$$\begin{align}
p_{\theta}(x) &= \frac{\exp(f_{\theta}(x))}{Z_{}(\theta)} \tag{1} \\
&= \frac{\exp(f_{\theta}(x))}{\int \exp(f_{\theta}(x))dx} \tag{2} \\
\end{align}$$

指数函数有一些好处，包括计算对数似然时取log后形式会更加简单，以及高维空间的概率密度往往数值很小，取指数后可以避免溢出等等。在式（2）中，通常称$$E(x) = -f_{\theta}(x)$$为能量函数。

乍一看EBM建模的似然式（2），会发现这个似然函数在$$x$$高维的情况下，分母是一个高维积分，计算很困难，而这会带来两个问题：

1. 采样困难。不像此前梳理的其它生成模型在概率建模时就已经定义了采样的过程，EBM的建模中没有引入先验分布$$p(z)$$，要采样$$x$$需要直接从神经网络的输出$$p_{\theta}(x)$$中采，但我们却很难写出其表达式，没法显式地直接采样$$x$$；
2. 似然难以计算。EBM的似然实际上就是模型的输出$$p_{\theta}(x)$$，它的归一化常数项$$Z_{\theta}(x) = \int \exp(f_{\theta}(x))dx$$在$$x$$维度很高的情况下（例如图片）是intractable的；

前面提到，EBM本质上还是一个基于MLE的生成模型。对EBM建模的似然函数做MLE，可以得到：

$$\begin{align}
\max_{\theta} \mathcal{L}(x;\theta) &= \log \frac{\exp(f_{\theta}(x))}{Z_{\theta}(x)} \tag{3} \\
&= f_{\theta}(x) - \log Z_{}(\theta) \tag{4} \\
\end{align}$$

这里$$Z(\theta)$$由于是一个高维积分，很难计算，因此需要借助其他方法进行优化。

## 参数学习：Contrastive Divergence

Contrastive Divergence(CD)是Hinton提出的一种对EBM做极大似然优化的算法。和VAE通过将似然转换成下界ELBO进行优化的思路不同，CD不对目标函数做任何近似。既然我们最终更新参数$$\theta$$是通过梯度上升（下降），那么有没有可能把式（4）对参数$$\theta$$的梯度写出来呢？我们试着推导一下：

$$\begin{align}
\nabla \mathcal{L}(x;\theta) &= \nabla f_{\theta}(x) - \nabla\log Z_{}(\theta) \tag{5} \\
&= \nabla f_{\theta}(x) - \frac{\nabla Z_{}(\theta)}{Z_{}(\theta)}\tag{6} \\
&= \nabla  f_{\theta}(x) - \frac{\nabla \int \exp(f_{\theta}(x)) dx}{Z_{}(\theta)}\tag{7} \\
&= \nabla  f_{\theta}(x) - \frac{\int \nabla  \exp(f_{\theta}(x)) dx}{Z_{}(\theta)}\tag{8} \\
&= \nabla  f_{\theta}(x) - \frac{\int \exp(f_{\theta}(x)) \nabla  f_{\theta}(x) dx}{Z_{}(\theta)}\tag{9} \\
&= \nabla  f_{\theta}(x) - \int \frac{\exp(f_{\theta}(x))}{Z_{}(\theta)} \nabla  f_{\theta}(x) dx\tag{10} \\
&= \nabla  f_{\theta}(x) - \int p_{\theta}(x) \nabla  f_{\theta}(x) dx\tag{11} \\
&= \nabla  f_{\theta}(x) - \mathbb{E}_{x \sim p_{\theta}(x)} \nabla  f_{\theta}(x) \tag{12} \\
&\approx  \nabla  f_{\theta}(x) - \nabla  f_{\theta}(x_{sample}) \tag{13} \\
\end{align}$$

其中式（12）到式（13）的变换采用了蒙特卡洛估计，用一次采样的结果代替了期望。式（13）告诉我们一个结论：EBM的MLE的梯度等于从真实数据集中采样的$$x$$对应神经网络输出的梯度减去从模型分布中采样的$$x$$对应神经网络输出的梯度，因此损失函数可以设计成从模型分布中采样的样本$$x_{sample}$$经过神经网络的输出减去真实数据集中的样本$$x$$经过神经网络的输出（由于随机梯度下降是最小化目标函数的过程，和MLE相反，因此这里相比式（13）取了负号）：

$$\begin{align}
\min_{\theta} \mathcal{L}(x;\theta) &= f_{\theta}(x_{sample}) - f_{\theta}(x) \tag{14} \\
\end{align}$$

我们再从直觉上理解一下这个结论，实际上EBM的概率模型式（2）和常见的softmax是比较像的，类似于连续变量版本的softmax，可以理解为样本$$x$$出现的机会占高维空间中全部样本出现的机会的比例，那么直觉上来说我们当然希望训练数据里的正样本$$x$$出现的机会要大于训练数据外的负样本$$x$$。在模型训练的早期阶段，模型并不能生成很像训练数据的$$x$$，因此对模型来说从早期模型的分布中采样的$$x$$就是相对于训练数据的负样本，我们希望这些负样本出现的可能性尽可能小，因此我们用负样本的梯度减去正样本的梯度，得到梯度下降的正确方向；在模型收敛阶段，模型已经可以生成很像训练数据集的$$x$$了，此时来自训练数据和来自模型采样的$$x$$的梯度会非常接近，模型也就会慢慢停止更新。

从能量函数的角度来理解式（14），也可以理解为EBM的优化过程是一个不断抬升负样本$$x_{sample}$$的能量同时不断降低正样本$$x$$的能量的过程（能量越低越稳定）。

![image-20240926172413662](C:\Users\l00850616\AppData\Roaming\Typora\typora-user-images\image-20240926172413662.png)

图片来自A Tutorial on Energy-Based Learning

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

score function是对数似然对样本$$x$$的梯度。将式（1）带入式（18），得到EBM的score function：

$$\begin{align}
s_{\theta}(x) := \nabla_x f_{\theta}(x)  - \nabla_x \log Z(\theta) = \nabla_x f_{\theta}(x) \tag{19} \\
\end{align}$$

由于$$Z(\theta)$$关于$$x$$是一个常量，与$$x$$无关，因此这一项对$$x$$求梯度为0，score function就是能量函数关于$$x$$的梯度。score function表示的是当前采样样本$$x$$在梯度场中最快增大概率密度$$p_{\theta}(x)$$的更新方向。有了score function，我们可以随机初始化一个$$x_0$$，然后通过梯度上升将$$x_0$$迭代更新1000次，然后认为$$x_{1000} \sim p_{\theta}(x)$$。

应用朗之万动力学，我们在EBM中可以这样采样：

1. 从一个简单的先验分布（0-1均匀分布）采样一个噪声$$x_0$$，$$x_0$$和样本$$x$$同维；
2. 计算EBM模型的$$s_{\theta}(x_0)= \nabla_x \log p_{\theta}(x_0)$$；
3. 根据$$s_{\theta}(x_0)$$更新$$x_0$$为$$x_1：x_1 = x_0 + \epsilon s_{\theta}(x_0) + \sqrt{2 \epsilon} z_0, z_0 \sim \mathcal{N}(0, 1)$$；
4. 重复2、3步骤$$t$$次，最终输出$$x_t$$认为是服从目标分布$$p_{\theta}(x)$$的样本；

这样的采样方法存在一个问题，就是$$t$$往往需要很大才能得到较好的采样，这大大降低了训练效率。为了提高训练效率，Hinton在提出CD（2002年）之时，就提出直接使用来自真实数据集中的样本$$x$$作为迭代的初值$$x_0$$，并使用吉布斯采样迭代有限几步后的样本作为负样本进行训练。虽然该方法用在RBM上是有效的，但后续的一些工作指出通过这种采样方法实现的CD得到的梯度实际上是对MLE的梯度的有偏估计，是有缺陷的。笔者在一个简单的CNN上尝试了该方法，结果是无法生成有意义的图片。

笔者尝试的有效的方法，是参考Implicit Generation and Modeling with Energy-Based Models中的sample replay buffer。这篇工作中对负样本的采样依然是从随机采样的噪声出发，但用一个buffer来存放每轮训练的采样结果，并在下一次训练时，沿用绝大部分上一轮训练的采样结果作为郎之万动力学的初值。在这样的设置下，会有大量的负样本随着训练的进行不断迭代，不断逼近模型分布的采样结果。除此之外，笔者在复现的过程中发现，如果损失函数仅用式（14），那么容易出现$$f_{\theta}(x)$$逐渐增大导致梯度爆炸的情况，因此实际实现时还需要加上L2正则化项来保证能量函数输出的数量级不会太大。



## 应用

虽然EBM计算似然是困难的，但是计算同一个EBM的两个输出似然之比却十分方便，因为对于同一个EBM来说，每一个输出$$p_{\theta}(x)$$对应的归一化常数项都是一致的，因此有：

$$\begin{align}
\frac{p_{\theta}(x1)}{p_{\theta}(x2)} = \frac{\exp(f_{\theta}(x1))}{\exp(f_{\theta}(x2))} = f_{\theta}(x1) - f_{\theta}(x2) \tag{20} \\
\end{align}$$

可以看到两个似然比就等于能量函数之差，同样的性质使得EBM非常适合用于PoE（product of experts)。PoE可以理解为将多个模型的输出进行融合最终输出的一种专家系统，类似于MoE，但区别在于PoE所做的是将每个模型的输出概率相乘，最终再归一化为合法的概率密度函数，类似于一个AND操作：

![image-20240903105421851](C:\Users\l00850616\AppData\Roaming\Typora\typora-user-images\image-20240903105421851.png)

当使用多个EBMs进行PoE时，实际上就变成了对多个EBMs的能量函数的求和操作后，再统一进行归一化。举个例子，我们希望生成多样化的人脸图像，人脸有这些标签：年轻/年长、男性/女性、笑/不笑、卷发/非卷发，假设我们希望最终的生成模型能够生成这四种标签描述的人脸图像，那么我们通常需要训练这四种标签的联合分布。但另一种更简单的做法是我们对每种标签训练一个EBM，然后应用PoE进行融合：我们可以直接将每个EBM的神经网络的输出（也就是能量函数）求和，然后再进行归一化，这样就能在不需要训联合分布的生成模型的条件下，直接融合多个已经训练好的EBM的知识得到一个融合模型，然后直接在上面采样$$x$$。

![image-20240903110911538](C:\Users\l00850616\AppData\Roaming\Typora\typora-user-images\image-20240903110911538.png)

事实上很多工作都是以EBM为基础的，往前可以追溯到2006年深度学习领域里程碑式的工作深度信念网络，现在Yann LeCun提出的world model也是一种EBM，只不过是一个非概率形式的（没做归一化）。

4. 

## Score Matching

既然EBM的似然是intractable的，且EBM的score function式（19）是非常好算的，那么有没有办法从score function出发驱动EBM的学习？或者说，MLE本质上是在最小化KL散度，那么有没有办法从score function出发定义一个散度，然后去最小化这样的散度，使得EBM得到更新？答案是确实有这样的散度——Fisher散度：

$$\begin{align}
D_F(p_{data} \Vert p_{\theta}) = \frac{1}{2}\mathbb{E}_{x \sim p_{data}}[\Vert \nabla_x \log p_{data}(x) - \nabla_x \log p_{\theta}(x)\Vert_2] \tag{21} \\
\end{align}$$

Fisher散度就是数据分布$$p_{data}(x)$$和模型分布$$p_{\theta}(x)$$的score function的二范数对数据分布$$p_{data}$$取期望。将EBM的score function式（19）代入式（21），得到

$$\begin{align}
D_F(p_{data}\Vert p_{\theta}) = \frac{1}{2}\mathbb{E}_{x \sim p_{data}}[\Vert \nabla_x \log p_{data}(x) - \nabla_x f_{\theta}(x)\Vert_2] \tag{22} \\
\end{align}$$

当$$D_F(p_{data}\Vert p_{\theta})$$等于0时，意味着$$f_{\theta}(x)$$和$$\log p_{data}(x)$$在函数空间上梯度处处相等，则它们最多相差一个常数$$C$$：

$$\begin{align}
f_{\theta}(x) = \log p_{data}(x) +C \tag{23} \\
\end{align}$$

将式（23）代入EBM建模的概率分布式（2），可以推导出当$$D_F(p_{data}\Vert p_{\theta})$$等于0时，EBM的模型分布就等于数据分布：

$$\begin{align}
p_{\theta}(x) &= \frac{\exp(f_{\theta}(x))}{\int \exp(f_{\theta}(x))dx} \tag{24} \\
&= \frac{\exp(\log p_{data}(x) +C)}{\int \exp(\log p_{data}(x) +C) dx} \tag{25} \\
&= \frac{\exp(C)\exp(\log p_{data}(x))}{\exp(C)\int \exp(\log p_{data}(x)) dx} \tag{26} \\
&= \frac{p_{data}(x)}{\int  p_{data}(x) dx} \tag{27} \\
&= p_{data}(x) \tag{28} \\
\end{align}$$

因此，我们可以不做MLE，转而通过最小化Fisher散度来更新EBM，这样的优化方法称为score matching：

$$\begin{align}
\min_{\theta} D_F(p_{data}\Vert p_{\theta}) = \frac{1}{2}\mathbb{E}_{x \sim p_{data}}[\Vert \nabla_x \log p_{data}(x) - \nabla_x f_{\theta}(x)\Vert_2] \tag{29} \\
\end{align}$$

式（29）实际上还是没法直接用，原因在于数据分布的score function是未知的，好在Fisher散度可以写成不带$$\nabla_x \log p_{data}(x)$$的形式：

$$\begin{align}
\min_{\theta} D_F(p_{data}, p_{\theta}) &= \mathbb{E}_{x \sim p_{data}}[\frac{1}{2}\Vert \nabla_x f_{\theta}(x)\Vert_2 + tr(\nabla_x^{2}f_{\theta}(x))] \tag{30} \\
&= \frac{1}{n} \sum_{i=1}^n [\frac{1}{2}\Vert \nabla_x f_{\theta}(x_i)\Vert_2 + tr(\nabla_x^{2}f_{\theta}(x_i))] \tag{31} \\
\end{align}$$

式（30）到式（31）是使用了蒙特卡罗估计。因此式（31）是score matching的优化目标，通过最小化式（31）来驱动EBM更新，然后在训练完成以后再通过郎之万动力学采样，是EBM的一种likelihood-free的学习方法。和基于CD的学习方法相比，score matching不需要在每个训练轮次都多次采样，但不足之处是$$f_{\theta}(x_i)$$的二阶导也就是hessian矩阵的计算也是复杂度比较高的。

## 总结

diffusion可以用EBM进行解释，并且在此基础上做POE（参考ECCV2022 Compositional Visual Generation with Composable Diffusion Models https://arxiv.org/pdf/2206.01714）
