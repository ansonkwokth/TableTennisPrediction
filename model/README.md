# Model

Using Elo-like system for estimating the score of each player. Each player $i$ ability is initialized by two parameters: score ($s_i$) and a scale ($\sigma_i$).



A prediciton of each play (each point) of player $i$ winsa point from player $j$:
$$p_{ij} ≡ p(i > j) \equiv \frac{e^{z_i}}{e^{z_i}+e^{z_j}}=\frac{1}{1+e^{-z_{ij}}}\, .$$
Let $z_{ij}\equiv z_i - z_j = \frac{s_i - s_j}{\sigma_{\text{tot}}}$, with $\sigma_{\text{tot}}^2\equiv \sigma_i^2 +  \sigma_j^2$. **This scaling method is very vague, with no math support, may need to come up with a better math version.**

Updating the parameters with the past data. For each set, we focus only on predicting if player $i$ wins, because player $j$ wins if player $i$ losing. We will focus on each round. The set points follow negative binomial distribution. For each set, who wins the set if one gets 11 points first. The estimate of the winning probability of player $i$ over player $j$ is $t_{ij} \equiv \frac{n_i}{n_i + n_j}$ according to the negative binomial distribution, where $n_{i, j}$ is the points wons by player $i, j$.

The parameters can be estimated by MLE, defining the likelihood be:
$$L\equiv \prod_I (p_{I, ij})^{t_{I, ij}} \times (1-p_{I, ij})^{(1-t_{I, ij})}\, .$$
Same as minimizing the following loss function:
$$\mathcal{L}≡-\log{L}=-\sum_I (t_{I, ij} \log{p_{I, ij}} + (1-t_{I, ij})\log{(1-p_{I, ij})})\, .$$
To update the parameters, one finds:
$$\frac{\partial \mathcal{L}}{\partial p_{ij}} = \frac{ p_{ij} - t_{ij}}{p_{ij}(1-p_{ij})}\, $$
and
$$\frac{\partial p_{ij}}{\partial z_{ij}} = p_{ij}(1-p_{ij})\, ,$$
therefore
$$\frac{\partial \mathcal{L}}{\partial z_{ij}} = p_{ij} - t_{ij}\, .$$
Given the deinfiation above for $z_{ij}$, one find that:
$$\frac{\partial z_{ij}}{\partial s_i} = \frac{1}{\sigma_{\text{tot}}}\, , \quad \frac{\partial z_{ij}}{\partial s_j} = -\frac{1}{\sigma_{\text{tot}}}\, , \quad\frac{\partial z_{ij}}{\partial \sigma_{i, j}} = -(s_i-s_j)\frac{\sigma_{i,j}}{\sigma^3_{\text{tot}}}\, .$$

With all these, one find update the parameters using GD, with a learning ratio $\nu$:
$$s_i ← s_i - \nu \frac{\partial \mathcal{L}}{\partial s_i} = s_i-\nu \frac{p_{ij}-t_{ij}}{\sigma_{\text{tot}}}$$
$$s_j ← s_j - \nu \frac{\partial \mathcal{L}}{\partial s_j} = s_j-\nu \frac{p_{ji}-t_{ji}}{\sigma_{\text{tot}}} = s_j-\nu \frac{(1-p_{ij})-(1-t_{ij})}{\sigma_{\text{tot}}} = s_j+\nu \frac{p_{ij}-t_{ij}}{\sigma_{\text{tot}}}$$
$$\sigma{i,j} ←\sigma_{i,j} - \nu \frac{\partial \mathcal{L}}{\partial \sigma_{i,j}} = \sigma_{i,j}+\nu(p_{ij}-t_{ij})(s_i-s_j) \frac{\sigma_{i,j}}{\sigma^3_{\text{tot}}}$$


With the estimated paramters, one can calculate the probablity of winning a point ($p_{ij}$) using the equation above. One can then calculate player $i$ winning a set using negative binomial distribution. Note that there are two parts: not deuce, and deuce. First, calculate the case without deuce, which includes the cases when total scores are from 11 ($11:0$) to 20 ($11:9$). To calculate the probability of getting a given total score, we can first get the probablity of winning $10:n$ (with $n<10$), and then player $i$ wins the 11th point:

$$p(11:n) = p(10+n)p_{ij} = (C^{10+n}_{10}p_{ij}^{10}(1-p_{ij})^n) p_{ij} = C^{10+n}_{10}p_{ij}^{11}(1-p_{ij})^n\, .$$

When the case it's deuce. Then it is a infinite series sum. Note that it is not possible to have total point being 21, 23, 25, ... because one needs to win two points in row. Therefore, one can write down the geometric series with factor $2p_{ij}(1-p_{ij})$.

$$p(12:10)= p(10+10)p_{ij}^2\, ,$$


$$p(13:11) = p(10+10)p_{ij}(1-p_{ij})p_{ij}^2 + p(10+10)(1-p_{ij})p_{ij}p_{ij}^2 = p(10+10)p_{ij}^2\times (2p_{ij}(1-p_{ij}))\, ,$$

$$p(14:12) =p(10+10)p_{ij}(1-p_{ij})p_{ij}(1-p_{ij})p_{ij}^2 + p(10+10)p_{ij}(1-p_{ij})(1-p_{ij})p_{ij}p_{ij}^2 + p(10+10)(1-p_{ij})p_{ij}(1-p_{ij})p_{ij}p_{ij}^2 + p(10+10)(1-p_{ij})p_{ij}p_{ij}(1-p_{ij})p_{ij}^2 = p(10+10)p_{ij}^2\times (2p_{ij}(1-p_{ij}))^2\, ,$$



So,

$$p(12:10) + p(13:11) + p(14:12) + ... = p(10+10)p_{ij}^2 \times (1 + (2p_{ij}(1-p_{ij})) + (2p_{ij}(1-p_{ij}))^2 + ...) = \frac{C_{10}^{20} p_{ij}^{12}(1-p_{ij})^{10}}{1-2p_{ij}(1-p_{ij})}$$



Therefore the total probability of player $i$ winning a set is:

$$p_{ij}^S = \sum_{n=0}^{9}C_{10}^{10+n} p_{ij}^{11}(1-p_{ij})^n + \frac{C_{10}^{20}p_{ij}^{12}(1-p_{ij})^{10}}{1-2p_{ij}(1-p_{ij})}$$

Similarly, one can calculate the player $i$ winning the game by winning $n_S$ sets. For example if $n_S=3$, then

$$ p_{ij}^G = \sum_{n_S=0}^{2} C_{2}^{3+n_S}(p^S_{ij})^{3}(1-p_{ij}^S)^{n_S}$$







