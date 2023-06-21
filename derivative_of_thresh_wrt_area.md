I consider the case of a single location first. I assume that the fixed pre-mask is $a$, the desired area is $x$ and the threshold within the sigmoid is $y$. 

we desire $y^\star$ s.t.:

$y^\star = \argmin_{y} (\sigma(a - y) - x)^2$

Note that we use the $L2$ distance instead of $L1$ due to its undefined gradient when $\sigma(a - y) = x$ which is the region we'll be operating in.

Lets refer to function under question as $g(x,y)$. By its definition, $y^\star$ satisfies:

$\frac{\partial (\sigma(a - y) - x)^2}{\partial y} |_{y^\star} = 0 $

i.e. at the desired $y^\star$ we achieve the minimum of the discrepancy. 
Expanding this out:

$\frac{\partial g(x,y)}{\partial y}  = 2 \times (\sigma (a - y) - x) \times \sigma (a - y) \times  (1 - \sigma (a - y)) \times -1$

substituting  $y^\star$:
$2 \times (\sigma (a - y^\star) - x) \times \sigma (a - y^\star) \times  (1 - \sigma (a - y^\star)) \times -1 = 0$

This relation also holds regardless of the $x$ we are aspiring on matching, as $y^\star$ "adapts" to make this condition true. As this equation is unchanging with x, we have:

$\frac{\partial 2 \times (\sigma (a - y^\star) - x) \times \sigma (a - y^\star) \times  (1 - \sigma (a - y^\star)) \times -1} {\partial x} = 0$

Reexpressing this while ignoring multiplicative constants:

$\frac{\partial \times T_1 \times T_2} {\partial x} = 0$ where $T_1 = (\sigma (a - y^\star) - x)$ and $T_2 = \sigma (a - y^\star) \times  (1 - \sigma (a - y^\star))$

$\frac{\partial T_1}{\partial x} = \sigma (a - y^\star) \times (1  - \sigma (a - y^\star)) \times \frac{\partial y^\star}{\partial x} - 1$


$\frac{\partial T_2}{\partial x} = \sigma (a - y^\star) \times  (1 - \sigma (a - y^\star)) \times (1 - 2\sigma (a - y^\star)) \times \frac{\partial y^\star}{\partial x}$

using these expressions:
$\frac{\partial \times T_1 \times T_2} {\partial x} = T_1 \times \frac{\partial T_2} {\partial x} + T_2 \times \frac{\partial  T_1 } {\partial x}$


$ (\sigma (a - y^\star) - x) \times (\sigma (a - y^\star) \times  (1 - \sigma (a - y^\star)) \times (1 - 2\sigma (a - y^\star)) \times \frac{\partial y^\star}{\partial x})  +  \sigma (a - y^\star) \times  (1 - \sigma (a - y^\star)) \times \sigma (a - y^\star) \times  (1 - \sigma (a - y^\star)) \times (1 - 2\sigma (a - y^\star)) \times \frac{\partial y^\star}{\partial x} = 0$

removing common terms:

$ (\sigma (a - y^\star) - x) \times ( \frac{\partial y^\star}{\partial x})  +  \sigma (a - y^\star) \times  (1 - \sigma (a - y^\star))  \times \frac{\partial y^\star}{\partial x} = 0$