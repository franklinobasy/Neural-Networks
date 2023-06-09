# Perceptron Algorithm

## Pseudocode

For a point with coordinates $(p,q)$ $(p,q)$, label y, and prediction given by the equation:

$\hat{y} = step(w_1x_1 + w_2x_2 + b)$

- If the point is correctly classified, do nothing.
  
- If the point is classified positive, but it has a negative label, subtract $αp, αq$, and $α$ from $w_1, w_2$​, and $b$ respectively.

- If the point is classified negative, but it has a positive label, add $αp, αq$, and $α$ to $w_1, w_2$​, and $b$ respectively.
