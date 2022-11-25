IEEE754 _(exp size) _(fraction size) = exp - [2, 8] - fraction - [2, 23] 
(ordine de modificare exp > frac)
precision = exp + frac + 1

exp 8 - 2 (cand acc scade sub threshold) - stop
frac 23 - 1 (cand acc scade sub threshold) - stop

Morris _(g) _(size) = g - [2, 4] - size - [4, 32]
MorrisHEB _ _ = g - [2, 4] - size - [4, 32]
MorrisBiasHEB _ _ = g - [2, 4] - size - [4, 32]
(ord de modificare g > size) g - pe cati biti este reprezentat valoarea exponentului 

Posit _(exp size) _(size) = exp - [0, 4] - size - [4, 32]

MorrisBiasHEB _(size) = size - [4, 32]
 
step_size = 1

Experiment #1: 
Single Precision Network:

IEEE:
exp 8 - 2 (cand acc scade sub threshold) - stop
frac 23 - 1 (cand acc scade sub threshold) - stop

+++++++++++++++++++++++++++++
1. folder npy per layer f32
2.0. ploturi ox precizie oy acc => ratio + thtrehold acc dgr
2. layer precision optimization de la inceput la final (dupa alte figuri)




