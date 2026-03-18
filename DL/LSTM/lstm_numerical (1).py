import math

sigmoid = lambda x: 1 / (1 + math.exp(-x))
tanh    = lambda x: math.tanh(x)

Wf=Uf=Wi=Ui=Wc=Uc=Wo=Uo = 0.5
bf=bi=bc=bo = 0.0
W_y, b_y = 5.0, 0.0

h, c = 0.0, 0.0

for t, x in enumerate([1.0, 2.0, 3.0], 1):
    f = sigmoid(Wf*x + Uf*h + bf)
    i = sigmoid(Wi*x + Ui*h + bi)
    g = tanh(Wc*x + Uc*h + bc)
    c = f*c + i*g
    o = sigmoid(Wo*x + Uo*h + bo)
    h = o * tanh(c)
    print(f"t={t} | h={h:.4f}  c={c:.4f}")

print(f"\nŷ = {W_y * h + b_y:.4f}")
