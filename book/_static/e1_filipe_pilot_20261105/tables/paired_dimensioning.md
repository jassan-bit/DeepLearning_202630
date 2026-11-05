| D | Par A−B | DE de diferencias | δ/DE | R normal | R t puntual | R t con DE superior | MDE con R=5 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 7 | MLP−LSTM | 0.0004838516 | 0.49488 | 57 | 61 | 320 | 0.00160093 |
| 7 | MLP−CNN1D | 0.0003640017 | 0.65782 | 32 | 36 | 183 | 0.00120438 |
| 7 | LSTM−CNN1D | 0.0003232003 | 0.74087 | 26 | 30 | 146 | 0.00106938 |
| 14 | MLP−LSTM | 0.0005483147 | 0.4367 | 73 | 77 | 410 | 0.001814221 |
| 14 | MLP−CNN1D | 0.000434509 | 0.55108 | 46 | 50 | 259 | 0.001437669 |
| 14 | LSTM−CNN1D | 0.0003379869 | 0.70845 | 28 | 32 | 159 | 0.001118305 |
| 21 | MLP−LSTM | 0.001114504 | 0.21485 | 298 | 302 | 1680 | 0.003687582 |
| 21 | MLP−CNN1D | 0.001028629 | 0.23278 | 254 | 258 | 1432 | 0.003403446 |
| 21 | LSTM−CNN1D | 0.0002152023 | 1.1127 | 12 | 16 | 67 | 0.0007120443 |
| 28 | MLP−LSTM | 0.0006434239 | 0.37215 | 100 | 104 | 563 | 0.00212891 |
| 28 | MLP−CNN1D | 0.0007737207 | 0.30948 | 144 | 148 | 812 | 0.002560026 |
| 28 | LSTM−CNN1D | 0.0004454778 | 0.53751 | 48 | 52 | 272 | 0.001473962 |

Máximo común con DE puntual: **R=302**. Recomendación condicional usando límites superiores marginales de DE: **R=1680 semillas nuevas por modelo e historia**; las cinco del piloto no se incluyen. Se planifica una diferencia absoluta δ=0.000239448478596 frente a cero, con α=0,05/12 y potencia marginal objetivo 0,80. No es potencia alcanzada ni conjunta, ni una prueba de superar la frontera avanzada del 5 %. Los límites de DE no son simultáneos y dependen de normalidad.

Shapiro–Wilk p<0,05 en 0 de los 12 pares. Cinco diferencias no acreditan normalidad; la incertidumbre piloto y las alertas deben conservarse en la planificación. La tabla CSV completa conserva MDE con DE puntual/superior al R común y los supuestos.
