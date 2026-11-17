# Cota condicional de la cabeza aditiva realmente evaluada

Fecha: 17 de noviembre de 2026. Preparación y revisión matemática asistidas por
IA; no constituye revisión humana externa ni declaración de novedad.

La receta es `additive_positive_head`, de 1210 parámetros, implementada en
`src/e3_innovation_model.py`. Conserva CNN causal, LSTM unidireccional y atención
por contenido, sin penalización de discordancia ni saturación radial. Esta
derivación no reutiliza la cota global sobre estados arbitrarios del primer
módulo de 1211 parámetros. Es una propiedad condicional conocida de atención
con estados acotados, no un mecanismo nuevo ni una certificación R1/R2/R3.

## Dominio y supuestos

Sean $H,H'\in\mathbb R^{n\times d}$, con cada fila de norma euclídea a lo sumo
$R$. Se usan norma de Frobenius para matrices de estados y norma espectral
para operadores. Los parámetros de inferencia permanecen fijos y satisfacen

$$
\|q\|_2\le Q,\quad \|w\|_2\le W,\quad
\tau\ge\tau_{\min}>0,\quad a_{\rm tr}>0,\quad A\ge0.
$$

El piso $A$ y la escala $a_{\rm tr}$ también permanecen fijos al comparar
$H$ y $H'$. La función estudiada, en aritmética real, es

$$
\alpha(H)=\operatorname{softmax}(Hq/\tau),\qquad
c(H)=H^\top\alpha(H),
$$

$$
f_A(H)=\sqrt A+a_{\rm tr}
\left[10^{-4}+\operatorname{softplus}(w^\top c(H)+b)\right].
$$

En la LSTM estándar implementada, cada coordenada de un estado oculto tiene la
forma $h_j=o_j\odot\tanh(C_j)$, con $0<o_{jk}<1$ para argumentos finitos.
Por tanto $|h_{jk}|\le1$ y se puede usar $R=\sqrt d$. Esto sólo identifica
un dominio que contiene los estados alcanzables: no acota cuánto cambia el
encoder cuando cambia su entrada. No se afirma que todo tensor de ese dominio
sea alcanzable por la LSTM.

La parametrización del código da $Q=1$, $W=4$, $\tau_{\min}=0,25$;
$n=7$, $d=12$. Los límites de norma proceden de las normalizaciones suaves
de los parámetros, no de una suposición sobre pesos libres no controlados.

## Teorema y demostración

Bajo esos supuestos,

$$
|f_A(H)-f_A(H')|\le K_{\rm ad}\|H-H'\|_F,
\qquad
K_{\rm ad}=a_{\rm tr}W
\left(1+\frac{\sqrt n\,R Q}{2\tau_{\min}}\right).
$$

Para softmax, su jacobiano es $J=\operatorname{diag}(\alpha)-\alpha\alpha^\top$.
La suma absoluta de cada fila es $2\alpha_j(1-\alpha_j)\le1/2$. Al ser
simétrico, $\|J\|_2\le1/2$. Integrar su jacobiano sobre el segmento de
puntajes y usar Cauchy–Schwarz da

$$
\|\alpha(H)-\alpha(H')\|_2
\le\frac{Q}{2\tau_{\min}}\|H-H'\|_F.
$$

La distribución de atención cumple $\|\alpha(H')\|_2\le1$. Además,
$\|H\|_2\le\|H\|_F\le\sqrt n R$. Por la descomposición

$$
c(H)-c(H')=H^\top[\alpha(H)-\alpha(H')]
+(H-H')^\top\alpha(H'),
$$

se obtiene

$$
\|c(H)-c(H')\|_2
\le\left(1+\frac{\sqrt n RQ}{2\tau_{\min}}\right)
\|H-H'\|_F.
$$

Softplus es 1-Lipschitz porque su derivada es sigmoid, entre cero y uno.
Componer la lectura lineal de norma a lo sumo $W$, multiplicar por la escala
positiva y cancelar el sesgo, el piso y la constante $10^{-4}$ demuestra el
enunciado. Para la dimensión registrada,

$$
K_{\rm ad}/a_{\rm tr}
=4\left(1+\frac{\sqrt{84}}{0,5}\right)
=77,32121111929344.
$$

La cota usa límites estructurales y puede ser muy conservadora. No se ajusta al
mayor cociente observado para hacer pasar la verificación.

## Qué ocurre cuando cambia el piso observable

Si $A\ne A'$, el mismo argumento sólo permite afirmar

$$
|f_A(H)-f_{A'}(H')|
\le |\sqrt A-\sqrt{A'}|+K_{\rm ad}\|H-H'\|_F
\le\sqrt{|A-A'|}+K_{\rm ad}\|H-H'\|_F.
$$

La raíz no tiene una constante Lipschitz uniforme respecto de $A$ cerca de
cero: con $A'=0$, el cociente $|\sqrt A-0|/|A-0|=1/\sqrt A$ diverge.
Esta observación no demuestra inestabilidad del pipeline desde retornos u
OHLCV; impide atribuirle una garantía que no se ha probado con ese estimando
y esa parametrización. Tampoco demuestra calibración o utilidad predictiva.

## Verificación numérica reproducible y límites

`verify_theory.py` carga la clase local y comprueba sus hashes frente a las
fuentes congeladas de la campaña. No carga pesos entrenados ni matrices de
mercado. Examina tres perfiles sintéticos: inicialización sembrada, parámetros
cercanos a sus límites de norma/temperatura y atención uniforme. Dentro de
cada perfil los parámetros permanecen fijos. Usa 1728 pares de estados con
coordenadas en $[-1,1]$, tres amplitudes y tres escalas de perturbación; además
comprueba 96 normas de gradiente de la misma fórmula en float64. Las constantes
$a_{\rm tr}=0,02$ y $A=0,0001$ son sintéticas, no estadísticas ajustadas al
mercado.

La vía mixta invoca los métodos originales de atención float32 y lectura/salida
float64. Su comparación con una referencia float64 registra el efecto numérico
sin cambiar el modelo científico. La tolerancia absoluta de comprobación de
pares es $10^{-7}a_{\rm tr}+10^{-12}$, declarada en el script. La afirmación
universal corresponde a la demostración en aritmética real: la cuantización de
float32 no hereda literalmente una garantía continua para cualquier diferencia
arbitrariamente pequeña. Los cocientes se evalúan sobre perturbaciones y
distancias registradas, nunca sobre entradas indistinguibles tras redondeo.

Los resultados y el ambiente están en `theory_verification.json`. Cero
violaciones finitas apoyan la implementación; no prueban universalidad. El
control de 32 ventanas artificiales verifica únicamente la forma y el rango
de los estados LSTM. No demuestra una cota para el encoder.

Reproducción desde la raíz del repositorio, con el entorno CPU existente:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
python experiments/e3_diseno_riguroso_20261117/verify_theory.py
```

La exigencia de la guía de mostrar que una propiedad importa para la predicción
**continúa abierta**: no hay aquí ablación estadística ni prueba de robustez en
datos reales que conecte esta cota con menor RMSE. La evidencia favorable de
la comparación histórica de E3 no identifica causalmente a esta propiedad como
su explicación y no completa el protocolo confirmatorio, la novedad o la
defensa académica.
