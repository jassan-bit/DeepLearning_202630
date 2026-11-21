# Sesgo de calendario finito: identidad exacta y reparación de alcance

Estado: derivación algebraica revisada por IA, **sin estimación sobre datos, implementación nueva o validación aleatoria**. No es una garantía del bootstrap actual ni un resultado nuevo de originalidad estadística.

## 1. Identidad de la brecha de Jensen

Para una variable X≥0 con μ=E[X]∈(0,∞), se cumple exactamente

\[
\sqrt X=\sqrt\mu+\frac{X-\mu}{2\sqrt\mu}
-\frac{(X-\mu)^2}{2\sqrt\mu(\sqrt X+\sqrt\mu)^2}.
\]

Se obtiene racionalizando √X−√μ y restando el término lineal. No usa una expansión asintótica. Tomando expectativas y usando (√X+√μ)²≥μ:

\[
0\le g=\sqrt\mu-E\sqrt X
=E\!\left[\frac{(X-\mu)^2}{2\sqrt\mu(\sqrt X+\sqrt\mu)^2}\right]
\le\frac{\operatorname{Var}(X)}{2\mu^{3/2}}.
\]

La última cota es finita si Var(X)<∞. Es deliberadamente más conservadora que el término local Var(X)/(8μ^(3/2)) de una expansión delta. Ese término local no es una cota general.

## 2. Aplicación a la RMSE de una ejecución

Condicionando al procedimiento y datos de entrenamiento fijados y a la semilla s, se toma X=\bar q_m,T(s), la media de errores cuadrados sobre el calendario. Si q tiene segundo momento finito y es débilmente estacionario condicionalmente a s,

\[
\operatorname{Var}(\bar q_{m,T}\mid s)=
\frac{T\gamma_{m,s}(0)+2\sum_{h=1}^{T-1}(T-h)\gamma_{m,s}(h)}{T^2}.
\]

Es una identidad exacta para ese diseño. Las covarianzas relevantes son de q=e², no sólo de los retornos. Un lag8 o una suma truncada estimada no son automáticamente una cota superior de toda esta varianza.

Definiendo

\[
G_m=E_S[\sqrt{\mu_m(S)}-E_Z\sqrt{\bar q_{m,T}(S)}],\quad
B_m=E_S\!\left[\frac{\operatorname{Var}(\bar q_{m,T}\mid S)}{2\mu_m(S)^{3/2}}\right],
\]

se tiene 0≤G_m≤B_m, si la esperanza que define B_m es finita. Para theta_j=Psi_j−Psi_c,

\[
b_j=E\widehat\theta_j-\theta_j=G_c-G_j\in[-B_j,B_c].
\]

La desigualdad no necesita independencia entre modelos para las expectativas. Esa dependencia sí importa para la distribución de la raíz y su cobertura.

## 3. Intervalo ampliado sin cambiar el ranking

Sea tau_j=E(thetahat_j) para el diseño finito fijado. Si un procedimiento válido produce [L_j,U_j] que cubre tau_j con probabilidad≥1−alpha_root, y los B_m son cotas conocidas defendibles, entonces

\[
\boxed{CI_{\theta_j}=[L_j-B_c,\ U_j+B_j]}
\]

cubre theta_j al menos con esa probabilidad. En efecto, theta_j=tau_j−b_j y b_j∈[−B_j,B_c]. La ampliación simétrica por B_c+B_j también sería válida, pero más amplia.

**Esto conserva thetahat y la tabla de media de RMSE individuales.** Tau es un objeto auxiliar de la derivación; no se renombra como verdad primaria Psi. Tampoco se ordenan los modelos por una corrección de sesgo elegida después de ver resultados.

Si sólo hay cotas aleatorias Bhat con un evento conjunto de validez de probabilidad≥1−eta, la unión de eventos da cobertura≥1−alpha_root−eta, sin exigir independencia. Para un IC95 se debe repartir de antemano el error total .05; no se puede llamar IC95 a una raíz95% más una cota incierta sin contabilizar eta. La cobertura simultánea de toda la familia requiere eventos simultáneos o contabilizar los errores de cada contraste/modelo.

Ampliar IC no produce automáticamente p válidos para Holm. La nueva construcción tendría que invertir una familia anidada de intervalos válidos o derivar explícitamente p conservadores con el presupuesto de error correspondiente. **El p histórico de P1/P2 no se valida por añadir esta fórmula al informe.**

## 4. Qué falta para aplicarla

| Insumo | Exigencia y estado |
|---|---|
| Media condicional μ_m(s) y riesgos pequeños | La identidad por semilla necesita μ>0. Una media empírica positiva no prueba una cota inferior poblacional; un epsilon numérico no la inventa. Si μ=0 realmente, X=0 casi seguramente y el gap es0; ese caso estructural se trata aparte. |
| Cota de varianza temporal | Necesita autocovarianzas o una cota estructural defendible para la suma completa. Un HAC plug-in no es una cota de confianza demostrada. |
| Esperanza sobre semillas | Tres/cinco semillas y un calendario expuesto no determinan E_S de la razón. Se necesita integrabilidad y, si se estima, controlar también este error. |
| Regularidad cerca de cero | Se requiere E_S[Var(\bar q|S)/μ(S)^(3/2)]<∞ para una cota útil. Puede divergir o ser demasiado ancha. Eso produce resultado inconcluso, no permiso para recortarla. |
| Raíz centrada | Todavía se debe justificar cobertura para tau y la dependencia cruzada completa. El bootstrap actual no la obtiene por esta identidad. |
| Multiplicidad | Definir evento familiar de raíz y sesgo y p/IC coherentes. El plan no transforma IC marginales en simultáneos. |
| Ámbito | Entrenamiento fijado no cubre automáticamente otras muestras de ajuste, cambios de régimen o decisiones de selección. |

## 5. Decisión y contraste con otras rutas

Esta derivación ofrece una reparación inferencial concreta: **conservar la media observada y estudiar intervalos con sesgo acotado**. Sólo cabe validación nueva cuando exista una construcción computable de las cotas y de la raíz, con condiciones explícitas. Si las cotas resultan no identificables o inútiles bajo los supuestos defendibles, se descarta esta ruta para ese alcance.

Una corrección de sesgo o un jackknife temporal sería otra propuesta de inferencia con código, resto y covarianza propios; no una enmienda silenciosa del ranking ni un teorema de validez finita. Condicionar al calendario es otra pregunta, admisible como sensibilidad declarada. Ninguna opción resuelve automáticamente P2 o la cobertura de todos los óptimos del MCS.

No se obtuvo una cota numérica de B_m, una cobertura nueva, un p de mercado, una potencia o una candidata. La revisión independiente por otro agente IA confirmó la identidad, los signos del intervalo y la unión de probabilidades; no equivale a aprobación estadística humana.
