# Mail Pablo
### On Thu, 23 Nov 2017 at 16:25 Pablo Granitto <granitto@cifasis-conicet.gov.ar> wrote:

Como muestra tomo RF entrenando con los 3 tiles, el ultimo caso:

b261 b278 b264 (TRAIN) Vs. b262 (TEST)
              precision    recall  f1-score   support

           0       0.99      1.00      1.00     20000
           3       0.98      0.52      0.68       297

avg / total       0.99      0.99      0.99     20297

b261 b278 b264 (TRAIN) Vs. b262 (TEST)
              precision    recall  f1-score   support

           0       0.98      1.00      0.99      5000
           3       0.99      0.60      0.75       297

avg / total       0.98      0.98      0.97      5297


b261 b278 b264 (TRAIN) Vs. b262 (TEST)
              precision    recall  f1-score   support

           0       0.96      1.00      0.98      2500
           3       0.97      0.65      0.78       297

avg / total       0.96      0.96      0.96      27970

Lo que esta pasando es que al balancear las clases balancea los
errores... esta eligiendo un distinto punto de trabajo en la relacion
FP-FN... si miran las AUC para SVM lineal y para RF dan lo mismo. O sea
que esta aprendiendo lo mismo en todos los casos.
Y eso basicamente dice que son buenos clasificadores los dos, y que no
les cambia mucho el desbalanceo entre clases. Esa de por si es una
conclusion que no es obvia desde el punto de vista de mineria.

Todo este analisis vale.
Resumen:
1.agregar tiles no cambia. Toda la info esta en un tile, los otros
parecen ser iguales.
2.El desbalance no cambia la performance general del modelo. Si el punto
de trabajo del clasificador, que es algo a definir.

Como sigue: proba con balancear a 2500 las dos clases. O sea,
subsampleas la clase 0 al azar, y usas SMOTE para aumentar la clase 3
hasta 2500. Pueden ser un poco variables esos numeros, la idea es
igualar las dos clases cerca de 2500.

Dale nomas...

----

## On Thu, 30 Nov 2017 at 19:08 Pablo Granitto <granitto@cifasis-conicet.gov.ar> wrote:

Bueno, muy buen laburo de programador ;)
Ahora hay que usar el cerebro, aprender lecciones y ver como desarrollar
un metodo para "aprender" en astronomia...

Les paso lo que veo:
Las curvas ROC son iguales en los cuatro casos (salvo fluctuaciones
minimas) en todas las situaciones. Eso quiere decir que:
1. el tipo de modelos de aprendizaje que estamos usando no se deja
influenciar por la clase mayoritaria, salvo para ver donde pone el
"umbral" de decision. Visto como un problema de "deteccion de clase 3",
todos los sampleos y sobresampleos dan lo mismo.
2. no hay informacion nueva en los otros tiles, practicamente. Uno
podria aprender el detector en un unico tile y usarlo en todos los demas
y esperar el mismo resultado que combinando muchos.
3. Las diferencias entre clasificares solo tienen sentido mirando las
curvas ROC, no hay que mirar otra cosa.
4. Mirando los resultados globalmente, da la sensacion que como problema
de deteccion estamos en el limite de eficiencia, que el problema son los
features que usamos que no tienen mas informacion que extraerle.

Ahora hay que ver si estas 4 cosas se mantienen al ampliar la clase 0.

Dale para adelante!
Saludos!
Pablo
