# Neko_AI
Primeros intentos de predicción de bolsa utilizando redes neuronales realizados en 2021.</br> 

## Resultados
De todas las versiones probablemente la más eficaz es la que se encuentra en la carpeta principal, seguida de la v2. Aunque ninguna es suficientemente capaz de sobrepasar el 2% de comisión por transacción del mundo real.</br>

Está todo bastante desordenado ya que son básicamente pruebas y herramientas de testing interno.</br>

## Versiones
Hay varias versiones:</br>
* La versión de livetrade es capaz de comprar y vender en el mercado real a intervalos regulares. </br>
* Las versiones de backtesting son capaces de simular el mercado step by step a partir de un archivo csv e ir ejecutando las decisiones tomadas por una función "brain" arbitraria. Tiene en cuenta las comisiones de la casa de cambio y puede realizar compras de un porcentaje arbitrario del portfolio, por lo que las predicciones deberían de ser bastante precisas. El resultado de la simulación se representa en una gráfica.</br>
* La versión NEAT utiliza el algoritmo de evolución de redes del mismo nombre, pero evidentemente no funciona bien porque no está hecho para eso, se hizo nada más para probar la librería.</br>

