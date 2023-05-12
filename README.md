# Proceso de despliegue en local - TICKIT

Este proyecto contiene cada uno de los archivos creados y usados para el despliegue de un ApiRest en Google Cloud Plataform que pronóstica ventas futuras de boletos para un evento especifico (según el problema dado). 

El Flujo de trabajo que se siguió fue el siguiente:

1. Se realizó el análisis exploratorio correspondiente de los datos dados.
2. Se creó y entrenó el modelo más óptimo encontrado.
3. Se creó un API con FastApi como se especificaba.
4. Se creó el Dockerfile correspondiente.
5. Se creó la imagen usando el script nombrado docker_buil.sh que ejecuta los comandos para realizar este proceso.
6. Se ejecutó el contenedor usando el script docker_run.sh para validar el correcto funcionamiento de la API.
7. Se desplegó la imagen que se encuentra en Docker en la nube de Google.
8. Se creó un archivo llamado requeriments.txt que contiene las librerías con las versiones utilizadas para ejecutar la API.


