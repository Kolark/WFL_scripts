# Wake Model Testing
Para probar las funciones de src/wake_models
* **wake_methods_benchmark.py**: Evalua con diferentes números de turbinas el wake_loss según diferentes implementaciones:
    - calc_wake_wrf: implementación original del modelo de wake loss (Hecha por Jose)
    - implementación serial: Toma la implementación de calc_wake_wrf y cambia la estructura con el objetivo de optimizar los cálculos ademas de usar njit.
    - implementación paralela: Version muy similar a la implementación serial pero con cambios menores para que la evaluación del modelo se ejecute en multiples hilos.

    La idea detras de este script es comparar los resultados de las diferentes implementaciónes para verificar la mejora que representa en cuanto al tiempo de ejecución. Ademas de asegurar que los resultados sean iguales para los mismos datos de entrada.
* **benchmark_explore.ipynb**: Notebook para explorar los resultados de ***wake_methods_benchmark.py***. Compara los tiempos de ejecución de cada implementación y la diferencia entre sus resultados(que deberia ser cercana a 0).

# Wind Dataset Graphs
Scripts para visualizar datos de viento en el tiempo de manera interactiva. Utiliza la clase **WindDataset** para cargar el .nc y obtener los datos de intensidad y dirección del viento.

* **wind_data_graph.py**: Visualiza datos de intensidad de viento y dirección a traves de un mapa de calor, y la dirección de este con un gráfico de flechas(quiver). Toma toda la región del dominio 2 que esta en el archivo ***ws_wd_ctm12_chunked_uvnorm.nc***.

* **wind_fullsample_graph.py**: Visualiza datos de intensidad de viento y dirección a traves de un gráfico de flechas, donde el color representa la intensidad del viento, esto para todo el dominio 2. A diferencia de *wind_data_graph.py*, este no muestra cada pixel, si no que genera coordenadas aleatorias limitadas al dominio 2 y obtiene su correspondiente velocidad y dirección.

* **wind_sample_graph.py**: Visualiza datos de intensidad de viento y dirección a traves de un gráfico de flechas, donde el color representa la intensidad del viento, esto para una subregion de el dominio 2. Ademas se encarga de cargar la subregión en memoria y luego consultar los datos para las correspondientes coordenadas.

* **plot_utils.py**: Script que contiene utilidades generales para gráficar datos de viento. Ayuda a mantener los otros scripts mas limpios y por tanto legible.