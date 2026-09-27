import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def plot_tif_con_pendiente_real(file_path):
    print("Cargando imagen...")
    pil_img = Image.open(file_path)
    img_array = np.array(pil_img, dtype=np.float32)

    # 1. Mascarar NoData y valores inválidos
    # El valor NoData formal es 3.4e38, pero filtramos cualquier número <= 0 o > 1000
    # (Ajusta el límite superior si tu zona realmente tiene acantilados > 500%)
    nodata_mask = (
        (img_array >= 1e30)
        | (img_array < 0)
        | np.isnan(img_array)
        | np.isinf(img_array)
    )
    img_array_masked = np.where(nodata_mask, np.nan, img_array)

    # 2. Calcular límites razonables de visualización usando percentiles (ignora valores extremos)
    datos_validos = img_array_masked[~np.isnan(img_array_masked)]

    if len(datos_validos) > 0:
        # P98 ignora el 2% de los píxeles más extremos que arruinan la escala
        vmin = np.percentile(datos_validos, 1)
        vmax = np.percentile(datos_validos, 98)
        print(f"Rango de visualización optimizado: {vmin:.1f}% a {vmax:.1f}%")
        print(f"Valor máximo absoluto en el raster: {datos_validos.max():.1f}%")
    else:
        vmin, vmax = 0, 100

    fig, ax = plt.subplots(figsize=(10, 8))

    # 3. Graficar con vmin/vmax y mapa de colores topográfico ('terrain' o 'YlOrRd')
    cax = ax.imshow(
        img_array_masked, cmap="terrain", vmin=vmin, vmax=vmax, origin="upper"
    )

    cbar = fig.colorbar(cax, label="Pendiente (%)", extend="max")

    ax.set_title("Pendiente del Terreno - Haz clic para consultar el valor real")

    # 4. Evento de clic (sigue mostrando el valor exacto sin recortar)
    def on_click(event):
        if event.xdata is not None and event.ydata is not None:
            col = int(round(event.xdata))
            row = int(round(event.ydata))

            height, width = img_array.shape[0], img_array.shape[1]
            if 0 <= row < height and 0 <= col < width:
                val = img_array_masked[row, col]

                if np.isnan(val):
                    texto = f"X: {col}, Y: {row} | Sin Datos (NoData)"
                else:
                    texto = f"X: {col}, Y: {row} | Pendiente: {val:.2f}%"

                print(texto)
                ax.set_title(texto)
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()


# Ejecutar:
plot_tif_con_pendiente_real(
    "/home/felipe/Desktop/Trabajo/Data/pendiente/Pendiente.tif"
)