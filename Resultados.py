import os
import json
import tkinter as tk
from tkinter import ttk
import argparse

# intervalo de actualización en segundos
REFRESH_INTERVAL = 2

def leer_experimentos(json_folder):
    """
    Lee los archivos JSON y devuelve una lista
    de tuplas (nombre del archivo, estado, resultado/ejecución)
    """
    resultados = []

    for archivo in os.listdir(json_folder):
        if archivo.endswith(".json"):
            try:
                with open(os.path.join(json_folder, archivo), "r") as f:
                    data = json.load(f)

                nombre_archivo = archivo

                # comprobar si ha concluido
                end_info = data.get("End_Info", None)
                if end_info:
                    estado = "concluido"
                    resultado = end_info.get("result", "desconocido")
                else:
                    # buscar el último paso
                    pasos = [key for key in data.keys() if key.startswith("Step_")]
                    if pasos:
                        ultimo_paso = max(pasos, key=lambda x: int(x.split("_")[1]))
                        paso_info = data[ultimo_paso]
                        paso_numero = paso_info.get("step", "?")
                        estado = "en ejecución"
                        resultado = f"paso {paso_numero}"
                    else:
                        estado = "en ejecución"
                        resultado = "sin pasos"

                resultados.append((nombre_archivo, estado, resultado))

            except Exception as e:
                print(f"Error leyendo {archivo}: {e}")

    return resultados

def actualizar_tabla(json_folder):
    """
    Actualiza la tabla y guarda los resultados en results.txt
    """
    for row in tree.get_children():
        tree.delete(row)

    experimentos = leer_experimentos(json_folder)

    # poner el título de la ventana como el primer experimento, opcional
    if experimentos:
        root.title(experimentos[0][0])
    else:
        root.title("Monitor Experimentos")

    # escribir resultados en results.txt
    with open("results.txt", "w", encoding="utf-8") as f:
        f.write("Nombre\tEstado\tResultado\n")
        for nombre, estado, resultado in experimentos:
            f.write(f"{nombre}\t{estado}\t{resultado}\n")

    # llenar la tabla
    for nombre, estado, resultado in experimentos:
        tree.insert("", tk.END, values=(nombre, estado, resultado))

    # volver a actualizar
    root.after(REFRESH_INTERVAL * 1000, lambda: actualizar_tabla(json_folder))

def iniciar_gui(json_folder):
    global root, tree
    root = tk.Tk()
    root.title("Monitor Experimentos")

    columns = ("Nombre archivo JSON", "Estado", "Resultado/Ejecución")
    tree = ttk.Treeview(root, columns=columns, show="headings")

    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=300)

    tree.pack(expand=True, fill="both")

    actualizar_tabla(json_folder)

    root.mainloop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor Experimentos JSON")
    parser.add_argument(
        "json_folder",
        nargs="?",
        default="./experimentos",
        help="Carpeta donde están los archivos JSON (por defecto ./experimentos)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.json_folder):
        print(f"Error: la carpeta {args.json_folder} no existe.")
        exit(1)

    iniciar_gui(args.json_folder)
