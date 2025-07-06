import os
import json
import tkinter as tk
from tkinter import ttk
from collections import defaultdict
import argparse

# intervalo de actualización en segundos
REFRESH_INTERVAL = 2

def leer_experimentos_agrupados(json_folder):
    """
    Lee los archivos JSON de la carpeta y agrupa por query + spawn_id
    devolviendo una lista de tuplas (nombre_grupo, estado, resultado/ejecución)
    """
    agrupados = defaultdict(list)

    for archivo in os.listdir(json_folder):
        if archivo.endswith(".json"):
            try:
                with open(os.path.join(json_folder, archivo), "r") as f:
                    data = json.load(f)

                exp_info = data.get("Experiment_Info", {})
                query = exp_info.get("query", "Unknown query")
                spawn_id = exp_info.get("spawn_id", "Unknown spawn")

                clave = f"{query} | spawn {spawn_id}"

                # saber si ha terminado
                end_info = data.get("End_Info", None)
                if end_info:
                    objetivo = end_info.get("result", "")
                    if objetivo == "goal_reached":
                        estado = "conseguido"
                    else:
                        estado = "chocado"
                else:
                    estado = "en ejecución"

                agrupados[clave].append(estado)

            except Exception as e:
                print(f"Error leyendo {archivo}: {e}")

    # transformar a lista de tuplas
    resultados = []
    for grupo, estados in agrupados.items():
        total = len(estados)
        conseguidos = sum(1 for s in estados if s == "conseguido")

        estado = f"repeat {total}"
        resultado = f"{conseguidos}/{total} conseguidos"

        resultados.append((grupo, estado, resultado))

    return resultados

def actualizar_tabla(json_folder):
    """
    Actualiza la tabla agrupada y guarda los resultados en results.txt
    """
    for row in tree.get_children():
        tree.delete(row)

    experimentos = leer_experimentos_agrupados(json_folder)

    # actualizar título
    if experimentos:
        root.title(experimentos[0][0])
    else:
        root.title("Monitor Experimentos Agrupados")

    # escribir en results.txt
    with open("results_mini.txt", "w", encoding="utf-8") as f:
        f.write("Nombre\tEstado\tResultado\n")
        for nombre, estado, resultado in experimentos:
            f.write(f"{nombre}\t{estado}\t{resultado}\n")

    for nombre, estado, resultado in experimentos:
        tree.insert("", tk.END, values=(nombre, estado, resultado))

    # volver a actualizar
    root.after(REFRESH_INTERVAL * 1000, lambda: actualizar_tabla(json_folder))

def iniciar_gui(json_folder):
    global root, tree
    root = tk.Tk()
    root.title("Monitor Experimentos Agrupados")

    columns = ("Grupo (query | spawn)", "Estado (repeats activos/total)", "Resultado/Ejecución")
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
