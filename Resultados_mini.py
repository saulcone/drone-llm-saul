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
                    elif objetivo == "collision_detected":
                        estado = "colision"
                    elif objetivo == "max_steps_reached":
                        estado = "max_steps"
                    else:
                        estado = "desconocido"
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
        colisiones = sum(1 for s in estados if s == "colision")
        max_steps = sum(1 for s in estados if s == "max_steps")

        estado = f"repeat {total}"
        resultado = f"{conseguidos}/{total} conseguidos"
        colision_info = f"{colisiones} colisiones" if colisiones > 0 else ""
        max_steps_info = f"{max_steps} max_steps" if max_steps > 0 else ""

        resultados.append((grupo, estado, resultado, colision_info,
                           max_steps_info))

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
    txt_path = os.path.join(json_folder, "results_mini.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Nombre\tEstado\tResultado\n")
        for nombre, estado, resultado, colision, max_step in experimentos:
            f.write(f"{nombre}\t{estado}\t{resultado}\t{colision}\t{max_step}\n")

    for nombre, estado, resultado, colision, max_step in experimentos:
        tree.insert("", tk.END, values=(nombre, estado, resultado, 
                                        colision, max_step))

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
