"""
mna_gui_with_canvas.py

Interfaz gráfica para ingresar resistencias (R) y fuentes de voltaje (V),
resolver el circuito por MNA (Modified Nodal Analysis), calcular corrientes
en resistencias, y dibujar un diagrama usando Tkinter Canvas.

"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import math
from PIL import Image, ImageTk

# -------------------------
# Utilidades y conversión
# -------------------------
def convertir_valor(valor_str):
    """
    Convierte una cadena con prefijo SI (k, M, m, u) a número float.
    Ejemplos: "2k" -> 2000.0, "4M" -> 4e6, "3m" -> 3e-3, "6u" -> 6e-6, "12" -> 12.0
    """
    s = valor_str.strip()
    if not s:
        raise ValueError("Valor vacío")
    multipliers = {'k':1e3, 'M':1e6, 'm':1e-3, 'u':1e-6}
    if s[-1] in multipliers:
        num = float(s[:-1])
        return num * multipliers[s[-1]]
    else:
        return float(s)

# -------------------------
# Lógica MNA y cálculos
# -------------------------
def construir_y_resolver(elementos):
    """
    Elementos: lista de tuplas (tipo, np, nn, val)
      - tipo: "R" o "V"
      - np, nn: enteros, nodos (0 es tierra)
      - val: float (resistencia en ohms o voltaje en V)

    Devuelve:
      - nodos_lista: lista de nodos no-cero (orden usada)
      - Vn: dict {nodo: voltaje} (incluye nodo 0 = 0)
      - corriente_fuentes: lista de corrientes para cada fuente V en orden de aparición
      - corriente_resistencias: lista de corrientes por cada R (en el mismo orden que elementos)
      - potencias_resistencias: lista de potencias absorbidas por cada R (W)
    """
    # 1) Determinar nodos (excluyendo tierra 0)
    nodos = set()
    for (t, np_, nn_, v) in elementos:
        nodos.add(np_)
        nodos.add(nn_)
    if 0 in nodos:
        nodos.remove(0)
    nodos = sorted(list(nodos))
    n = len(nodos)

    # 2) Mapear nodo -> índice en la matriz (0..n-1)
    nodo_index = {nodo: i for i, nodo in enumerate(nodos)}

    # 3) Contar fuentes de voltaje (m)
    fuentes = [(t, np_, nn_, v) for (t, np_, nn_, v) in elementos if t == "V"]
    m = len(fuentes)

    # 4) Construir matrices G (n x n), B (n x m), I (n x 1), E (m x 1)
    G = np.zeros((n, n))
    B = np.zeros((n, m))
    I = np.zeros((n, 1))  # corrientes independientes 
    E = np.zeros((m, 1))

    # Llenar G y B
    idx_v = 0
    for elem in elementos:
        tipo, np_, nn_, val = elem
        if tipo == "R":
            # conductancia
            g = 1.0 / val
            # si nodo positivo no es tierra, sumar conductancia en diagonal
            if np_ != 0:
                i = nodo_index[np_]
                G[i, i] += g
            if nn_ != 0:
                j = nodo_index[nn_]
                G[j, j] += g
            if np_ != 0 and nn_ != 0:
                i = nodo_index[np_]; j = nodo_index[nn_]
                G[i, j] -= g
                G[j, i] -= g
        elif tipo == "V":
            # Para fuentes de voltaje llenamos B y E
            if np_ != 0:
                B[nodo_index[np_], idx_v] = 1
            if nn_ != 0:
                B[nodo_index[nn_], idx_v] = -1
            E[idx_v, 0] = val
            idx_v += 1
        else:
            raise ValueError("Tipo de elemento desconocido: " + str(tipo))

    # 5) Formar la matriz aumentada A y el vector z
    if n == 0:
        # Sólo hay tierra: manejar caso trivial
        Vn = {}
        corriente_fuentes = []
        corriente_resistencias = []
        potencias_resistencias = []
        # si hay fuentes, su corriente es E/0 => mal planteado (singular). devolvemos error.
        if m > 0:
            raise np.linalg.LinAlgError("Circuito singular: no nodos, pero hay fuentes de voltaje.")
        return [], Vn, [], [], []

    A = np.block([
        [G, B],
        [B.T, np.zeros((m, m))]
    ])
    z = np.vstack((I, E))

    # 6) Resolver el sistema lineal A x = z
    try:
        x = np.linalg.solve(A, z)
    except np.linalg.LinAlgError as e:
        # Propagar error para que la GUI informe
        raise

    # x contiene [Vn (n x 1); Iv (m x 1)]
    Vn_vec = x[:n, 0]
    Iv_vec = x[n:, 0]

    # 7) Construir diccionario de voltajes con nodo 0 = 0
    Vn = {0: 0.0}
    for i, nodo in enumerate(nodos):
        Vn[nodo] = Vn_vec[i]

    # 8) Calcular corrientes en resistencias (por cada elemento R)
    corriente_resistencias = []
    potencias_resistencias = []
    idx_f = 0
    for elem in elementos:
        tipo, np_, nn_, val = elem
        if tipo == "R":
            # convención: corriente positiva VA -> VB (desde nodo+ hacia nodo-)
            Va = Vn.get(np_, 0.0)
            Vb = Vn.get(nn_, 0.0)
            I_R = (Va - Vb) / val  # Ampers
            P_R = I_R * I_R * val  # P = I^2 * R 
            corriente_resistencias.append(I_R)
            potencias_resistencias.append(P_R)
        elif tipo == "V":
            idx_f += 1
        else:
            pass

    # 9) Corrientes de fuentes: Iv_vec (en orden de aparición de fuentes V)
    corriente_fuentes = list(Iv_vec)

    return nodos, Vn, corriente_fuentes, corriente_resistencias, potencias_resistencias

class CircuitDrawer:
    """
    Dibujo simplificado
    - Nodos en fila horizontal
    - Elementos representados como rectángulos (R) y círculos (V)
    """

    def __init__(self, canvas, elementos):
        self.canvas = canvas
        self.elementos = elementos
        self.node_positions = {}
        self.radius = 10  
        self.elem_width = 60
        self.elem_height = 20

    def compute_node_order(self):
        # Ordenar nodos según aparición
        nodos = set()
        for t, np, nn, val in self.elementos:
            nodos.add(np)
            nodos.add(nn)
        nodos = list(nodos)
        if 0 in nodos:
            nodos.remove(0)
        nodos.sort()
        nodos = [0] + nodos  # Tierra primero
        return nodos

    def assign_positions(self, nodos):
        canvas_width = int(self.canvas.winfo_width())
        spacing = 220 

        # Ancho total que ocuparán los nodos
        total_width = spacing * (len(nodos) - 1)
        
        x_start = (canvas_width - total_width) // 2  
        if x_start < 50: x_start = 50

        y_base = 150  

        positions = {}
        for i, n in enumerate(nodos):
            positions[n] = (x_start + i * spacing, y_base)

        self.node_positions = positions

    def draw_node(self, x, y, n):
        self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="black")
        self.canvas.create_text(x, y - 15, text=f"N{n}", font=("Arial", 9, "bold"))

    def draw_wire_connection(self, x_node, y_node, x_comp, y_comp):
        """Dibuja la 'patita' vertical desde el nodo hasta la altura del componente"""
        self.canvas.create_line(x_node, y_node, x_node, y_comp, width=2)
        # Un pequeño punto de soldadura en la intersección si hay desplazamiento vertical
        if abs(y_node - y_comp) > 5:
            r = 3
            self.canvas.create_oval(x_node-r, y_comp-r, x_node+r, y_comp+r, fill="black")    

    def draw_resistor(self, x1, x2, y_node, y_comp, name, val):
        midx = (x1 + x2) / 2
        # La altura (y) es fija en y_comp para el componente
        
        # 1. Patitas verticales
        self.draw_wire_connection(x1, y_node, x1, y_comp)
        self.draw_wire_connection(x2, y_node, x2, y_comp)

        # 2. Líneas horizontales hacia el componente
        self.canvas.create_line(x1, y_comp, midx - self.elem_width/2, y_comp, width=2)
        self.canvas.create_line(midx + self.elem_width/2, y_comp, x2, y_comp, width=2)

        # 3. Rectángulo de elemento
        self.canvas.create_rectangle(
            midx - self.elem_width/2, y_comp - self.elem_height/2,
            midx + self.elem_width/2, y_comp + self.elem_height/2,
            fill="#fcecdc", outline="black"
        )

        # 4. Etiquetas (Nombre arriba, valor abajo)
        self.canvas.create_text(midx, y_comp - 18, text=name, font=("Arial", 10, "bold"), fill="blue")
        self.canvas.create_text(midx, y_comp + 18, text=format_val(val), font=("Arial", 10))

    def draw_voltage(self, x1, x2, y_node, y_comp, name, val):
        midx = (x1 + x2) / 2
        r = 20  # Radio de la fuente

        # 1. Patitas verticales
        self.draw_wire_connection(x1, y_node, x1, y_comp)
        self.draw_wire_connection(x2, y_node, x2, y_comp)

        # 2. Patitas horizontales
        self.canvas.create_line(x1, y_comp, midx - r, y_comp, width=2)
        self.canvas.create_line(midx + r, y_comp, x2, y_comp, width=2)

        # 3. Círculo
        self.canvas.create_oval(midx - r, y_comp - r, midx + r, y_comp + r, width=2, fill="white")

        # 4. Signos + y -
        self.canvas.create_text(midx - 10, y_comp - 8, text="+", font=("Arial", 12, "bold"))
        self.canvas.create_text(midx + 10, y_comp - 8, text="-", font=("Arial", 14, "bold"))

        # 5. Etiquetas
        self.canvas.create_text(midx, y_comp - 28, text=name, font=("Arial", 10, "bold"), fill="blue")
        self.canvas.create_text(midx, y_comp + 28, text=format_val(val), font=("Arial", 10))

    def draw(self):
        self.canvas.delete("all")

        if not self.elementos:
            return

        nodos = self.compute_node_order()
        self.assign_positions(nodos)

        # Dibujar línea base de nodos 
        if len(nodos) > 1:
            x_min = self.node_positions[nodos[0]][0]
            x_max = self.node_positions[nodos[-1]][0]
            y_base = self.node_positions[nodos[0]][1]
            self.canvas.create_line(x_min, y_base, x_max, y_base, width=3, fill="#555")

        # Dibujar puntos nodales
        for n, (x, y) in self.node_positions.items():
            self.draw_node(x, y, n)

        # ----- AGRUPAR ELEMENTOS POR PARES DE NODO -----
        groups = {}
        for elem in self.elementos:
            t, np_, nn_, val = elem
            key = tuple(sorted([np_, nn_])) # Clave ordenada para agrupar sin importar dirección
            if key not in groups:
                groups[key] = []
            groups[key].append(elem)

        # ----- DIBUJAR ELEMENTOS POR GRUPO -----
        rcount = 1
        vcount = 1

        for key, elems in groups.items():
            np_, nn_ = key
            
            # Verificar que existan en el diccionario de posiciones
            if np_ not in self.node_positions or nn_ not in self.node_positions:
                continue

            x1, y_node = self.node_positions[np_]
            x2, _ = self.node_positions[nn_]

            total = len(elems)
            step_y = 100  # Espacio vertical entre componentes 

            # Cálculo del offset inicial para centrar verticalmente
            start_offset = -((total - 1) * step_y) / 2

            for i, (t, np_orig, nn_orig, val) in enumerate(elems):
                current_offset = start_offset + (i * step_y)
                y_comp = y_node + current_offset

                if t == "R":
                    name = f"R{rcount}"
                    rcount += 1
                    self.draw_resistor(x1, x2, y_node, y_comp, name, val)
                elif t == "V":
                    name = f"V{vcount}"
                    vcount += 1
                    self.draw_voltage(x1, x2, y_node, y_comp, name, val)


# Helper para formatear valores con prefijos para el dibujo
def format_val(v):
    """
    Formatea un valor numérico en una cadena con prefijo si es apropiado.
    Ej: 2000 -> '2.00k', 0.000001 -> '1.00u'
    """
    if v == 0:
        return "0"
    abs_v = abs(v)
    if abs_v >= 1e6:
        return f"{v/1e6:.2f}M"
    elif abs_v >= 1e3:
        return f"{v/1e3:.2f}k"
    elif abs_v >= 1:
        return f"{v:.2f}"
    elif abs_v >= 1e-3:
        return f"{v*1e3:.2f}m"
    elif abs_v >= 1e-6:
        return f"{v*1e6:.2f}u"
    else:
        return f"{v:.2e}"

# ---------------------------
# Interfaz principal (Tkinter)
# -------------------------
class MNA_GUI:
    def __init__(self, root):
        # === Panel derecho de instrucciones ===
        frm_help = tk.Frame(root, bd=2, relief="groove", padx=10, pady=10)
        frm_help.pack(side="right", fill="y")

        help_text = (
            "INSTRUCCIONES:\n\n"
            "1. Selecciona el tipo de elemento:\n"
            "   - R = Resistencia\n"
            "   - V = Fuente de Voltaje\n\n"
            "2. Ingresa los nodos:\n"
            "   • Nodo donde inicia -> Nodo (+)\n"
            "   • Nodo donde termina -> Nodo (-)\n"
            "   • Usa 0 para tierra\n\n"
            "3. Ingresa el valor con o sin prefijos SI:\n"
            "   Ejemplos: 2k, 5m, 10u, 3.3M, 47\n\n"
            "4. Presiona «Agregar elemento».\n\n"
            "5. Observa el dibujo del circuito abajo.\n\n"
            "6. Presiona «Resolver circuito (MNA)» para obtener:\n"
            "   - Voltajes nodales\n"
            "   - Corrientes por fuentes\n"
            "   - Corrientes y potencias en resistencias\n\n"
            "7. Usa «Limpiar elementos» para reiniciar.\n"
        )
        lbl_help = tk.Label(frm_help, text=help_text, justify="left", anchor="nw", font=("Arial", 10))
        lbl_help.pack(fill="both", expand=True)

        #--------------Imagenes---------------------
        try:
            ruta_imagen1 = "formulario1.jpg"
            # CORRECCIÓN: Usar ruta_imagen1
            imagen_pil1 = Image.open(ruta_imagen1) 

            base_width = 200
            # CORRECCIÓN: Usar imagen_pil1 en todos los cálculos
            w_percent = (base_width / float(imagen_pil1.size[0]))
            h_size = int((float(imagen_pil1.size[1]) * float(w_percent)))
            imagen_redimensionada1 = imagen_pil1.resize((base_width, h_size), Image.Resampling.LANCZOS)

            self.foto_tk1 = ImageTk.PhotoImage(imagen_redimensionada1)
            
            lbl_imagen1 = tk.Label(frm_help, image=self.foto_tk1)
            lbl_imagen1.pack(side="bottom", pady=(10, 20)) 

        except Exception as e:
            print(f"Error imagen 1: {e}")
            lbl_error = tk.Label(frm_help, text="(formulario1.jpg no encontrada)", fg="red")
            lbl_error.pack(side="bottom", pady=5)

       

        # Ventana principal
        self.root = root
        root.title("Analizador de Circuitos - MNA (Con dibujo)")
        root.geometry("1050x700")

        # Lista de elementos (tipo, nodo+, nodo-, valor)
        self.elementos = []

        # ---------- Input: tipo, nodos y valor ----------
        frm_top = tk.Frame(root)
        frm_top.pack(fill="x", padx=6, pady=6)

        tk.Label(frm_top, text="Tipo").grid(row=0, column=0, padx=4)
        tk.Label(frm_top, text="Nodo donde inicia (nodo +)").grid(row=0, column=1)
        tk.Label(frm_top, text="Nodo donde termina (nodo -)").grid(row=0, column=2)
        tk.Label(frm_top, text="Valor").grid(row=0, column=3, padx=4)

        self.tipo_var = tk.StringVar()
        self.tipo_cb = ttk.Combobox(frm_top, textvariable=self.tipo_var, values=["R", "V"], width=6, state="readonly")
        self.tipo_cb.current(0)
        self.tipo_cb.grid(row=1, column=0)

        self.np_entry = tk.Entry(frm_top, width=8)
        self.nn_entry = tk.Entry(frm_top, width=8)
        self.val_entry = tk.Entry(frm_top, width=12)
        self.np_entry.grid(row=1, column=1, padx=6)
        self.nn_entry.grid(row=1, column=2, padx=6)
        self.val_entry.grid(row=1, column=3, padx=6)

        btn_add = tk.Button(frm_top, text="Agregar elemento", command=self.agregar)
        btn_add.grid(row=1, column=4, padx=6)
        btn_add = tk.Button(frm_top, text="Agregar elemento", command=self.agregar,
                            bg="#0275d8", fg="white", font=("Arial", 9, "bold"))
        btn_add.grid(row=1, column=4, padx=6)

        # ---------- Tabla (Treeview) de elementos ----------
        columns = ("tipo", "np", "nn", "val")
        self.tree = ttk.Treeview(root, columns=columns, show="headings", height=8)
        for c in columns:
            self.tree.heading(c, text=c.upper())
        self.tree.pack(fill="x", padx=6, pady=6)


        # ---------- Canvas para dibujo ----------
        canvas_frame = tk.Frame(root, bd=2, relief="sunken")
        canvas_frame.pack(fill="both", expand=True, padx=6, pady=6)
        
        #GRID
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)

        #barrita de desplazamiento
        self.v_scroll = tk.Scrollbar(canvas_frame, orient="vertical")
        self.h_scroll = tk.Scrollbar(canvas_frame, orient="horizontal")
        self.canvas = tk.Canvas(canvas_frame, bg="white",
                                yscrollcommand=self.v_scroll.set,
                                xscrollcommand=self.h_scroll.set)
        self.v_scroll.config(command=self.canvas.yview)
        self.h_scroll.config(command=self.canvas.xview)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scroll.grid(row=0, column=1, sticky="ns")
        self.h_scroll.grid(row=1, column=0, sticky="ew")


        # ====== Frame para botones de acciones ======
        frm_bottom = tk.Frame(root)
        frm_bottom.pack(fill="x", padx=10, pady=5)

        # Botón para limpiar elementos
        btn_clear = tk.Button(frm_bottom, text="Limpiar elementos", command=self.limpiar, width=18)
        btn_clear.grid(row=0, column=0, padx=10)
        btn_clear = tk.Button(frm_bottom, text="Limpiar elementos", command=self.limpiar, 
                              width=18, bg="#f0ad4e", fg="white", font=("Arial", 9, "bold"))
        btn_clear.grid(row=0, column=0, padx=10)

        btn_solve = tk.Button(frm_bottom, text="Resolver circuito (MNA)", command=self.resolver, width=18)
        btn_solve.grid(row=0, column=1, padx=10)
        btn_solve = tk.Button(frm_bottom, text="Resolver circuito (MNA)", command=self.resolver, 
                              width=20, bg="#5cb85c", fg="white", font=("Arial", 9, "bold"))
        btn_solve.grid(row=0, column=1, padx=10)

        btn_del_sel = tk.Button(frm_bottom, text="Eliminar seleccionado", command=self.eliminar_seleccionado, width=18)
        btn_del_sel.grid(row=0, column=2, padx=10)
        btn_del_sel = tk.Button(frm_bottom, text="Eliminar seleccionado", command=self.eliminar_seleccionado, 
                                width=20, bg="#E88CCF", fg="white", font=("Arial", 9, "bold"))
        btn_del_sel.grid(row=0, column=2, padx=10)

        # ====== RESULTADOS ======
        self.results_frame = tk.Frame(root)
        self.results_frame.pack(fill="x", padx=10, pady=10)

        self.result_text = tk.Text(self.results_frame, width=100, height=100)
        self.result_text.pack(fill="both", expand=True)

        # Dibujo inicial vacío
        self.redraw()


    def agregar(self):
        """
        Lee las entradas, valida, agrega a la lista de elementos y actualiza la tabla y dibujo
        """
        tipo = self.tipo_var.get()
        try:
            np_ = int(self.np_entry.get())
            nn_ = int(self.nn_entry.get())
            val = convertir_valor(self.val_entry.get())
        except Exception as e:
            messagebox.showerror("Error", f"Datos inválidos: {e}")
            return

        # Guardar el elemento
        self.elementos.append((tipo, np_, nn_, val))
        # Agregar a la vista de tabla
        self.tree.insert("", tk.END, values=(tipo, np_, nn_, format_val(val)))

        # Limpiar inputs
        self.np_entry.delete(0, tk.END)
        self.nn_entry.delete(0, tk.END)
        self.val_entry.delete(0, tk.END)

        # Redibujar canvas
        self.redraw()

    def eliminar_seleccionado(self):
        selected_item = self.tree.selection() # Obtiene el ID del item seleccionado
        
        if not selected_item:
            messagebox.showwarning("Para eliminar", "Selecciona un elemento de la lista para eliminarlo.")
            return

        #sacamos el indice item de la tabla
        index = self.tree.index(selected_item)
        del self.elementos[index] #elimina
        self.tree.delete(selected_item)
        self.result_text.delete("1.0", tk.END) #limpia
        self.redraw()

    def limpiar(self):
        """Limpia lista de elementos, tabla y canvas"""
        self.elementos = []
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.redraw()
        self.result_text.delete("1.0", tk.END)

    def redraw(self):
        """Redibuja el circuito en el canvas usando CircuitDrawer y actualiza el scroll"""
        drawer = CircuitDrawer(self.canvas, self.elementos)
        drawer.draw()

        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def resolver(self):
        """Llamar a la función de MNA, mostrar resultados y actualizar el dibujo si hace falta"""
        if not self.elementos:
            messagebox.showerror("Error", "No hay elementos en el circuito")
            return

        try:
            nodos, Vn, corr_fuentes, corr_res, pot_res = construir_y_resolver(self.elementos)
        except np.linalg.LinAlgError:
            messagebox.showerror("Error", "El circuito es singular (mal definido) - revisa conexiones")
            return
        except Exception as e:
            messagebox.showerror("Error", f"Error al resolver: {e}")
            return

        # Mostrar resultados en el textbox
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, "=== Voltajes nodales ===\n")
        # Asegurarse de imprimir nodos en orden
        keys = sorted(list(Vn.keys()))
        for k in keys:
            self.result_text.insert(tk.END, f"V({k}) = {Vn[k]:.6f} V\n")

        self.result_text.insert(tk.END, "\n=== Voltajes en Resistencias ===\n")
        r_idx = 0
        for elem in self.elementos:
            tipo, np_, nn_, val = elem
            if tipo == "R":
                Va = Vn.get(np_, 0.0)
                Vb = Vn.get(nn_, 0.0)
                V_R = Va - Vb
                self.result_text.insert(tk.END, f"R{r_idx+1}: VR = {V_R:.6f} V\n")
                r_idx += 1

        self.result_text.insert(tk.END, "\n=== Corrientes en fuentes de voltaje ===\n")
        for i, I in enumerate(corr_fuentes):
            self.result_text.insert(tk.END, f"I(V{i+1}) = {I:.6f} A\n")

        # Imprimir corrientes en resistencias (en orden de aparición)
        self.result_text.insert(tk.END, "\n=== Corrientes y potencias en resistencias ===\n")
        r_idx = 0
        for elem in self.elementos:
            if elem[0] == "R":
                I_R = corr_res[r_idx]
                P_R = pot_res[r_idx]
                self.result_text.insert(tk.END, f"R{r_idx+1}: I = {I_R:.6f} A   P = {P_R:.6f} W\n")
                r_idx += 1

        # Re-dibujar 
        self.redraw()

# -------------------------
# Ejecutar GUI
# -------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = MNA_GUI(root)
    root.mainloop()