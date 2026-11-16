"""
verificar_documento.py - Comprueba core.py con el ejemplo de 8 puntos del documento
(Reconocimiento de señales por análisis de banco de filtros, señales tipo A y B).

Usa las mismas funciones que la aplicación y compara contra los valores del documento.
Uso:  python verificar_documento.py
"""
import numpy as np
import core

A = [(2, 1, 2, 1, 2, 1, 1, 1), (2, 2, 1, 1, 1, 2, 1, 1)]
B = [(-1, 1, -1, 2, 0, 1, -1, 2), (1, -2, 2, -1, 1, -2, 1, -1)]
x = (2, -1, 1, -1, 1, -1, 1, -1)
K_CORTES = [3, 5]

E_A_DOC, E_B_DOC, E_X_DOC = (125 / 3, 3, 5 / 3), (11 / 3, 51, 7 / 3), (1, 41, 1)
D_AB_DOC, D_A_DOC, D_B_DOC = 61.22, 55.66, 10.43


def energias_de(senal):
    """Pasos 1 a 3: X(k) -> subbandas -> E = (E_1, E_2, E_3)."""
    X = core.espectro(senal)
    return core.energia(X, K_CORTES)


def comprobar(nombre, obtenido, esperado, tolerancia=0.005):
    ok = np.allclose(obtenido, esperado, atol=tolerancia)
    print(f"{'OK   ' if ok else 'FALLA'} {nombre}: {np.round(obtenido, 2)}   (documento: {np.round(esperado, 2)})")
    return ok


def main():
    E_promedio = {"A": core.energia_promedio([energias_de(s) for s in A]),
                  "B": core.energia_promedio([energias_de(s) for s in B])}
    E_x = energias_de(x)
    d = {clase: core.diferencia_euclidiana(E_x, E_clase)
         for clase, E_clase in E_promedio.items()}
    tipo = min(d, key=d.get)

    resultados = [
        comprobar("E_A", E_promedio["A"], E_A_DOC),
        comprobar("E_B", E_promedio["B"], E_B_DOC),
        comprobar("||E_A - E_B||", core.diferencia_euclidiana(E_promedio["A"], E_promedio["B"]), D_AB_DOC),
        comprobar("E_x", E_x, E_X_DOC),
        comprobar("d_A", d["A"], D_A_DOC),
        comprobar("d_B", d["B"], D_B_DOC),
    ]
    print(f"{'OK   ' if tipo == 'B' else 'FALLA'} la señal x se reconoce como tipo {tipo}   (documento: B)")
    return all(resultados) and tipo == "B"


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
