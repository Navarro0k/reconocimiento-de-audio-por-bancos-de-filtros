import numpy as np

class MovingAverageFilter:
    def __init__(self, M: int = 5):
        if not isinstance(M, int) or M < 1:
            raise ValueError("M debe ser un entero positivo (M >= 1).")
        self.M = M

    def apply(self, x: np.ndarray, fs: int = None) -> np.ndarray:
        x = np.asarray(x).flatten().astype(float)
        N = x.size

        if N == 0:
            return x

        if self.M >= N:
            raise ValueError(f"M ({self.M}) debe ser menor que N ({N}).")

        y = np.zeros(N) # Inicializar un arreglo del tamaño correcto en lugar de []

        for n in range(N):
            suma = 0.0
            for k in range(self.M):
                if n - k >= 0:
                    suma += x[n - k]

            y[n] = suma / self.M

        return y