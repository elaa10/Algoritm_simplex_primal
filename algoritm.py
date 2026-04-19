import numpy as np

# --- GASESTE PIVOTUL SI SOLUTIA ---

def gaseste_rand_pivot(T):
    """
    Cauta max α0j pe ultima coloana (excluzand ultima linie care e b).
    Returneaza indexul randului sau -1 daca baza e optima.
    """
    col_α0j = T[:-1, -1]
    max_val = np.max(col_α0j)

    # Toleranta pentru erori float
    if max_val > 1e-9:
        return int(np.argmax(col_α0j))
    return -1


def gaseste_coloana_pivot(T, rand_pivot):
    """
    Testul raportului minim calculat pe coloanele bazei.
    Imparte linia 'b' la elementele strict pozitive din randul pivot ales.
    """
    m = T.shape[1] - 1
    raport_min = float('inf')
    col_pivot = -1

    for j in range(m):
        element = T[rand_pivot, j]
        if element > 1e-9:
            raport = T[-1, j] / element
            if raport < raport_min:
                raport_min = raport
                col_pivot = j

    return col_pivot


def pivotare(T, rand_pivot, col_pivot):
    """
    Aplica transformarile Gauss-Jordan.
    """
    rows, cols = T.shape
    T_nou = np.zeros_like(T)
    p = T[rand_pivot, col_pivot]

    for i in range(rows):
        for j in range(cols):
            if i == rand_pivot and j == col_pivot:
                # 1. Elementul pivot devine 1/pivot
                T_nou[i, j] = 1.0 / p
            elif i == rand_pivot:
                # 2. Linia pivot se imparte la OPUSUL pivotului (-p)
                T_nou[i, j] = -T[i, j] / p
            elif j == col_pivot:
                # 3. Coloana pivot se imparte la pivot (p)
                T_nou[i, j] = T[i, j] / p
            else:
                # 4. Regula dreptunghiului pentru restul elementelor
                T_nou[i, j] = T[i, j] - (T[i, col_pivot] * T[rand_pivot, j]) / p

    return T_nou