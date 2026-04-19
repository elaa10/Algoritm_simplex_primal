# --- PERSOANA 2: PIVOTARE ȘI OPERAȚII GAUSS-JORDAN ---
import numpy as np

def gaseste_coloana_pivot(tabel):
    """
    Caută primul element strict pozitiv pe ultima linie (linia d0j).
    Dacă toate sunt <= 0, baza e optimă (returnează -1).
    """
    m, n_plus_1 = tabel.shape
    n = n_plus_1 - 1
    linia_evaluare = tabel[m - 1, :n]

    max_val = np.max(linia_evaluare)
    # Folosim o toleranță mică (1e-9) pentru a evita erorile de tip floating-point
    if max_val > 1e-9:
        # Alege coloana cu cel mai mare element strict pozitiv (cum e uzual)
        return int(np.argmax(linia_evaluare))
    return -1  # Baza este optimă


def gaseste_linia_pivot(tabel, col_pivot):
    """
    Efectuează testul raportului minim: împarte coloana b la coloana pivot.
    Sunt luate în calcul DOAR elementele strict pozitive din coloana pivot.
    """
    m, n_plus_1 = tabel.shape
    num_restr = m - 1

    raport_min = float('inf')
    linie_pivot = -1

    for i in range(num_restr):
        element_pivot = tabel[i, col_pivot]
        if element_pivot > 1e-9:  # Luăm doar numerele strict pozitive
            raport = tabel[i, -1] / element_pivot
            if raport < raport_min:
                raport_min = raport
                linie_pivot = i

    return linie_pivot


def actualizare_gauss_jordan(tabel, linie_pivot, col_pivot):
    """
    Aplică transformările Gauss-Jordan pe tot tabelul.
    """
    m, cols = tabel.shape
    noul_tabel = np.zeros_like(tabel)
    element_pivot_val = tabel[linie_pivot, col_pivot]

    # 1. Linia pivot se împarte la valoarea pivotului
    noul_tabel[linie_pivot, :] = tabel[linie_pivot, :] / element_pivot_val

    # 2. Toate celelalte linii (inclusiv linia de evaluare) se transformă prin Regula Dreptunghiului
    for i in range(m):
        if i != linie_pivot:
            factor_anulare = tabel[i, col_pivot]
            # Noua_linie = Linia_Veche - Linia_Pivot_Noua * Elementul_pe_coloana_pivot_vechi
            noul_tabel[i, :] = tabel[i, :] - (noul_tabel[linie_pivot, :] * factor_anulare)

    return noul_tabel