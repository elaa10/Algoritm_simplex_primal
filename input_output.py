import numpy as np
from itertools import combinations

# --- CITIRE, INITIALIZARE SI AFISARE ---

def citeste_date(nume_fisier):
    """Citeste datele problemei si determina o baza primal admisibila."""

    with open(nume_fisier, 'r') as f:
        linii = f.readlines()

    m, n = map(int, linii[0].split())
    c = np.array(list(map(float, linii[1].split())))

    A = []
    for i in range(m):
        A.append(list(map(float, linii[2 + i].split())))
    A = np.array(A)

    b = np.array(list(map(float, linii[2 + m].split())))

    # 1. Incercam intai sa gasim o baza canonica (rapid, ideal)
    baza = gaseste_baza_canonica(A)

    # 2. Daca nu exista baza canonica, cautam automat orice b.p.a.
    if baza is None:
        print("[!] Nu am gasit baza canonica in A. Cautam automat o baza primal admisibila...")
        baza = gaseste_baza_primal_admisibila(A, b)

        if baza is None:
            print("[X] Nu exista nicio baza primal admisibila pentru aceasta problema.")
            return m, n, c, A, b, None
        print(f"[OK] Baza primal admisibila gasita: {[f'A{j+1}' for j in baza]}")
    else:
        print(f"[OK] Baza canonica gasita: {[f'A{j+1}' for j in baza]}")

    return m, n, c, A, b, baza


def gaseste_baza_canonica(A):
    """
    Cauta indecsii bazei canonice initiale (matricea identitate I_m) in A.
    Returneaza lista cu indecsii coloanelor in ordinea liniilor, sau None daca nu exista.
    """
    m, n = A.shape
    baza = [-1] * m

    for j in range(n):
        coloana = A[:, j]
        # Verificam daca pe coloana exista un singur element nenul si acela este exact 1
        if np.count_nonzero(coloana) == 1 and np.max(coloana) == 1.0:
            index_linie = int(np.argmax(coloana))
            # Daca nu am atribuit deja o variabila pentru aceasta linie, o adaugam in baza
            if baza[index_linie] == -1:
                baza[index_linie] = j

    # Daca baza canonica nu este completa, returnam None
    if -1 in baza:
        return None

    return baza


def gaseste_baza_primal_admisibila(A, b, tol=1e-9):
    """
    Cauta automat o baza primal admisibila incercand toate combinatiile de m coloane.

    O submatrice B (m x m) formata din m coloane ale lui A este baza primal admisibila daca:
      1. B este inversabila (coloanele sunt liniar independente)
      2. B^(-1) * b >= 0 (toate coordonatele lui b in baza B sunt nenegative)

    Returneaza lista de indecsi (in ordinea liniilor matricei B^(-1)*A) sau None.

    NOTA: Complexitatea este C(n, m), deci fezabila doar pentru probleme mici.
    Pentru probleme mari ar fi nevoie de Metoda celor doua faze.
    """
    m, n = A.shape

    for combinatie in combinations(range(n), m):
        indecsi = list(combinatie)
        B = A[:, indecsi]

        # Verificam daca B este inversabila
        try:
            # Verificam rangul mai intai - mai stabil numeric decat det
            if np.linalg.matrix_rank(B) < m:
                continue
            B_inv = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            continue

        # Verificam daca baza este primal admisibila: B^(-1) * b >= 0
        b_in_baza = B_inv @ b
        if np.all(b_in_baza >= -tol):
            # Reordonam indecsii astfel incat baza sa fie in ordinea liniilor.
            # Fiecare coloana a bazei corespunde unei linii din B^(-1)*B = I.
            # Trebuie sa stabilim ordinea: pentru fiecare linie i din 0..m-1,
            # gasim coloana din baza care are 1 pe pozitia i in B^(-1)*B.
            # Cum B^(-1)*B = I, coloana j_k (a k-a din baza) corespunde liniei k.
            # Deci ordinea este chiar ordinea din `indecsi`.
            return indecsi

    return None


def initializare_tabel(m, n, c, A, b, baza):
    """
    Construieste tabelul simplex initial pentru o baza B oarecare.
    """
    non_baza = [j for j in range(n) if j not in baza]

    # Extragem submatricea B si calculam inversa
    B = A[:, baza]
    B_inv = np.linalg.inv(B)

    # Coordonatele lui b in baza B
    b_in_baza = B_inv @ b

    # Tabelul are (n - m + 1) linii si (m + 1) coloane
    T = np.zeros((n - m + 1, m + 1))

    # 1. Completam liniile variabilelor care nu sunt in baza
    #    Coordonatele lui A^j in baza B = B^(-1) * A^j
    for i, j_nb in enumerate(non_baza):
        T[i, :m] = B_inv @ A[:, j_nb]

    # 2. Completam linia lui b (coordonatele lui b in baza B)
    T[-1, :m] = b_in_baza

    # 3. Calculam coloana de evaluare (α_i0)
    c_B = c[baza]
    for i, j_nb in enumerate(non_baza):
        T[i, m] = np.dot(c_B, T[i, :m]) - c[j_nb]

    # 4. Valoarea functiei scop (α_00 )
    T[-1, m] = np.dot(c_B, b_in_baza)

    return T, non_baza


def afisare_tabel(T, baza, non_baza, iteratie):
    print(f"\n=== TABEL SIMPLEX: Iteratia {iteratie} ===")

    # Header-ul contine variabilele din baza si α0j
    header = "Var\t| " + "\t".join([f"A{baza[j] + 1}" for j in range(len(baza))]) + "\t| α0j"
    print(header)
    print("-" * 60)

    # Liniile contin variabilele care nu sunt in baza
    for i in range(len(non_baza)):
        nume_var = f"A{non_baza[i] + 1}"
        valori = "\t".join([f"{val:6g}" for val in T[i, :-1]])
        α0j = f"{T[i, -1]:6g}"
        print(f"{nume_var}\t| {valori}\t| {α0j}")

    print("-" * 60)

    # Afisam ultima linie (pentru vectorul b)
    valori_b = "\t".join([f"{val:6g}" for val in T[-1, :-1]])
    val_f = f"{T[-1, -1]:6g}"
    print(f"b\t| {valori_b}\t| {val_f}")