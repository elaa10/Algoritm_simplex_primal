import numpy as np

# --- CITIRE, INITIALIZARE SI AFISARE ---

def citeste_date(nume_fisier):
    """Citeste datele problemei si genereaza automat baza."""

    with open(nume_fisier, 'r') as f:
        linii = f.readlines()

    m, n = map(int, linii[0].split())
    c = np.array(list(map(float, linii[1].split())))

    A = []
    for i in range(m):
        A.append(list(map(float, linii[2 + i].split())))
    A = np.array(A)

    b = np.array(list(map(float, linii[2 + m].split())))

    baza = gaseste_baza_canonica(A)

    return m, n, c, A, b, baza


def gaseste_baza_canonica(A):
    """
    Cauta automat indecsii bazei canonice initiale analizand matricea A.
    Returneaza o lista cu indecsii coloanelor din baza, in ordinea liniilor.
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
    # Verificam daca programul a gasit o baza completa
    if -1 in baza:
        raise ValueError("Eroare: Nu s-a putut identifica o baza canonica completa!")

    return baza


def initializare_tabel(m, n, c, A, b, baza):
    """Construieste tabelul simplex initial"""
    non_baza = [j for j in range(n) if j not in baza]

    # Tabelul are (n - m + 1) linii si (m + 1) coloane
    T = np.zeros((n - m + 1, m + 1))

    # 1. Completam liniile variabilelor care nu sunt in baza
    for i, j_nb in enumerate(non_baza):
        T[i, :m] = A[:, j_nb]

    # 2. Completam linia lui b
    T[-1, :m] = b

    # 3. Calculam coloana de evaluare (α0j)
    c_B = c[baza]
    for i, j_nb in enumerate(non_baza):
        T[i, m] = np.dot(c_B, A[:, j_nb]) - c[j_nb]

    # 4. Valoarea functiei scop (coltul dreapta-jos)
    T[-1, m] = np.dot(c_B, b)

    return T, non_baza


def afisare_tabel(T, baza, non_baza, iteratie):
    """Printeaza tabelul exact in formatul restrans din PDF."""

    print(f"\n=== TABEL SIMPLEX: Iteratia {iteratie} ===")

    # Header-ul contine variabilele din baza si α0j
    header = "Var\t|" + "".join([f"\tA{baza[j] + 1}" for j in range(len(baza))]) + "\t|\t α0j"
    print(header)
    print("-" * 60)

    # Liniile contin variabilele care nu sunt in baz
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