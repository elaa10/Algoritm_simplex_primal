import numpy as np


# --- CITIRE, INITIALIZARE SI AFISARE ---

"""

STRUCTURA FISIERULUI DE INPUT (input.txt)

Linia 1: m n -> (m = numarul de restrictii, n = numarul de variabile)
Linia 2: vectorul c -> (coeficientii functiei scop, separati prin spatiu)
Urmatoarele 'm' linii: matricea A -> (coeficientii restrictiilor)
Linia m + 3: vectorul b -> (termenii liberi)

"""

def citeste_date(nume_fisier="input.txt"):
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

    baza_initiala = gaseste_baza_canonica(A)

    return m, n, c, A, b, baza_initiala


def gaseste_baza_canonica(A):
    """
    Cauta automat indecsii bazei canonice initiale analizand matricea A.
    Returneaza o lista cu indecsii coloanelor din baza, in ordinea liniilor.
    """
    m, n = A.shape
    baza_initiala = [-1] * m  # Initializam un vector gol pentru baza

    for j in range(n):
        coloana = A[:, j]
        # Verificam daca pe coloana exista un singur element nenul si acela este exact 1
        if np.count_nonzero(coloana) == 1 and np.max(coloana) == 1.0:
            # Aflam pe ce linie se afla cifra 1
            index_linie = np.argmax(coloana)

            # Daca nu am atribuit deja o variabila pentru aceasta linie, o adaugam in baza
            if baza_initiala[index_linie] == -1:
                baza_initiala[index_linie] = j

    # Verificam daca programul a gasit o baza completa pentru toate cele 'm' linii
    if -1 in baza_initiala:
        raise ValueError("Eroare: Nu s-a putut identifica o baza canonica completa din matricea A!")

    return baza_initiala



def initializare_tabel(m, n, c, A, b, baza):
    """Construieste tabelul simplex initial, inclusiv linia de evaluare d0j."""

    # Tabelul va avea (m + 1) linii si (n + 1) coloane
    # Ultima linie e pentru d0j, ultima coloana e pentru b
    tabel = np.zeros((m + 1, n + 1))

    # Punem A si b in tabel
    tabel[:m, :n] = A
    tabel[:m, n] = b

    # Calculam coeficientii bazei c_B
    c_B = c[baza]

    # Calculam linia de evaluare: d_0j = (c_B * A^j) - c_j
    tabel[m, :n] = np.dot(c_B, A) - c

    # Valoarea curenta a functiei scop in coltul dreapta-jos: c_B * b
    tabel[m, n] = np.dot(c_B, b)

    return tabel


def afisare_tabel(tabel, baza, iteratie):
    """Printeaza tabelul frumos, asemanator cu cel din PDF."""
    m, cols = tabel.shape
    n = cols - 1

    print(f"\n=== TABEL SIMPLEX: Iteratia {iteratie} ===")
    header = "Baza\t| " + "\t".join([f"A{j + 1}" for j in range(n)]) + "\t| b"
    print(header)
    print("-" * 60)

    # Afisam liniile pentru baza curenta
    for i in range(m - 1):
        nume_baza = f"A{baza[i] + 1}"
        valori_linie = "\t".join([f"{val:6.2f}" for val in tabel[i, :n]])
        valoare_b = f"{tabel[i, n]:6.2f}"
        print(f"{nume_baza}\t| {valori_linie}\t| {valoare_b}")

    print("-" * 60)
    # Afisam linia de evaluare
    valori_evaluare = "\t".join([f"{val:6.2f}" for val in tabel[m - 1, :n]])
    val_scop = f"{tabel[m - 1, n]:6.2f}"
    print(f"d0j\t| {valori_evaluare}\t| {val_scop}")