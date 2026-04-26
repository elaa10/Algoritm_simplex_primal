# --- AFISARE ---

FISIER_INTRARE = "input2.txt"

from algoritm import *
from input_output import *


def extrage_solutia(T, baza, non_baza, n):
    """Reconstruieste solutia x* din tabel"""

    solutie = np.zeros(n)

    # Variabilele din baza primesc valorile din ultima linie (b)
    for j in range(len(baza)):
        solutie[baza[j]] = T[-1, j]

    valoare_optima = T[-1, -1]

    print("\n" + "=" * 40)
    print("[OK] ALGORITMUL S-A OPRIT. SOLUTIE OPTIMA GASITA!")
    solutie_text = ", ".join([f"{float(val):g}" for val in solutie])
    print(f"Vectorul solutie optima: X = [{solutie_text}]")
    print(f"Valoarea optima (minima) a functiei scop: α00 = {valoare_optima:g}")
    print("=" * 40 + "\n")


def ruleaza_simplex():
    """Functia principala care leaga toate modulele."""

    m, n, c, A, b, baza = citeste_date(FISIER_INTRARE)
    T, non_baza = initializare_tabel(m, n, c, A, b, baza)

    iteratie = 0
    afisare_tabel(T, baza, non_baza, iteratie)

    while True:
        # 1. Cautam cine intra in baza (randul pivot)
        rand_pivot = gaseste_rand_pivot(T)

        # Conditia de oprire: Baza este optima
        if rand_pivot == -1:
            extrage_solutia(T, baza, non_baza, n)
            break

        # 2. Cautam cine iese din baza (coloana pivot)
        col_pivot = gaseste_coloana_pivot(T, rand_pivot)

        # Conditia de oprire: Problema nemarginita
        if col_pivot == -1:
            print(f"\n[!] Eroare: S-a ales randul A{non_baza[rand_pivot] + 1}, dar elementele sunt <= 0.")
            print("Problema este NEMARGINITA. Nu se poate gasi un optim finit.")
            break

        print(
            f"\n>>> Intra in baza A{non_baza[rand_pivot] + 1} (iese A{baza[col_pivot] + 1}). Pivot: {T[rand_pivot, col_pivot]:g}")

        # 3. Facem matematica din algoritmul condensat
        T = pivotare(T, rand_pivot, col_pivot)

        # 4. Facem rocada intre variabila care a intrat si cea care a iesit
        var_care_intra = non_baza[rand_pivot]
        var_care_iese = baza[col_pivot]

        non_baza[rand_pivot] = var_care_iese
        baza[col_pivot] = var_care_intra

        iteratie += 1
        afisare_tabel(T, baza, non_baza, iteratie)


if __name__ == "__main__":
    ruleaza_simplex()