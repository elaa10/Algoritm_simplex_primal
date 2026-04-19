# --- PERSOANA 3: ORCHESTRAREA ȘI EXTRAGEREA SOLUȚIEI ---

from algoritm import *
from input_output import *

def extrage_solutia(tabel, baza, n):
    """Reconstruiește soluția x* și afișează minimul funcției."""
    m_restr = len(baza)
    # Vector plin cu 0 la început
    solutie = np.zeros(n)

    # Variabilele din bază iau valorile din ultima coloană (coloana b)
    for i in range(m_restr):
        solutie[baza[i]] = tabel[i, -1]

    valoare_optima = tabel[-1, -1]

    print("\n" + "=" * 40)
    print("✓ ALGORITMUL S-A OPRIT. SOLUȚIE OPTIMĂ GĂSITĂ!")
    solutie_text = ", ".join([f"{float(val):g}" for val in solutie])
    print(f"Vectorul soluție optimă: x* = [{solutie_text}]")
    print(f"Valoarea funcției scop (min): f(x*) = {valoare_optima:.2f}")
    print("=" * 40 + "\n")


def ruleaza_simplex():
    """Funcția principală care leagă toate părțile."""
    # Persoana 3 folosește funcțiile Persoanei 1
    m, n, c, A, b, baza = citeste_date("input.txt")
    tabel = initializare_tabel(m, n, c, A, b, baza)

    iteratie = 0
    afisare_tabel(tabel, baza, iteratie)

    # Baza algoritmului (Loop-ul iterativ)
    while True:
        # Persoana 3 folosește funcțiile Persoanei 2
        col_pivot = gaseste_coloana_pivot(tabel)

        # Condiția de oprire 1: Baza este optimă
        if col_pivot == -1:
            extrage_solutia(tabel, baza, n)
            break

        linie_pivot = gaseste_linia_pivot(tabel, col_pivot)

        # Condiția de oprire 2: Problema nemărginită (niciun element > 0 pe coloana pivot)
        if linie_pivot == -1:
            print(f"\n⚠️ Eroare: S-a ales coloana A{col_pivot + 1}, dar toate elementele sunt <= 0.")
            print("Conform teoriei, funcția este NEMĂRGINITĂ inferior. Problema nu are optim finit.")
            break

        # Dacă totul e ok, afișăm pivotul și trecem la pasul următor
        print(
            f"\n>>> Se introduce în bază A{col_pivot + 1} (iese A{baza[linie_pivot] + 1}). Pivotul este: {tabel[linie_pivot, col_pivot]:.2f}")

        # Actualizăm tabelul și vectorul bazei curente
        tabel = actualizare_gauss_jordan(tabel, linie_pivot, col_pivot)
        baza[linie_pivot] = col_pivot

        iteratie += 1
        afisare_tabel(tabel, baza, iteratie)


# Declanșatorul programului
if __name__ == "__main__":
    ruleaza_simplex()