import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Fungsi Interpolasi
def interpolasi_linear(x, y, x_interp):
    i = np.searchsorted(x, x_interp) - 1
    return y[i] + (y[i + 1] - y[i]) / (x[i + 1] - x[i]) * (x_interp - x[i])

def interpolasi_kuadratik(x, y, x_interp):
    # Cari 3 titik terdekat dengan x_interp
    idx = np.argsort(np.abs(x - x_interp))[:3]
    x_dekat = x[idx]
    y_dekat = y[idx]

    # Hitung polinom Lagrange orde 2
    L0 = ((x_interp - x_dekat[1]) * (x_interp - x_dekat[2])) / ((x_dekat[0] - x_dekat[1]) * (x_dekat[0] - x_dekat[2]))
    L1 = ((x_interp - x_dekat[0]) * (x_interp - x_dekat[2])) / ((x_dekat[1] - x_dekat[0]) * (x_dekat[1] - x_dekat[2]))
    L2 = ((x_interp - x_dekat[0]) * (x_interp - x_dekat[1])) / ((x_dekat[2] - x_dekat[0]) * (x_dekat[2] - x_dekat[1]))

    # Hitung nilai interpolasi
    return L0 * y_dekat[0] + L1 * y_dekat[1] + L2 * y_dekat[2]

def interpolasi_kubik(x, y, x_interp):
    # Find the 4 nearest points to x_interp
    idx = np.argsort(np.abs(x - x_interp))[:4]
    x_dekat = x[idx]
    y_dekat = y[idx]

    # Calculate the cubic Lagrange polynomial coefficients
    a = (x_dekat[1] - x_interp) * (x_dekat[2] - x_interp) * (x_dekat[3] - x_interp) / ((x_dekat[0] - x_dekat[1]) * (x_dekat[0] - x_dekat[2]) * (x_dekat[0] - x_dekat[3]))
    b = (x_dekat[0] - x_interp) * (x_dekat[2] - x_interp) * (x_dekat[3] - x_interp) / ((x_dekat[1] - x_dekat[0]) * (x_dekat[1] - x_dekat[2]) * (x_dekat[1] - x_dekat[3]))
    c = (x_dekat[0] - x_interp) * (x_dekat[1] - x_interp) * (x_dekat[3] - x_interp) / ((x_dekat[2] - x_dekat[0]) * (x_dekat[2] - x_dekat[1]) * (x_dekat[2] - x_dekat[3]))
    d = (x_dekat[0] - x_interp) * (x_dekat[1] - x_interp) * (x_dekat[2] - x_interp) / ((x_dekat[3] - x_dekat[0]) * (x_dekat[3] - x_dekat[1]) * (x_dekat[3] - x_dekat[2]))

    # Calculate the interpolated value
    return a * y_dekat[0] + b * y_dekat[1] + c * y_dekat[2] + d * y_dekat[3]

def interpolasi_lagrange(x, y, x_interp):
    n = len(x)
    result = 0.0
    for i in range(n):
        term = y[i]
        for j in range(n):
            if j != i:
                term *= (x_interp - x[j]) / (x[i] - x[j])
        result += term
    return result

def gauss_elimination(A, B):
    n = len(B)
    M = np.hstack([A, B.reshape(-1,1)])
    for i in range(n):
        max_row = np.argmax(abs(M[i:,i])) + i
        M[[i,max_row]] = M[[max_row,i]]
        for j in range(i+1, n):
            ratio = M[j,i] / M[i,i]
            M[j,i:] -= ratio * M[i,i:]
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (M[i,-1] - np.dot(M[i,i+1:n], x[i+1:n])) / M[i,i]
    return x

def gauss_jordan(A, B):
    n = len(B)
    M = np.hstack([A, B.reshape(-1,1)])
    for i in range(n):
        M[i] = M[i] / M[i,i]
        for j in range(n):
            if i != j:
                M[j] -= M[i] * M[j,i]
    return M[:,-1]

def hitung_interpolasi(metode, x_data, y_data, x_interp):
    x = np.array(x_data)
    y = np.array(y_data)

    if x_interp < min(x) or x_interp > max(x):
        return "Error: Nilai x yang diinterpolasi diluar rentang data."

    if metode == "Linear":
        return interpolasi_linear(x, y, x_interp)
    elif metode == "Kuadratik":
        return interpolasi_kuadratik(x, y, x_interp)
    elif metode == "Kubik":
        return interpolasi_kubik(x, y, x_interp)
    elif metode == "Lagrange":
        return interpolasi_lagrange(x, y, x_interp)

def main():
    st.set_page_config(page_title="Kalkulator Metode Numerik", layout="wide")
    st.title("🧮 Kalkulator Metode Numerik")
    st.write("Aplikasi ini membantu Anda dalam melakukan perhitungan interpolasi dan eliminasi menggunakan metode numerik.")

    # Sidebar yang Lebih Rapi
    with st.sidebar:
        st.subheader("🧮 Kalkulator Metode Numerik")
        metode = st.selectbox("Pilih Metode:", ["Linear", "Kuadratik", "Kubik", "Lagrange", "Gauss", "Gauss-Jordan"])
        st.write("___")

    # Input Data yang Dinamis
    if metode in ["Linear", "Kuadratik", "Kubik", "Lagrange"]:
        with st.form("input_interpolasi"):
            num_points = st.number_input("Jumlah Titik Data:", min_value=2, value=2)
            cols = st.columns(num_points)
            x_data = []
            y_data = []
            for i, col in enumerate(cols):
                with col:
                    x_data.append(st.number_input(f"x{i+1}:"))
                    y_data.append(st.number_input(f"y{i+1}:"))
            x_interp = st.number_input("Nilai x yang akan diinterpolasi:")
            submit = st.form_submit_button("Hitung Interpolasi")

        # Perhitungan dan Visualisasi
        if submit:
            x_data = np.array(x_data, dtype=float)  # Ubah ke float
            y_data = np.array(y_data, dtype=float)
            result = hitung_interpolasi(metode, x_data, y_data, x_interp)
            st.write(f"Hasil Interpolasi ({metode}): {result:.4f}")

            # Visualisasi (opsional)
            x_plot = np.linspace(min(x_data), max(x_data), 400)
            y_plot = [hitung_interpolasi(metode, x_data, y_data, x) for x in x_plot]

            plt.figure(figsize=(8, 6))
            plt.plot(x_data, y_data, 'o', label="Titik Data")
            plt.plot(x_plot, y_plot, label=f"Interpolasi {metode}")
            plt.plot(x_interp, result, 'ro', label="Hasil Interpolasi")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title(f"Interpolasi {metode}")
            plt.legend()
            st.pyplot(plt)

    elif metode in ["Gauss", "Gauss-Jordan"]:
        with st.form("input_eliminasi"):
            num_eq = st.number_input("Jumlah Persamaan:", min_value=2, value=2)
            cols = st.columns(num_eq + 1)
            A = []
            B = []
            for i in range(num_eq):
                row_a = []
                for j in range(num_eq):
                    with cols[j]:
                        row_a.append(st.number_input(f"A[{i+1},{j+1}]:"))
                A.append(row_a)
                with cols[-1]:
                    B.append(st.number_input(f"B[{i+1}]:"))
            A = np.array(A)
            B = np.array(B)
            submit = st.form_submit_button("Hitung Eliminasi")

        if submit:
            if metode == "Gauss":
                result = gauss_elimination(A, B)
            elif metode == "Gauss-Jordan":
                result = gauss_jordan(A, B)

            # Format hasil penyelesaian
            hasil_str = ", ".join([f"(x{i+1}: {val:.4f})" for i, val in enumerate(result)])
            st.write(f"Hasil Penyelesaian ({metode}): [{hasil_str}]")

if __name__ == "__main__":
    main()
