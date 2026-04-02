def factorial(n):
    if not isinstance(n, int):
        raise TypeError("El valor debe ser un entero.")
    if n < 0:
        raise ValueError("El factorial no está definido para números negativos.")
    resultado = 1
    for i in range(2, n + 1):
        resultado *= i
    return resultado

# Ejemplos
print(f"factorial(5) = {factorial(5)}")
print(f"factorial(0) = {factorial(0)}")