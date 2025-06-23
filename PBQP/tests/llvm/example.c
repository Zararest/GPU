// Compile with: gcc -O3 -march=native -fno-optimize-sibling-calls
double high_register_pressure() {
  // Initialize 30 variables (exceeds typical physical registers)
  double v0 = 1.0,  v1 = 1.1,  v2 = 1.2,  v3 = 1.3,  v4 = 1.4;
  double v5 = 1.5,  v6 = 1.6,  v7 = 1.7,  v8 = 1.8,  v9 = 1.9;
  double v10 = 2.0, v11 = 2.1, v12 = 2.2, v13 = 2.3, v14 = 2.4;
  double v15 = 2.5, v16 = 2.6, v17 = 2.7, v18 = 2.8, v19 = 2.9;
  double v20 = 3.0, v21 = 3.1, v22 = 3.2, v23 = 3.3, v24 = 3.4;
  double v25 = 3.5, v26 = 3.6, v27 = 3.7, v28 = 3.8, v29 = 3.9;

  // Long dependency chain forcing variables to stay live
  for (int i = 0; i < 100; i++) {
      // Independent computations (no dead stores)
      v0 = v0 * v1 + v2;  v1 = v1 * v2 + v3;
      v2 = v2 * v3 + v4;  v3 = v3 * v4 + v5;
      v4 = v4 * v5 + v6;  v5 = v5 * v6 + v7;
      v6 = v6 * v7 + v8;  v7 = v7 * v8 + v9;
      v8 = v8 * v9 + v10; v9 = v9 * v10 + v11;
      v10 = v10 * v11 + v12; v11 = v11 * v12 + v13;
      v12 = v12 * v13 + v14; v13 = v13 * v14 + v15;
      v14 = v14 * v15 + v16; v15 = v15 * v16 + v17;
      v16 = v16 * v17 + v18; v17 = v17 * v18 + v19;
      v18 = v18 * v19 + v20; v19 = v19 * v20 + v21;
      v20 = v20 * v21 + v22; v21 = v21 * v22 + v23;
      v22 = v22 * v23 + v24; v23 = v23 * v24 + v25;
      v24 = v24 * v25 + v26; v25 = v25 * v26 + v27;
      v26 = v26 * v27 + v28; v27 = v27 * v28 + v29;
      v28 = v28 * v29 + v0;  v29 = v29 * v0 + v1;
  }

  // Merge results to ensure all are live at end
  return v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9 +
         v10 + v11 + v12 + v13 + v14 + v15 + v16 + v17 + v18 + v19 +
         v20 + v21 + v22 + v23 + v24 + v25 + v26 + v27 + v28 + v29;
}

int main() {
  return (int)high_register_pressure();
}