import subprocess as subproc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as matplotlib
import numpy as np

def run_matmul(matmul_path, mode, parameters):
  result = []
  for size in parameters:
    print('\tsize:', size)
    time_output = subproc.run([matmul_path, 
                                  mode,
                                  '--matrix',
                                  '2048',
                                  '2048',
                                  str(size),
                                  '--only-time'], stdout=subproc.PIPE)
    time = 0
    try:
        time = int(time_output.stdout) 
    except ValueError:
        print(time_output.stdout, 'is not an int')
        return
    result.append(time)
  return result

def main():
  # Эти переменные можно менять
  CPU_measures_num = 0
  matmul_path = './build/MatMul'
  max_size = 4096 * 2
  min_size = 100
  num_of_measures = 30

  N_array = np.linspace(min_size, max_size, num_of_measures)
  N_array_CPU = N_array[:CPU_measures_num]
  fig, ax = plt.subplots(figsize=(10, 7))

  print('GPU:')
  time_array_tiled = run_matmul(matmul_path, '--tiled', N_array)
  print('CPU')
  time_array_CPU = run_matmul(matmul_path, '--CPU', N_array_CPU)

  ax.plot(N_array, time_array_tiled)  
  ax.scatter(N_array, time_array_tiled, marker='+', label='с shared памятью')

  ax.plot(N_array_CPU, time_array_CPU)  
  ax.scatter(N_array_CPU, time_array_CPU, marker='o', label='на CPU')

  ax.set_xlabel('Размер смежных сторон')
  ax.set_ylabel('Время умножения[мс]')
  ax.set_title('Умножение матриц')
  ax.legend()
  ax.grid()
  plt.show()

if __name__ == '__main__':
    main()