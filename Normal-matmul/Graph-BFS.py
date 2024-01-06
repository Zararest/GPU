import subprocess as subproc
import matplotlib.pyplot as plt
import matplotlib as matplotlib
import numpy as np

def run_BFS(matmul_path, mode, parameters):
  result = []
  for size in parameters:
    print('\tsize:', size)
    time_output = subproc.run([matmul_path, 
                               mode,
                               '--size', str(size),
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
  BFS_path = './build/BFS'
  max_size = 4096 * 4
  min_size = 100
  num_of_measures = 10

  N_array = np.linspace(min_size, max_size, num_of_measures)
  fig, ax = plt.subplots(figsize=(10, 7))

  print('GPU:')
  time_array_GPU = run_BFS(BFS_path, '--GPU', N_array)
  print('CPU')
  time_array_CPU = run_BFS(BFS_path, '--CPU', N_array)
  print('GPUNoCopy')
  time_array_GPU_no_copy = run_BFS(BFS_path, '--GPUNoCopy', N_array)

  ax.plot(N_array, time_array_GPU)  
  ax.scatter(N_array, time_array_GPU, marker='+', label='GPU с копированием графа')

  ax.plot(N_array, time_array_GPU_no_copy)  
  ax.scatter(N_array, time_array_GPU_no_copy, marker='*', 
             label='GPU без копирования графа')

  ax.plot(N_array, time_array_CPU)  
  ax.scatter(N_array, time_array_CPU, marker='o', label='на CPU')

  ax.set_xlabel('Размер смежных сторон')
  ax.set_ylabel('Время поиска BFS[мс]')
  ax.set_title('Поиск BFS')
  ax.legend()
  ax.grid()
  plt.show()

if __name__ == '__main__':
  main()